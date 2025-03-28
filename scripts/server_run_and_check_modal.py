import os
import shutil
import tempfile
from typing import Dict, List, Optional, Any
import sys
import traceback
import importlib.util
import time

import weave
import torch
import modal
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import project-specific modules
# Assuming 'src' is in the Python path
try:
    from src import eval as kernel_eval
    from src import utils as kernel_utils
except ImportError as e:
    print(f"[ERROR] Failed to import project modules at startup: {e}")
    # Decide how to handle this - exit, log, or proceed cautiously
    kernel_eval = None
    kernel_utils = None
    
# Create Modal app
app = modal.App("kernel-benchmark-server") # Still here

# GPU architecture mapping
gpu_arch_mapping = {
    "L40S": ["Ada"], 
    "H100": ["Hopper"], 
    "A100": ["Ampere"], 
    "L4": ["Ada"], 
    "T4": ["Turing"], 
    "A10G": ["Ampere"]
}

# Configure Modal image
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "fastapi==0.115.0",
        "uvicorn==0.27.1",
        "python-multipart==0.0.9",
        "pydantic==2.6.1",
        "aiofiles==23.2.1",  # For serving static files
        "weave"
    )
    .pip_install_from_requirements("requirements.txt")
    # Add source directories
    .add_local_python_source("scripts", "src")
    .add_local_dir("static", "/root/static")
)

# Define response models
class KernelExecResult(BaseModel):
    compiled: bool
    correctness: bool
    runtime: Optional[float] = None
    metadata: Dict[str, Any] = {}

class BenchmarkResult(BaseModel):
    kernel_result: KernelExecResult
    ref_exec_eager_time_ms: Optional[float] = None
    ref_exec_compile_time_ms: Optional[float] = None
    kernel_exec_time_ms: Optional[float] = None
    speedup_vs_eager: Optional[float] = None
    speedup_vs_compile: Optional[float] = None
    compile_time_ms: Optional[float] = None
    total_benchmark_time_ms: Optional[float] = None
    error: Optional[str] = None

@app.cls(image=image, gpu="L40S", scaledown_window=300, secrets=[modal.Secret.from_name("wandb-api-key")])
class BenchmarkService:
    def __init__(self):
        pass
        
    @weave.op()
    def evaluate_single_sample_src(self, ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> KernelExecResult:
        """Evaluate a single sample source code against a reference source code"""
        # Check if kernel_eval was imported successfully
        if kernel_eval is None:
            print("[ERROR] src.eval module not available.")
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"import_error": "Failed to import src.eval at startup"}
            )
            
        try:
            print(f"[DEBUG] Python paths: {sys.path}")
            
            kernel_hash = str(hash(kernel_src))
            build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
            
            if configs["clear_cache"]:
                print(f"[INFO] Clearing cache for build directory: {build_dir}")
                shutil.rmtree(build_dir, ignore_errors=True)
            
            try:
                eval_result = kernel_eval.eval_kernel_against_ref(
                    original_model_src=ref_arch_src,
                    custom_model_src=kernel_src,
                    measure_performance=configs["measure_performance"],
                    verbose=configs["verbose"],
                    num_correct_trials=configs["num_correct_trials"],
                    num_perf_trials=configs["num_perf_trials"],
                    build_dir=build_dir,
                    device=device
                )
                return KernelExecResult(
                    compiled=eval_result.compiled,
                    correctness=eval_result.correctness,
                    runtime=eval_result.runtime,
                    metadata=eval_result.metadata or {}
                )
            except Exception as e:
                print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
                if "CUDA error" in str(e): 
                    metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                                "hardware": torch.cuda.get_device_name(device=device),
                                "device": str(device)
                                }
                else:
                    metadata = {"other_error": f"error: {str(e)}",
                                "hardware": torch.cuda.get_device_name(device=device),
                                "device": str(device)
                                }
                return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        except ImportError as e: # This catch might be less likely now, but keep for safety
            print(f"[ERROR] Import error during evaluation (unexpected): {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"import_error": f"Unexpected import error during eval: {str(e)}"}
            )
        except Exception as e:
            print(f"[ERROR] Unexpected error during evaluation: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"unexpected_error": str(e)}
            )
        
    @weave.op()
    def measure_program_time(self, ref_arch_name, ref_arch_src, num_trials, 
                            use_torch_compile=False, torch_compile_backend=None, 
                            torch_compile_options=None, gpu_arch=None):
        """Measure the execution time of a reference program"""
        # Removed imports: torch, numpy, importlib.util, sys, os, tempfile, src.utils
        
        # Check if kernel_utils was imported successfully
        if kernel_utils is None:
            print("[ERROR] src.utils module not available.")
            # Return an error structure or raise an exception
            return {
                "error": "src.utils module not available",
                "mean": None, "std": None, "min": None, "max": None, "median": None
            }
            
        # Setup
        if gpu_arch:
            kernel_utils.set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create temporary module
        temp_dir = tempfile.mkdtemp()
        ref_module_path = os.path.join(temp_dir, "ref_module.py")
        
        with open(ref_module_path, "w") as f:
            f.write(ref_arch_src)
        
        # Load reference module
        spec = importlib.util.spec_from_file_location("ref_module", ref_module_path)
        ref_module = importlib.util.module_from_spec(spec)
        sys.modules["ref_module"] = ref_module
        spec.loader.exec_module(ref_module)
        
        # Create model instance
        if hasattr(ref_module, "get_init_inputs"):
            init_inputs = ref_module.get_init_inputs()
            init_inputs = [
                x if (isinstance(x, torch.Tensor) and x.device == device) 
                else (x.to(device) if isinstance(x, torch.Tensor) else x)
                for x in init_inputs
            ]
            ref_model = ref_module.Model(*init_inputs).to(device)
        else:
            ref_model = ref_module.Model().to(device)
        
        # Apply torch.compile if needed
        if use_torch_compile:
            if torch_compile_backend is not None:
                if torch_compile_options is not None and torch_compile_options != "default":
                    compile_options = {"mode": torch_compile_options} if torch_compile_options in ["max-autotune", "reduce-overhead"] else {}
                    ref_model = torch.compile(ref_model, backend=torch_compile_backend, options=compile_options)
                else:
                    ref_model = torch.compile(ref_model, backend=torch_compile_backend)
            else:
                ref_model = torch.compile(ref_model)
        
        # Generate inputs
        if hasattr(ref_module, "get_inputs"):
            inputs = ref_module.get_inputs()
            inputs = [
                x if (isinstance(x, torch.Tensor) and x.device == device) 
                else (x.to(device) if isinstance(x, torch.Tensor) else x)
                for x in inputs
            ]
        elif hasattr(ref_module, "INPUT_SHAPE"):
            input_shape = ref_module.INPUT_SHAPE
            if isinstance(input_shape, tuple):
                inputs = (torch.randn(input_shape, device=device),)
            elif isinstance(input_shape, list):
                inputs = tuple(torch.randn(shape, device=device) for shape in input_shape)
            else:
                raise ValueError(f"Invalid INPUT_SHAPE: {input_shape}")
        else:
            # Infer inputs from model
            if hasattr(ref_model, "forward"):
                argcount = ref_model.forward.__code__.co_argcount
                inputs = tuple(torch.randn(1, 128, device=device) for _ in range(argcount - 1))
            else:
                raise ValueError("Could not determine appropriate inputs for the model")
        
        # Warmup
        for _ in range(10):
            ref_model(*inputs)
        
        # Timing
        torch.cuda.synchronize()
        times = []
        for _ in range(num_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            ref_model(*inputs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        # Clean up
        try:
            os.remove(ref_module_path)
            os.rmdir(temp_dir)
        except OSError:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Calculate statistics
        times = np.array(times)
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "median": float(np.median(times)),
        }
        
    @modal.method()
    def run_benchmark(self, ref_arch_src: str, kernel_src: str, 
                      gpu_type: str = "L40S", 
                      num_correct_trials: int = 5, 
                      num_perf_trials: int = 100,
                      verbose: bool = False):
        """Run a complete benchmark of kernel vs reference implementation"""
        print(f"[DEBUG] Starting benchmark on GPU: {gpu_type}")
        
        import time
        start_time = time.time()
        
        # Check if kernel_utils was imported successfully
        if kernel_utils is None:
             print("[ERROR] src.utils module not available.")
             return BenchmarkResult(
                 kernel_result=KernelExecResult(compiled=False, correctness=False),
                 error="src.utils module not available"
             )
             
        try:
            # Get GPU architecture
            gpu_arch = gpu_arch_mapping.get(gpu_type, ["Ada"])
            print(f"[DEBUG] Using GPU architecture: {gpu_arch}")
            
            # Removed from src import utils as kernel_utils
            kernel_utils.set_gpu_arch(gpu_arch)
            
            # Default device 
            device = torch.device("cuda:0")
            print(f"[DEBUG] Using device: {device}")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                print(f"[DEBUG] CUDA is available. Device count: {torch.cuda.device_count()}")
                print(f"[DEBUG] Current device: {torch.cuda.current_device()}")
                print(f"[DEBUG] Device name: {torch.cuda.get_device_name(device)}")
            else:
                print(f"[WARNING] CUDA is not available. Using CPU.")
            
            # Config dictionary
            configs = {
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "verbose": verbose,
                "measure_performance": True,
                "build_dir_prefix": "api_builds",
                "clear_cache": False
            }
            print(f"[DEBUG] Using configs: {configs}")
            
            try:
                # Time the compilation specifically
                compile_start_time = time.time()
                kernel_result = self.evaluate_single_sample_src(
                    ref_arch_src=ref_arch_src,
                    kernel_src=kernel_src,
                    configs=configs,
                    device=device
                )
                compile_time = (time.time() - compile_start_time) * 1000  # Convert to ms
                
                # Evaluate kernel
                print(f"[DEBUG] Evaluating kernel against reference...")
                kernel_exec_time = kernel_result.runtime
                print(f"[DEBUG] Kernel execution time: {kernel_exec_time} ms")
                
                # Measure baseline time for PyTorch Eager
                print(f"[DEBUG] Measuring PyTorch Eager execution time...")
                ref_time_eager_result = self.measure_program_time(
                    ref_arch_name="Reference Program",
                    ref_arch_src=ref_arch_src,
                    num_trials=num_perf_trials,
                    use_torch_compile=False,
                    torch_compile_backend=None,
                    torch_compile_options=None,
                    gpu_arch=gpu_arch
                )
                ref_exec_eager_time = ref_time_eager_result.get("mean", None)
                print(f"[DEBUG] PyTorch Eager execution time: {ref_exec_eager_time} ms")
                
                # Measure Torch Compile time
                print(f"[DEBUG] Measuring PyTorch Compiled execution time...")
                ref_time_compile_result = self.measure_program_time(
                    ref_arch_name="Reference Program",
                    ref_arch_src=ref_arch_src,
                    num_trials=num_perf_trials,
                    use_torch_compile=True,
                    torch_compile_backend="inductor",
                    torch_compile_options="default",
                    gpu_arch=gpu_arch
                )
                ref_exec_compile_time = ref_time_compile_result.get("mean", None)
                print(f"[DEBUG] PyTorch Compiled execution time: {ref_exec_compile_time} ms")
                
                # Calculate speedups
                speedup_vs_eager = None
                speedup_vs_compile = None
                
                if kernel_result.correctness and kernel_exec_time and ref_exec_eager_time:
                    speedup_vs_eager = ref_exec_eager_time / kernel_exec_time
                    print(f"[DEBUG] Speedup vs Eager: {speedup_vs_eager}x")
                    
                if kernel_result.correctness and kernel_exec_time and ref_exec_compile_time:
                    speedup_vs_compile = ref_exec_compile_time / kernel_exec_time
                    print(f"[DEBUG] Speedup vs Compiled: {speedup_vs_compile}x")
                
                # Round all float values to 2 decimal places
                if ref_exec_eager_time:
                    ref_exec_eager_time = round(ref_exec_eager_time, 2)
                if ref_exec_compile_time:
                    ref_exec_compile_time = round(ref_exec_compile_time, 2)
                if kernel_exec_time:
                    kernel_exec_time = round(kernel_exec_time, 2)
                if speedup_vs_eager:
                    speedup_vs_eager = round(speedup_vs_eager, 2)
                if speedup_vs_compile:
                    speedup_vs_compile = round(speedup_vs_compile, 2)
                
                # Calculate total benchmark time
                total_time = round((time.time() - start_time) * 1000, 2)  # Convert to ms and round
                compile_time = round(compile_time, 2)
                
                # Build response
                print(f"[DEBUG] Building response...")
                return BenchmarkResult(
                    kernel_result=kernel_result,
                    ref_exec_eager_time_ms=ref_exec_eager_time,
                    ref_exec_compile_time_ms=ref_exec_compile_time,
                    kernel_exec_time_ms=kernel_exec_time,
                    speedup_vs_eager=speedup_vs_eager,
                    speedup_vs_compile=speedup_vs_compile,
                    compile_time_ms=compile_time,
                    total_benchmark_time_ms=total_time
                )
            except Exception as e:
                print(f"[ERROR] Error during benchmark execution: {str(e)}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                return BenchmarkResult(
                    kernel_result=KernelExecResult(compiled=False, correctness=False),
                    error=f"Benchmark execution error: {str(e)}"
                )
        except Exception as e:
            print(f"[ERROR] Fatal error in run_benchmark: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return BenchmarkResult(
                kernel_result=KernelExecResult(compiled=False, correctness=False),
                error=str(e)
            )

    @modal.asgi_app()
    def fastapi_app(self):
        web_app = FastAPI(title="KernelBench Benchmarking API")
        
        # Add CORS middleware
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Determine if we're running locally or in Modal
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
        modal_static_dir = "/root/static"
        
        # Check both possible locations for static files
        if os.path.exists(static_dir):
            # Mount static files directory (local development)
            web_app.mount("/static", StaticFiles(directory=static_dir), name="static")
            
            @web_app.get("/")
            async def root():
                return FileResponse(os.path.join(static_dir, "index.html"))
        elif os.path.exists(modal_static_dir):
            # Mount static files directory (Modal environment)
            web_app.mount("/static", StaticFiles(directory=modal_static_dir), name="static")
            
            @web_app.get("/")
            async def root():
                return FileResponse(os.path.join(modal_static_dir, "index.html"))
        else:
            # Fallback for when static directory isn't available
            @web_app.get("/")
            async def root():
                return {
                    "name": "KernelBench Benchmarking API",
                    "version": "1.0.0",
                    "description": "API for benchmarking CUDA kernels against PyTorch reference implementations",
                    "endpoints": {
                        "/benchmark": "POST endpoint for benchmarking kernels",
                        "/status": "GET endpoint for checking server status"
                    }
                }
        
        @web_app.post("/benchmark", response_model=BenchmarkResult)
        async def benchmark_endpoint(
            ref_file: UploadFile = File(...),
            kernel_file: UploadFile = File(...),
            gpu_type: str = Form("L40S"),
            num_correct_trials: int = Form(5),
            num_perf_trials: int = Form(100),
            verbose: bool = Form(False)
        ):
            weave.init("gpu-server-modal")
            try:
                print(f"[DEBUG] Received benchmark request with GPU: {gpu_type}, trials: {num_correct_trials}/{num_perf_trials}")
                
                # Validate GPU type
                if gpu_type not in gpu_arch_mapping:
                    raise HTTPException(status_code=400, detail=f"Invalid GPU type. Must be one of: {list(gpu_arch_mapping.keys())}")
                
                # Read file contents
                try:
                    ref_content = await ref_file.read()
                    print(f"[DEBUG] Read reference file: {len(ref_content)} bytes")
                    kernel_content = await kernel_file.read()
                    print(f"[DEBUG] Read kernel file: {len(kernel_content)} bytes")
                    
                    ref_arch_src = ref_content.decode("utf-8")
                    kernel_src = kernel_content.decode("utf-8")
                except Exception as e:
                    print(f"[ERROR] Failed to read uploaded files: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Failed to read uploaded files: {str(e)}")
                
                # Run the benchmark with the specified GPU type
                try:
                    print(f"[DEBUG] Calling run_benchmark method")
                    result = self.run_benchmark.remote(
                        ref_arch_src=ref_arch_src,
                        kernel_src=kernel_src,
                        gpu_type=gpu_type,
                        num_correct_trials=num_correct_trials,
                        num_perf_trials=num_perf_trials,
                        verbose=verbose
                    )
                    print(f"[DEBUG] Benchmark completed successfully")
                    return result
                except Exception as e:
                    print(f"[ERROR] Benchmark execution failed: {str(e)}")
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")
            except Exception as e:
                print(f"[ERROR] Unexpected error in benchmark endpoint: {str(e)}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")
            finally:
                weave.finish()
        
        @web_app.get("/status")
        async def status():
            return {
                "status": "online",
                "gpu_types": list(gpu_arch_mapping.keys())
            }
        
        @web_app.get("/test_imports")
        async def test_imports():
            """Test endpoint to check if we can import the necessary modules"""
            result = {
                "python_version": sys.version,
                "sys_path": sys.path,
                "env_vars": dict(os.environ),
                "imports": {}
            }
            
            # Check modules that should have been imported at the top
            try:
                # Verify torch import
                if 'torch' in sys.modules:
                     result["imports"]["torch"] = {
                         "version": torch.__version__,
                         "cuda_available": torch.cuda.is_available(),
                         "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None
                     }
                else:
                    result["imports"]["torch"] = {"error": "torch module not loaded"}
            except Exception as e:
                 result["imports"]["torch"] = {"error": f"Error checking torch: {str(e)}"}

            # Verify src.eval import
            if kernel_eval is not None:
                 result["imports"]["src.eval"] = {"success": True}
            else:
                 result["imports"]["src.eval"] = {"error": "src.eval module failed to load at startup"}
                 
            # Verify src.utils import
            if kernel_utils is not None:
                 result["imports"]["src.utils"] = {"success": True}
            else:
                 result["imports"]["src.utils"] = {"error": "src.utils module failed to load at startup"}

            # Check for file existence
            result["files"] = {
                "static_local": os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")),
                "static_modal": os.path.exists("/root/static"),
                "requirements_txt": os.path.exists("requirements.txt")
            }
            
            return result
        
        return web_app

def main():
    # For local development, you can use:
    # modal serve scripts.server_run_and_check_modal
    print("Starting KernelBench API server...")
    print("Use 'modal serve scripts.server_run_and_check_modal' to start the development server")
    print("Use 'modal deploy scripts.server_run_and_check_modal' to deploy to production")

if __name__ == "__main__":
    main() 