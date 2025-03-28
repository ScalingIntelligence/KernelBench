import fastapi
import uvicorn
import tempfile
import os
import shutil
from fastapi import UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Import the relevant modules directly
from scripts.run_and_check import evaluate_single_sample_src
from scripts.generate_baseline_time import measure_program_time
from src.utils import read_file, set_gpu_arch
import torch

# Define the response model
class BenchmarkResult(BaseModel):
    compiled: bool
    correctness: bool
    ref_exec_eager_time_ms: Optional[float] = None
    ref_exec_compile_time_ms: Optional[float] = None
    kernel_exec_time_ms: Optional[float] = None
    speedup_vs_eager: Optional[float] = None
    speedup_vs_compile: Optional[float] = None
    metadata: Dict[str, Any]
    error: Optional[str] = None

app = fastapi.FastAPI()

@app.post("/benchmark", response_model=BenchmarkResult)
async def run_benchmark(
    ref_file: UploadFile = File(...),
    kernel_file: UploadFile = File(...),
    gpu_arch: List[str] = ["Ada"],
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    verbose: bool = False
):
    # Create temporary files for the uploaded code
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="wb") as ref_tmp, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="wb") as kernel_tmp:
        try:
            # Save uploaded file contents to temporary files
            shutil.copyfileobj(ref_file.file, ref_tmp)
            shutil.copyfileobj(kernel_file.file, kernel_tmp)

            # Ensure files are flushed and closed before script access
            ref_path = ref_tmp.name
            kernel_path = kernel_tmp.name
        finally:
            ref_file.file.close()
            kernel_file.file.close()

    try:
        # Read the contents of the files
        ref_arch_src = read_file(ref_path)
        kernel_src = read_file(kernel_path)
        
        # Set up GPU architecture
        set_gpu_arch(gpu_arch)
        
        # Default device
        device = torch.device("cuda:0")
        
        # Prepare configs
        configs = {
            "num_correct_trials": num_correct_trials,
            "num_perf_trials": num_perf_trials,
            "verbose": verbose,
            "measure_performance": True,
            "build_dir_prefix": "server_builds",
            "clear_cache": False
        }
        
        # Evaluate kernel against reference
        kernel_eval_result = evaluate_single_sample_src(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            configs=configs,
            device=device
        )
        
        # Measure reference times
        ref_time_eager_result = measure_program_time(
            ref_arch_name="Reference Program", 
            ref_arch_src=ref_arch_src, 
            num_trials=num_perf_trials,
            use_torch_compile=False,
            device=device
        )
        
        ref_time_compile_result = measure_program_time(
            ref_arch_name="Reference Program", 
            ref_arch_src=ref_arch_src, 
            num_trials=num_perf_trials,
            use_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_options="default",
            device=device
        )
        
        # Extract values
        kernel_exec_time = kernel_eval_result.runtime
        ref_exec_eager_time = ref_time_eager_result.get("mean", None)
        ref_exec_compile_time = ref_time_compile_result.get("mean", None)
        
        # Calculate speedups
        speedup_vs_eager = None
        speedup_vs_compile = None
        
        if kernel_eval_result.correctness and kernel_exec_time and ref_exec_eager_time:
            speedup_vs_eager = ref_exec_eager_time / kernel_exec_time
            
        if kernel_eval_result.correctness and kernel_exec_time and ref_exec_compile_time:
            speedup_vs_compile = ref_exec_compile_time / kernel_exec_time
            
        # Prepare output summary
        raw_output = f"""
==============================
[Eval] Kernel eval result: {kernel_eval_result}
------------------------------
[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms
[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms
[Timing] Custom Kernel exec time: {kernel_exec_time} ms
------------------------------
"""
        if kernel_eval_result.correctness:
            raw_output += f"""
[Speedup] Speedup over eager: {speedup_vs_eager:.2f}x
[Speedup] Speedup over torch.compile: {speedup_vs_compile:.2f}x
"""
        else:
            raw_output += "[Speedup] Speedup Not Available as Kernel did not pass correctness"
        
        raw_output += "=============================="
            
        # Prepare the response
        response = BenchmarkResult(
            compiled=kernel_eval_result.compiled,
            correctness=kernel_eval_result.correctness,
            ref_exec_eager_time_ms=ref_exec_eager_time,
            ref_exec_compile_time_ms=ref_exec_compile_time,
            kernel_exec_time_ms=kernel_exec_time,
            speedup_vs_eager=speedup_vs_eager,
            speedup_vs_compile=speedup_vs_compile,
            metadata=kernel_eval_result.metadata or {},

        )
        print(raw_output)        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during benchmarking: {str(e)}"
        )
    finally:
        # Clean up temporary files
        if 'ref_path' in locals() and os.path.exists(ref_path):
            os.remove(ref_path)
        if 'kernel_path' in locals() and os.path.exists(kernel_path):
            os.remove(kernel_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 