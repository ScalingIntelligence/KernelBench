import os
import shutil
import importlib.util
import sys
import os
import tempfile

import modal
import pydra
import torch
import numpy as np

from pydra import REQUIRED, Config


from kernelbench.eval import KernelExecResult, eval_kernel_against_ref
from kernelbench.utils import read_file, set_gpu_arch


# from run_and_check.py
def evaluate_single_sample_src(
    ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device
) -> KernelExecResult:
    """
    Evaluate a single sample source code against a reference source code
    """

    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)

    if configs["clear_cache"]:  # fresh kernel build
        print(f"[INFO] Clearing cache for build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device,
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {
                "other_error": f"error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result


"""
Run a pair of (reference, solution) to check if solution is correct and compute speedup using Modal

Usage:
python3 scripts/run_and_check_modal.py ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add.py
"""

torch.set_printoptions(precision=4, threshold=10)
app = modal.App("run_and_check")
gpu_arch_mapping = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "A100": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"],
}


class ScriptConfig(Config):
    def __init__(self):
        # Required file paths
        self.ref_arch_src_path = REQUIRED  # Reference implementation
        self.kernel_src_path = REQUIRED  # Custom kernel implementation
        self.gpu = "L40S"  # GPU type for modal
        self.num_correct_trials = 5  # Number of trials for correctness
        self.num_perf_trials = 100  # Number of trials for performance
        self.timeout = 300  # Timeout for each trial
        self.verbose = False  # Verbose logging
        self.measure_performance = True  # Whether to measure performance
        self.build_dir_prefix = ""  # Custom build directory prefix
        self.clear_cache = False  # Whether to clear build cache
        self.gpu_arch = ["Ada"]  # Default GPU architecture

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"


# Configure Modal image
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install_from_requirements("scripts/server_requirements.txt")
    .add_local_python_source("kernelbench")
    .add_local_dir("KernelBench", "/KernelBench")
)


@app.cls(image=image)
class EvalFunc:
    @modal.method()
    def evaluate_single_sample_src_modal(
        self, ref_arch_src, kernel_src, configs, gpu_arch
    ):
        """Evaluate a single sample source code against a reference source code"""

        set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        eval_result = evaluate_single_sample_src(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            configs=configs,
            device=device,
        )

        return {
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "runtime": eval_result.runtime,
            "metadata": eval_result.metadata,
        }

    @modal.method()
    def measure_program_time(
        self,
        ref_arch_name,
        ref_arch_src,
        num_trials,
        use_torch_compile=False,
        torch_compile_backend=None,
        torch_compile_options=None,
        gpu_arch=None,
    ):
        """Measure the execution time of a reference program"""

        # Setup
        if gpu_arch:
            set_gpu_arch(gpu_arch)
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
                (
                    x
                    if (isinstance(x, torch.Tensor) and x.device == device)
                    else (x.to(device) if isinstance(x, torch.Tensor) else x)
                )
                for x in init_inputs
            ]
            ref_model = ref_module.Model(*init_inputs).to(device)
        else:
            ref_model = ref_module.Model().to(device)

        # Apply torch.compile if needed
        if use_torch_compile:
            if torch_compile_backend is not None:
                if (
                    torch_compile_options is not None
                    and torch_compile_options != "default"
                ):
                    compile_options = (
                        {"mode": torch_compile_options}
                        if torch_compile_options in ["max-autotune", "reduce-overhead"]
                        else {}
                    )
                    ref_model = torch.compile(
                        ref_model,
                        backend=torch_compile_backend,
                        options=compile_options,
                    )
                else:
                    ref_model = torch.compile(ref_model, backend=torch_compile_backend)
            else:
                ref_model = torch.compile(ref_model)

        # Generate inputs
        if hasattr(ref_module, "get_inputs"):
            inputs = ref_module.get_inputs()
            inputs = [
                (
                    x
                    if (isinstance(x, torch.Tensor) and x.device == device)
                    else (x.to(device) if isinstance(x, torch.Tensor) else x)
                )
                for x in inputs
            ]
        elif hasattr(ref_module, "INPUT_SHAPE"):
            input_shape = ref_module.INPUT_SHAPE
            if isinstance(input_shape, tuple):
                inputs = (torch.randn(input_shape, device=device),)
            elif isinstance(input_shape, list):
                inputs = tuple(
                    torch.randn(shape, device=device) for shape in input_shape
                )
            else:
                raise ValueError(f"Invalid INPUT_SHAPE: {input_shape}")
        else:
            # Infer inputs from model
            if hasattr(ref_model, "forward"):
                argcount = ref_model.forward.__code__.co_argcount
                inputs = tuple(
                    torch.randn(1, 128, device=device) for _ in range(argcount - 1)
                )
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


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):
    print("Running with config", config)

    # Read source files
    ref_arch_src = read_file(config.ref_arch_src_path)
    kernel_src = read_file(config.kernel_src_path)

    # Prepare GPU architecture settings
    gpu_arch = gpu_arch_mapping.get(config.gpu, config.gpu_arch)
    print(f"[INFO] Using GPU architecture: {gpu_arch}")

    # Start Evaluation
    with app.run():
        # Evaluate kernel against reference code
        print("[INFO] Evaluating kernel against reference code")
        kernel_eval_result_dict = EvalFunc.with_options(
            gpu=config.gpu
        )().evaluate_single_sample_src_modal.remote(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            configs=config.to_dict(),
            gpu_arch=gpu_arch,
        )

        # Convert dict back to KernelExecResult object
        kernel_eval_result = KernelExecResult(
            compiled=kernel_eval_result_dict["compiled"],
            correctness=kernel_eval_result_dict["correctness"],
            runtime=kernel_eval_result_dict["runtime"],
            metadata=kernel_eval_result_dict["metadata"],
        )
        kernel_exec_time = kernel_eval_result.runtime

        # Measure baseline time for PyTorch Eager
        print("[INFO] Measuring reference program time (eager mode)")
        ref_time_eager_result = EvalFunc.with_options(
            gpu=config.gpu
        )().measure_program_time.remote(
            ref_arch_name="Reference Program",
            ref_arch_src=ref_arch_src,
            num_trials=config.num_perf_trials,
            use_torch_compile=False,
            torch_compile_backend=None,
            torch_compile_options=None,
            gpu_arch=gpu_arch,
        )
        ref_exec_eager_time = ref_time_eager_result.get("mean", None)

        # Measure Torch Compile time
        print("[INFO] Measuring reference program time (torch.compile)")
        ref_time_compile_result = EvalFunc.with_options(
            gpu=config.gpu
        )().measure_program_time.remote(
            ref_arch_name="Reference Program",
            ref_arch_src=ref_arch_src,
            num_trials=config.num_perf_trials,
            use_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_options="default",
            gpu_arch=gpu_arch,
        )
        ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    # Print results
    print("=" * 40)
    print(f"[Eval] Kernel eval result: {kernel_eval_result}")
    print("-" * 40)
    print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
    print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
    print(f"[Timing] Custom Kernel exec time: {kernel_exec_time} ms")
    print("-" * 40)

    if kernel_eval_result.correctness:
        print(
            f"[Speedup] Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x"
        )
        print(
            f"[Speedup] Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x"
        )
    else:
        print("[Speedup] Speedup Not Available as Kernel did not pass correctness")

    print("=" * 40)


if __name__ == "__main__":
    main()
