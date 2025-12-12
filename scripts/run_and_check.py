import shutil
import torch
import pydra
from pydra import REQUIRED, Config
import os
from datasets import load_dataset
import modal

from src import eval as kernel_eval
from src import utils as kernel_utils
from scripts.generate_baseline_time import measure_program_time
from src.utils import read_file
from scripts.tk_compile import (
    compile_thunderkittens_cuda,
    compile_cuda_on_modal,
    prepare_kernel_src_with_cuda
)

# Modal setup
app = modal.App("run_and_check")
gpu_arch_mapping = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "H200": ["Hopper"],
    "A100": ["Ampere"],
    "A100-80GB": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"]
}

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
THUNDERKITTENS_LOCAL_PATH = os.path.join(REPO_TOP_PATH, "ThunderKittens")
SRC_PATH = os.path.join(REPO_TOP_PATH, "src")

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# ThunderKittens support - use TK image if directory exists locally
if os.path.isdir(THUNDERKITTENS_LOCAL_PATH):
    # ThunderKittens image with TK environment and mounting
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
        .apt_install("git", "gcc-10", "g++-10", "clang")
        .pip_install_from_requirements(os.path.join(REPO_TOP_PATH, "requirements.txt"))
        .pip_install("pybind11")  # pybind11 needed for ThunderKittens compilation
        .env({
            "THUNDERKITTENS_ROOT": "/root/ThunderKittens",
            "THUNDERKITTENS_PATH": "/root/ThunderKittens",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "CXX": "g++-10",
            "CC": "gcc-10",
        })
        .add_local_dir(THUNDERKITTENS_LOCAL_PATH, remote_path="/root/ThunderKittens", copy=True)
        .add_local_dir(KERNEL_BENCH_PATH, remote_path="/root/KernelBench")
        .add_local_dir(SRC_PATH, remote_path="/root/src")
        .add_local_python_source("src")
        .add_local_python_source("scripts")
    )
else:
    # Standard image without ThunderKittens
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
        .apt_install("git", "gcc-10", "g++-10", "clang")
        .pip_install_from_requirements(os.path.join(REPO_TOP_PATH, "requirements.txt"))
        .pip_install("pybind11")  # Ensure pybind11 is available
        .add_local_dir(KERNEL_BENCH_PATH, remote_path="/root/KernelBench")
        .add_local_python_source("src")
        .add_local_python_source("scripts")
    )

"""
Run a pair of KernelBench format (problem, solution) to check if solution is correct and compute speedup

You will need two files
1. Reference: PyTorch reference (module Model) implementation with init and input shapes
2. Solution: PyTorch solution (module ModelNew) with inline CUDA Code OR separate .cu/.py files

The Reference could be either
1. a local file: specify the path to the file
2. a kernelbench problem: specify level and problem id

====================================================
Usage:
1. PyTorch reference is a local file (local eval)
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add.py eval_mode=local

2. PyTorch reference is a kernelbench problem (local eval)
python3 scripts/run_and_check.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to model-generated kernel> eval_mode=local

3. PyTorch reference is a local file (modal eval on cloud GPU)
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add.py eval_mode=modal gpu=H100

4. PyTorch reference is a kernelbench problem (modal eval on cloud GPU)
python3 scripts/run_and_check.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to model-generated kernel> eval_mode=modal gpu=L40S

5. ThunderKittens separate .cu and .py files (like original framework)
python3 scripts/run_and_check.py ref_origin=kernelbench level=1 problem_id=1 cuda_src_path=results/eval_logs/Archive/1_1.cu kernel_src_path=results/eval_logs/Archive/1_1.py eval_mode=modal gpu=H100
====================================================

"""

torch.set_printoptions(precision=4, threshold=10)

class ScriptConfig(Config):
    def __init__(self):

        # Problem and Solution definition
        # Input src origin definition
        self.ref_origin = REQUIRED # either local or kernelbench
        # ref_origin is local, specify local file path
        self.ref_arch_src_path = ""
        # ref_origin is kernelbench, specify level and problem id
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = ""
        self.problem_id = ""
        # Solution src definition
        self.kernel_src_path = ""  # .py file with ModelNew
        # Optional: separate CUDA source file (ThunderKittens style)
        # If provided, this .cu file will be compiled as tk_kernels module
        # and the .py file (kernel_src_path) can import it
        self.cuda_src_path = ""  # .cu file with PYBIND11_MODULE(tk_kernels, ...)
        
        # Module name for the compiled CUDA kernel (default: tk_kernels)
        self.cuda_module_name = "tk_kernels"

        # Evaluation mode
        self.eval_mode = "local"  # either "local" or "modal"
        self.gpu = "L40S"  # GPU type for modal (L40S, H100, H200, A100, etc.)

        # KernelBench Eval specific
        # number of trials to run for correctness
        self.num_correct_trials = 5
        # number of trials to run for performance
        self.num_perf_trials = 100
        # timeout for each trial
        self.timeout = 300
        # verbose logging
        self.verbose = False
        self.measure_performance = True
        self.build_dir_prefix = "" # if you want to specify a custom build directory
        self.clear_cache = False # TODO

        # Replace with your NVIDIA GPU architecture, e.g. ["Hopper"]
        self.gpu_arch = ["Hopper"]
        self.precision = "fp16"
        self.backend = "cuda"

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"

def evaluate_single_sample_src(ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code against a reference source code
    """

    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
    
    if configs["clear_cache"]: # fresh kernel build
        print(f"[INFO] Clearing cache for build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)
    
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]    
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    try:
        eval_result = kernel_eval.eval_kernel_against_ref(
        original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=configs["backend"],
            precision=kernel_eval.get_torch_dtype_from_string(configs["precision"])
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False,
                                                metadata=metadata)
            return eval_result




# Modal evaluation class
@app.cls(image=image, scaledown_window=5)
class EvalFunc:

    @modal.method()
    def evaluate_single_sample_src_modal(self, ref_arch_src: str, kernel_src: str, configs: dict, gpu_arch: list,
                                          cuda_src: str = None, cuda_module_name: str = "tk_kernels"):
        """Evaluate a single sample source code against a reference source code on Modal"""
        from src.utils import set_gpu_arch
        from src.eval import eval_kernel_against_ref, get_torch_dtype_from_string
        from scripts.tk_compile import compile_cuda_on_modal, prepare_kernel_src_with_cuda

        set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        # If CUDA source provided, compile it first
        if cuda_src:
            cuda_module_path = compile_cuda_on_modal(cuda_src, cuda_module_name, gpu_arch)
            
            # Modify kernel_src to import the compiled module
            kernel_src = prepare_kernel_src_with_cuda(kernel_src, cuda_module_path, cuda_module_name)
            print(f"[Modal] Modified kernel source to use compiled module at {cuda_module_path}")

        num_correct_trials = configs["num_correct_trials"]
        num_perf_trials = configs["num_perf_trials"]
        verbose = configs["verbose"]
        measure_performance = configs["measure_performance"]

        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            device=device,
            backend=configs["backend"],
            precision=get_torch_dtype_from_string(configs["precision"])
        )
        return eval_result

    @modal.method()
    def measure_program_time_modal(
        self,
        ref_arch_src: str,
        num_trials: int,
        use_torch_compile: bool,
        torch_compile_backend: str,
        torch_compile_options: str,
        gpu_arch: list
    ):
        """Measure the execution time of a reference program on Modal"""
        from scripts.generate_baseline_time import measure_program_time
        from src.utils import set_gpu_arch

        set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        return measure_program_time(
            ref_arch_name="Reference Program",
            ref_arch_src=ref_arch_src,
            num_trials=num_trials,
            use_torch_compile=use_torch_compile,
            torch_compile_backend=torch_compile_backend,
            torch_compile_options=torch_compile_options,
            device=device
        )


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):

    print("Running with config", config)

    # Fetch reference and kernel code

    assert config.ref_origin == "local" or config.ref_origin == "kernelbench", "ref_origin must be either local or kernelbench"
    assert config.kernel_src_path != "", "kernel_src_path is required"  
    
    if config.ref_origin == "local":
        assert config.ref_arch_src_path != "", "ref_arch_src_path is required"
        ref_arch_src = read_file(config.ref_arch_src_path)
    elif config.ref_origin == "kernelbench":
        assert config.dataset_name != "", "dataset_name is required"
        assert config.level != "", "level is required"
        assert config.problem_id != "", "problem_id is required"

        # for now use the HuggingFace dataset
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

        problem_number = int(problem_name.split("_")[0])
        assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

        print(f"Fetched problem {config.problem_id} from KernelBench level {config.level}: {problem_name}")


    else:
        raise ValueError("Invalid ref_origin")
    
    kernel_src = read_file(config.kernel_src_path)
    
    # Handle separate .cu file if provided (ThunderKittens style)
    # For modal mode, compilation happens on the remote - skip local compilation
    cuda_module_path = None
    if config.cuda_src_path and config.eval_mode == "local":
        print(f"[INFO] Separate CUDA source provided: {config.cuda_src_path}")
        
        # Create a unique build directory based on the cuda source hash
        cuda_src_content = read_file(config.cuda_src_path)
        cuda_hash = str(hash(cuda_src_content))
        cuda_build_dir = os.path.join(
            config.build_dir_prefix if config.build_dir_prefix else os.path.join(REPO_TOP_PATH, "cache"),
            "tk_cuda_build",
            cuda_hash
        )
        
        # Compile the CUDA module if not in cache
        so_file = os.path.join(cuda_build_dir, f"{config.cuda_module_name}*.so")
        import glob
        existing_so = glob.glob(so_file)
        
        if existing_so and not config.clear_cache:
            print(f"[INFO] Using cached compiled module from {cuda_build_dir}")
            cuda_module_path = cuda_build_dir
        else:
            if config.clear_cache and os.path.exists(cuda_build_dir):
                print(f"[INFO] Clearing CUDA cache: {cuda_build_dir}")
                shutil.rmtree(cuda_build_dir, ignore_errors=True)
            
            cuda_module_path = compile_thunderkittens_cuda(
                cuda_src_path=config.cuda_src_path,
                module_name=config.cuda_module_name,
                build_dir=cuda_build_dir,
                verbose=config.verbose,
                repo_top_path=REPO_TOP_PATH
            )
        
        # Modify kernel_src to import the compiled module
        kernel_src = prepare_kernel_src_with_cuda(kernel_src, cuda_module_path, config.cuda_module_name)
        
        if config.verbose:
            print(f"[DEBUG] Modified kernel source with CUDA module path: {cuda_module_path}")
    elif config.cuda_src_path and config.eval_mode == "modal":
        print(f"[INFO] Separate CUDA source provided: {config.cuda_src_path}")
        print(f"[INFO] CUDA compilation will happen on Modal (remote GPU)")

    # Start Evaluation
    assert config.eval_mode in ["local", "modal"], "eval_mode must be either 'local' or 'modal'"

    if config.eval_mode == "local":
        # Local evaluation (existing code path)
        device = torch.device("cuda:0")
        kernel_utils.set_gpu_arch(config.gpu_arch)

        print("[INFO] Evaluating kernel against reference code (LOCAL)")
        # Evaluate kernel against reference code
        kernel_eval_result = evaluate_single_sample_src(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            configs=config.to_dict(),
            device=device
        )
        kernel_exec_time = kernel_eval_result.runtime

        # Measure baseline time
        print("[INFO] Measuring reference program time")
        # Default using PyTorch Eager here
        ref_time_eager_result = measure_program_time(ref_arch_name="Reference Program",
                                                    ref_arch_src=ref_arch_src,
                                                    num_trials=config.num_perf_trials,
                                                    use_torch_compile=False,
                                                    device=device)
        ref_exec_eager_time = ref_time_eager_result.get("mean", None)

        # Measure Torch Compile time
        ref_time_compile_result = measure_program_time(ref_arch_name="Reference Program",
                                                    ref_arch_src=ref_arch_src,
                                                    num_trials=config.num_perf_trials,
                                                    use_torch_compile=True,
                                                    torch_compile_backend="inductor",
                                                    torch_compile_options="default",
                                                    device=device)
        ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    elif config.eval_mode == "modal":
        # Modal evaluation (remote execution)
        gpu_arch = gpu_arch_mapping.get(config.gpu, config.gpu_arch)
        print(f"[INFO] Using GPU: {config.gpu} with architecture: {gpu_arch}")
        
        # Read CUDA source if provided (will be compiled on Modal)
        cuda_src = None
        if config.cuda_src_path:
            print(f"[INFO] Will compile CUDA source on Modal: {config.cuda_src_path}")
            cuda_src = read_file(config.cuda_src_path)
            # For Modal, we use the original kernel_src (without local path modifications)
            kernel_src = read_file(config.kernel_src_path)

        with app.run():
            print("[INFO] Evaluating kernel against reference code (MODAL)")
            # Evaluate kernel against reference code
            kernel_eval_result = EvalFunc.with_options(
                gpu=config.gpu
            )().evaluate_single_sample_src_modal.remote(
                ref_arch_src=ref_arch_src,
                kernel_src=kernel_src,
                configs=config.to_dict(),
                gpu_arch=gpu_arch,
                cuda_src=cuda_src,
                cuda_module_name=config.cuda_module_name
            )
            kernel_exec_time = kernel_eval_result.runtime

            # Measure baseline time
            print("[INFO] Measuring reference program time (PyTorch Eager)")
            ref_time_eager_result = EvalFunc.with_options(
                gpu=config.gpu
            )().measure_program_time_modal.remote(
                ref_arch_src=ref_arch_src,
                num_trials=config.num_perf_trials,
                use_torch_compile=False,
                torch_compile_backend=None,
                torch_compile_options=None,
                gpu_arch=gpu_arch
            )
            ref_exec_eager_time = ref_time_eager_result.get("mean", None)

            # Measure Torch Compile time
            print("[INFO] Measuring reference program time (torch.compile)")
            ref_time_compile_result = EvalFunc.with_options(
                gpu=config.gpu
            )().measure_program_time_modal.remote(
                ref_arch_src=ref_arch_src,
                num_trials=config.num_perf_trials,
                use_torch_compile=True,
                torch_compile_backend="inductor",
                torch_compile_options="default",
                gpu_arch=gpu_arch
            )
            ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    print("="*40)
    print(f"[Eval] Kernel eval result: {kernel_eval_result}")
    print("-"*40)
    print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
    print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
    print(f"[Timing] Custom Kernel exec time: {kernel_exec_time} ms")
    print("-"*40)   
    
    if kernel_eval_result.correctness:
        print(f"[Speedup] Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x")
        print(f"[Speedup] Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x")
    else:
        print("[Speedup] Speedup Not Available as Kernel did not pass correctness")

    print("="*40)


if __name__ == "__main__":
    main()