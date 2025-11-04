import torch
import numpy as np
from src.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
    fetch_ref_arch_from_problem_id,
)
from src.dataset import construct_kernelbench_dataset
from src.utils import read_file
import os
import json
from tqdm import tqdm
import multiprocessing as mp
import time
import einops
import pydra
from pydra import Config, REQUIRED

"""
Generate baseline time for KernelBench
This profiles the wall clock time for each KernelBench reference problem

You can find a list of pre-generated baseline time in /results/timing/
But we recommend you run this script to generate the baseline time for your own hardware configurations

Using various configurations
- torch (Eager)

Torch Compile with various modes
https://pytorch.org/docs/main/generated/torch.compile.html
- torch.compile: backend="inductor", mode="default" (this is usually what happens when you do torch.compile(model))
- torch.compile: backend="inductor", mode="reduce-overhead" 
- torch.compile: backend="inductor", mode="max-autotune"
- torch.compile: backend="inductor", mode="max-autotune-no-cudagraphs"

In addition to default Torch Compile backend, you can always use other or your custom backends
https://pytorch.org/docs/stable/torch.compiler.html
- torch.compile: backend="cudagraphs" (CUDA graphs with AOT Autograd)

Usage:
```
python scripts/generate_baseline_time_modal.py gpu=L40S level=1 subset="[1,5]"
```
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")


class BaselineConfig(Config):
    def __init__(self):
        self.gpu = "L40S"  # GPU type (L40S, A10G, H100, A100, etc.)
        self.level = 1  # KernelBench level to evaluate (1, 2, or 3)
        self.subset = None  # Subset of problems to evaluate, e.g., "[1,5]" for problems 1-5
        self.hardware_name = ""  # Optional: override hardware name, defaults to {gpu}_modal
        self.batch_size = 10  # Number of problems to process in parallel
        self.timeout = 1800  # Timeout per batch in seconds
        
    def __repr__(self):
        return f"BaselineConfig({self.to_dict()})"

# Modal Infra
import modal
app = modal.App("generate_baseline_modal")
gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "A100-80GB": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}
batch_size = 10
gpu = "L40S"
timeout = 1800
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang" # note i skip a step 
                )
    .pip_install(  # required to build flash-attn
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "einops",
        "python-dotenv",
    )
    .add_local_dir(
        KERNEL_BENCH_PATH,
        remote_path="/root/KernelBench"
    )
    .add_local_python_source("src") 
)

def write_batch_to_json(entries_to_write: list, f_path: str):
    """
    Write batch of data to JSON file (append or overwrite, do not completely overwrite)
    """
    # Read existing data if file exists
    existing_data = {}
    if os.path.exists(f_path):
        with open(f_path, 'r') as f_r:
            existing_data = json.load(f_r)
            
    # Add new entries
    for (level, problem, entry) in entries_to_write:
        # Initialize nested structure if it doesn't exist
        if str(level) not in existing_data:
            existing_data[level] = {}
        existing_data[level][problem] = entry

    # Write updated results back to file
    if not os.path.exists(f_path):
        os.makedirs(os.path.dirname(f_path), exist_ok=True)

    # Write back combined data
    with open(f_path, "w") as f_w:
        json.dump(existing_data, f_w, indent=4)
    
    print(f"[INFO] Wrote {len(entries_to_write)} entries to {f_path}")


@app.cls(image=image, scaledown_window=5)
class EvalFunc:

    @modal.method()
    def measure_program_time(
            self,
            ref_arch_name: str,
            ref_arch_src: str, 
            num_trials: int = 100,
            use_torch_compile: bool = False,
            torch_compile_backend: str="inductor", 
            torch_compile_options: str="default",
            device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None,
            verbose: bool = False,
    ):
        """
        Measure the time of a KernelBench reference architecture
        """
        context = {}
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            ref_arch_src, context
        )
        try:
            with torch.no_grad():
                torch.cuda.synchronize(device=device)
                set_seed(42)
                inputs = get_inputs()
                set_seed(42)
                init_inputs = get_init_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                init_inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in init_inputs
                ]

                # Initialize PyTorch model, use this for eager mode execution
                model = Model(*init_inputs)
                
                if use_torch_compile:
                    print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                    model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
                else:
                    print(f"Using PyTorch Eager Execution on {ref_arch_name}")
                
                model = model.cuda(device=device)
                torch.cuda.synchronize(device=device)
                elapsed_times = time_execution_with_cuda_event(
                    model, *inputs, num_trials=num_trials, verbose=verbose, device=device
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"{ref_arch_name} {runtime_stats}")
                
                return runtime_stats
        except Exception as e:
            print(f"[Eval] Error in Measuring Performance: {e}")

def measure_program_time_wrapper(*args, **kwargs):
    with app.run():
        return EvalFunc.with_options(gpu=gpu)().measure_program_time.remote(*args, **kwargs)

def record_baseline_times(config: BaselineConfig,
                          use_torch_compile: bool = False, 
                          torch_compile_backend: str="inductor", 
                          torch_compile_options: str="default",
                          file_name: str="baseline_time.json"):
    """
    Generate baseline time for KernelBench, 
    configure profiler options for PyTorch
    save to specified file
    """
    json_results = []
    
    # Parse subset if provided
    subset_problems = None
    if config.subset:
        # config.subset is already parsed by pydra as a list
        subset_range = config.subset
        if isinstance(subset_range, str):
            # Handle case where it's passed as a string (shouldn't happen with pydra, but just in case)
            import ast
            subset_range = ast.literal_eval(subset_range)
        
        if isinstance(subset_range, list) and len(subset_range) == 2:
            subset_problems = list(range(subset_range[0], subset_range[1] + 1))
        else:
            subset_problems = subset_range

    # Determine which levels to process
    levels_to_process = [config.level] if config.level else [1, 2, 3]

    for level in levels_to_process:
        dataset = construct_kernelbench_dataset(level)
        
        # Get actual problem IDs from dataset
        all_problem_ids = dataset.get_problem_ids()
        
        # Filter by subset if specified
        if subset_problems:
            problem_ids = [pid for pid in all_problem_ids if pid in subset_problems]
        else:
            problem_ids = all_problem_ids
        
        # Prepare work items with problem metadata
        total_work = []
        for problem_id in problem_ids:
            ref_arch_path = dataset.get_problem_by_id(problem_id)
            ref_arch_name = os.path.basename(ref_arch_path)
            ref_arch_src = read_file(ref_arch_path)
            total_work.append((problem_id, ref_arch_path, ref_arch_name, ref_arch_src))

        with tqdm(total=len(total_work), desc=f"Processing level {level}") as pbar:
            while len(total_work) > 0:
                curr_work_batch = total_work[:config.batch_size]
                total_work = total_work[config.batch_size:]  # pop the first batch_size elements

                with mp.Pool() as pool:

                    work_args = [
                        (
                            ref_arch_name,
                            ref_arch_src,
                            100,
                            use_torch_compile,
                            torch_compile_backend,
                            torch_compile_options,
                            torch.device(f"cuda:0"),
                            False # do not print  
                        )
                        for i, (p_id, ref_arch_path, ref_arch_name, ref_arch_src) in enumerate(curr_work_batch)
                    ]

                    start_time = time.time()

                    async_results = []
                    for work_arg in work_args:
                        async_results.append(
                            pool.apply_async(measure_program_time_wrapper, work_arg)
                        )

                    batch_timeout = config.timeout
                    for i, async_result in enumerate(async_results):
                        problem_id, _, ref_arch_name, _ = curr_work_batch[i]

                        try:
                            elapsed_time = time.time() - start_time
                            remaining_time = max(0, batch_timeout - elapsed_time)
                            result = async_result.get(timeout=remaining_time)
                            json_results.append((f"level{level}", ref_arch_name, result))
                            
                        except mp.TimeoutError:
                            print(
                                f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}"
                            )
                            json_results.append((f"level{level}", ref_arch_name, None))
            
                        except Exception as e:
                            print(
                                f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}: {str(e)}"
                            )
                            json_results.append((f"level{level}", ref_arch_name, None))

                    pbar.update(len(curr_work_batch))

    save_path = os.path.join(TIMING_DIR, file_name)
    write_batch_to_json(json_results, save_path)
    return json_results


@pydra.main(base=BaselineConfig)
def main(config: BaselineConfig):
    # Set global variables from config
    global gpu, batch_size, timeout
    gpu = config.gpu
    batch_size = config.batch_size
    timeout = config.timeout
    
    # Determine hardware name
    hardware_name = config.hardware_name if config.hardware_name and config.hardware_name != "" else f"{gpu}_modal"
    
    print(f"Generating baseline time for {hardware_name}")
    print(f"Configuration: {config}")
    
    # Check if directory exists
    if os.path.exists(os.path.join(TIMING_DIR, hardware_name)):
        print(f"[WARNING] Directory {hardware_name} already exists and will be updated/overwritten.")

    # 1. Record Torch Eager
    print("\n[1/2] Recording Torch Eager baseline...")
    record_baseline_times(config,
                        use_torch_compile=False, 
                        torch_compile_backend=None,
                        torch_compile_options=None, 
                        file_name=f"{hardware_name}/baseline_time_torch.json")
    
    # 2. Record Torch Compile with Inductor (default mode)
    print("\n[2/2] Recording Torch Compile (inductor, default) baseline...")
    record_baseline_times(config,
                        use_torch_compile=True, 
                        torch_compile_backend="inductor",
                        torch_compile_options="default", 
                        file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_default.json")
    
    print(f"\n✓ Baseline generation complete! Results saved to: results/timing/{hardware_name}/")
    
    # Optional: Record additional Torch Compile modes (uncomment if needed)
    # for torch_compile_mode in ["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
    #     record_baseline_times(config,
    #                           use_torch_compile=True, 
    #                           torch_compile_backend="inductor",
    #                           torch_compile_options=torch_compile_mode, 
    #                           file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_{torch_compile_mode}.json")
 
    # Optional: Record Torch Compile using cudagraphs (uncomment if needed)
    # record_baseline_times(config,
    #                       use_torch_compile=True, 
    #                       torch_compile_backend="cudagraphs",
    #                       torch_compile_options=None, 
    #                       file_name=f"{hardware_name}/baseline_time_torch_compile_cudagraphs.json")


if __name__ == "__main__":
    main()




################################################################################
# Deprecated
################################################################################


def get_time_old(level_num, problem_id, num_trials=100, torch_compile=False):
    raise DeprecationWarning("Use New measure_program_time instead")
    ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(
        level_num, problem_id, with_name=True
    )
    ref_arch_name = ref_arch_name.split("/")[-1]
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]
            model = Model(*init_inputs)
            
            if torch_compile:
                model = torch.compile(model)
                print("Compiled model Done")
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=False, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)
            # json_results[f"level{level_num}"][ref_arch_name] = runtime_stats
            print(f"{ref_arch_name} {runtime_stats}")
            return (ref_arch_name, runtime_stats)
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")


