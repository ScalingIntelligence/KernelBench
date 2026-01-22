import torch
import numpy as np
from kernelbench.eval import (
    load_original_model_and_inputs,
    set_seed,
    fetch_ref_arch_from_problem_id,
)
from kernelbench.timing import (
    get_timing_function,
    get_timing_stats,
)
from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset
from kernelbench.utils import read_file
import os
import json
from tqdm import tqdm
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
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")  # Dataset directory

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")


class BaselineConfig(Config):
    def __init__(self):
        # Problem level to generate baseline for
        self.level = REQUIRED

        # GPU type for Modal (L40S, H100, A100, A100-80GB, L4, T4, A10G)
        self.gpu = REQUIRED

        # Hardware name for saving results
        self.hardware_name = REQUIRED

        # Number of parallel GPU containers to use
        self.num_gpu_devices = 8

        # Timeout for each batch in seconds
        self.timeout = 1800

        # Number of trials for timing
        self.num_trials = 100


# Modal Infra
import modal
app = modal.App("generate_baseline_modal")
gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "A100-80GB": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

SRC_DIR = os.path.join(REPO_TOP_PATH, "src")
KERNELBENCH_DIR = os.path.join(REPO_TOP_PATH, "KernelBench")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang" # note i skip a step
                )
    .uv_sync(uv_project_dir=REPO_TOP_PATH, extras=["gpu"])
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(SRC_DIR, remote_path="/root/src")
    .add_local_dir(KERNELBENCH_DIR, remote_path="/root/KernelBench")  # must be last
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
            timing_method: str="cuda_event",
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
                timing_func = get_timing_function(timing_method)
                torch.cuda.synchronize(device=device)
                elapsed_times = timing_func(
                    model,
                    inputs,
                    num_warmup=3,  # or any default you prefer
                    num_trials=num_trials,
                    discard_first=1,  # or 0 to include first trial
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"{ref_arch_name} {runtime_stats}")
                
                return runtime_stats
        except Exception as e:
            print(f"[Eval] Error in Measuring Performance: {e}")

def record_baseline_times(config: BaselineConfig,
                          use_torch_compile: bool = False,
                          torch_compile_backend: str="inductor",
                          torch_compile_options: str="default",
                          file_name: str="baseline_time.json"):
    """
    Generate baseline time for KernelBench using Modal's native parallelization.
    Spawns multiple GPU containers in parallel for faster processing.
    """
    json_results = []

    level = config.level
    dataset = construct_kernelbench_dataset(level)
    num_problems = len(dataset)
    total_work = [(i, *fetch_ref_arch_from_dataset(dataset, i)) for i in dataset.get_problem_ids()]

    batch_size = config.num_gpu_devices
    print(f"[Modal] Processing {len(total_work)} problems in parallel batches of {batch_size}")

    with app.run():
        evaluator_cls = EvalFunc.with_options(gpu=config.gpu) if config.gpu != "L40S" else EvalFunc

        with tqdm(total=len(total_work), desc="Processing") as pbar:
            while len(total_work) > 0:
                curr_work_batch = total_work[:batch_size]
                total_work = total_work[batch_size:]

                # Spawn all tasks in parallel using Modal
                futures = []
                for p_id, ref_arch_path, ref_arch_name, ref_arch_src in curr_work_batch:
                    future = evaluator_cls().measure_program_time.spawn(
                        ref_arch_name=ref_arch_name,
                        ref_arch_src=ref_arch_src,
                        num_trials=config.num_trials,
                        timing_method="cuda_event",
                        use_torch_compile=use_torch_compile,
                        torch_compile_backend=torch_compile_backend,
                        torch_compile_options=torch_compile_options,
                        device=torch.device("cuda:0"),
                        verbose=False,
                    )
                    futures.append((p_id, ref_arch_name, future))

                # Collect results
                for p_id, ref_arch_name, future in futures:
                    try:
                        result = future.get(timeout=config.timeout)
                        json_results.append((f"level{level}", ref_arch_name, result))
                    except Exception as e:
                        print(f"[ERROR] Problem {p_id} ({ref_arch_name}): {str(e)}")
                        json_results.append((f"level{level}", ref_arch_name, None))

                pbar.update(len(curr_work_batch))

    save_path = os.path.join(TIMING_DIR, file_name)
    write_batch_to_json(json_results, save_path)
    return json_results


@pydra.main(base=BaselineConfig)
def main(config: BaselineConfig):
    """
    Generate baseline time for KernelBench problems using Modal GPUs
    """
    print(f"Generating baseline time for level {config.level} on {config.gpu} Modal")
    print(f"Hardware name: {config.hardware_name}")
    print(f"Parallel GPUs: {config.num_gpu_devices}, Timeout: {config.timeout}s, Num trials: {config.num_trials}")

    # 1. Record Torch Eager
    print("\n[1/2] Recording baseline times with PyTorch Eager execution...")
    record_baseline_times(
        config=config,
        use_torch_compile=False,
        torch_compile_backend=None,
        torch_compile_options=None,
        file_name=f"{config.hardware_name}/baseline_time_torch.json"
    )

    # 2. Record Torch Compile using Inductor (default mode)
    print("\n[2/2] Recording baseline times with Torch Compile (inductor, default mode)...")
    record_baseline_times(
        config=config,
        use_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_options="default",
        file_name=f"{config.hardware_name}/baseline_time_torch_compile_inductor_default.json"
    )

    print(f"\n✓ Baseline time generation complete!")
    print(f"Results saved to: {os.path.join(TIMING_DIR, config.hardware_name)}")


if __name__ == "__main__":
    main()
    



    # Random debuging
    # get_torch_compile_triton(2, 12)
    # record_baseline_times()

    # run_profile(2, 43)
    # get_time(2, 43, torch_compile=False)
    # get_time(2, 43, torch_compile=True)




################################################################################
# Deprecated
################################################################################


def get_time_old(level_num, problem_id, num_trials=100, torch_compile=False):
    raise DeprecationWarning("Use New measure_program_time instead")
    ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(
        level_num, problem_id, with_name=True
    )
    ref_arch_name = os.path.basename(ref_arch_name)
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


