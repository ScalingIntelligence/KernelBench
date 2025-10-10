import json
import multiprocessing as mp
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pydra
import torch

from datasets import load_dataset
from pydra import Config, REQUIRED
from src import compile, eval, utils

from src.dataset import construct_kernelbench_dataset
from src.eval import (
    build_compile_cache,
    check_metadata_serializable_all_types,
    eval_kernel_against_ref,
    KernelExecResult,
)
from src.utils import read_file, set_gpu_arch
from tqdm import tqdm

# Modal support
import modal

"""
Batch Evaluation from Existing Generations

This expects you have generated the kernels and stored them in the runs/{run_name} directory
This eval script will evaluate the kernels against the reference architecture, and store the results in the runs/{run_name}/eval_results.json file

Usually with eval, we check
- correctness (n_correct): 5 randomized input trials
- performance (n_trials): 100 randomized input trials

You can increase the number of trials for correctness and performance
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_DIR, "KernelBench")

torch.set_printoptions(precision=4, threshold=10)

# Modal Infrastructure Setup
app = modal.App("eval_from_generations_modal")
gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang"
                )
    .pip_install(
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
        "python-dotenv",
    )
    .add_local_dir(
        KERNEL_BENCH_PATH,
        remote_path="/root/KernelBench"
    )
    .add_local_python_source("src")
)


class EvalConfig(Config):
    def __init__(self):

        self.run_name = REQUIRED  # name of the run to evaluate

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

        # subset of problems to evaluate
        self.subset = (None, None)  # (start_id, end_id), these are the logical index

        # Evaluation Mode: local (requires GPU), modal (cloud GPU)
        self.eval_mode = "local"

        # For Modal: GPU type to use (L40S, H100, A100, L4, T4, A10G)
        self.gpu = "A10G"

        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        self.verbose = False

        # Eval settings
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 180  # in seconds
        self.measure_performance = True

        # Eval Flow setting
        # To speedup evaluation, you can start building the kernel on CPU on disk as cache
        self.build_cache = False
        self.num_cpu_workers = (
            20  # number of parallel process to to parallelize the build on CPUs
        )

        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")

        # number of GPUs to do batch evaluation
        self.num_gpu_devices = 1

        # Number of samples per problem to evaluate for pass@k analysis
        self.num_samples_per_problem = 1  # Default to 1 sample per problem

        # List of k values for pass@k calculation (e.g., [1, 5, 10])
        self.pass_at_k_values = [1]  # Default to only pass@1

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int
    sample_id: int
    device: torch.device


# Modal Evaluation Class
# GPU must be specified here for all instances
# Retries are configured at the class level to handle GPU attachment failures
# @modal.concurrent: Each container handles exactly ONE evaluation at a time - prevents memory leaks
@app.cls(
    image=image, 
    gpu="A10G",
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    )
)
@modal.concurrent(max_inputs=1)  # One input per container - prevents GPU memory leaks
class ModalEvaluator:
    
    @modal.method()
    def evaluate_single_sample_modal(
        self,
        ref_arch_src: str,
        kernel_src: str,
        gpu_arch: list[str],
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        measure_performance: bool = True,
        verbose: bool = False,
    ):
        """
        Evaluate a single sample on Modal GPU with automatic retries for GPU attachment failures
        """
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        import torch
        import time
        
        max_wait_time = 30
        start_time = time.time()
        gpu_available = False
        
        while time.time() - start_time < max_wait_time:
            if torch.cuda.is_available():
                gpu_available = True
                break
            # Progressive backoff: 0.5s, 1s, 2s, 4s, 8s...
            wait_time = min(0.5 * (2 ** int((time.time() - start_time) / 2)), 8.0)
            time.sleep(wait_time)
        
        if not gpu_available:
            raise RuntimeError(
                f"GPU not attached to container after {max_wait_time}s - Modal will retry with new container"
            )
        
        set_gpu_arch(gpu_arch)
        
        result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=None,  # Modal doesn't need persistent build dir
            device=torch.device("cuda:0"),  # Modal has one GPU per container
        )
        
        # Force cleanup and exit to prevent container reuse and memory leaks
        torch.cuda.empty_cache()
        
        return result  # Never reached, but needed for type checking


def fetch_ref_arch_from_problem_id(
    dataset, problem_id: int, dataset_src: str
) -> str | None:
    """
    Fetch reference architecture from problem directory
    Either from Hugging Face or Local Dataset
    """
    if dataset_src == "huggingface":
        curr_problem_row = dataset.filter(
            lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None
        )
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif dataset_src == "local":
        problem_idx_in_dataset = (
            problem_id - 1
        )  # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # verify
    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert (
        problem_number == problem_id
    ), f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"

    return ref_arch_src


def fetch_kernel_from_disk(
    run_dir: str, level: int, problem_id: int, sample_id: int
) -> str | None:
    """
    Fetch kernel file from disk (stored in runs/{run_name})
    """
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )

    if os.path.exists(kernel_path):
        return read_file(kernel_path)
    else:
        return None


def evaluate_single_sample(
    work_args: WorkArgs, configs: EvalConfig, dataset, run_dir: str
) -> KernelExecResult | None:
    """
    Evaluate a single sample on a single GPU
    """
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # fetch reference architecture from problem directory
    ref_arch_src = fetch_ref_arch_from_problem_id(
        dataset, problem_id, configs.dataset_src
    )

    # fetch kernel from disk
    # Add database support in the future
    kernel_src = fetch_kernel_from_disk(run_dir, configs.level, problem_id, sample_id)

    assert (
        kernel_src is not None
    ), f"Kernel not found for problem {problem_id} sample {sample_id}"

    build_dir = os.path.join(
        configs.kernel_eval_build_dir, configs.run_name, f"{problem_id}", f"{sample_id}"
    )

    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=configs.measure_performance,
            verbose=configs.verbose,
            num_correct_trials=configs.num_correct_trials,
            num_perf_trials=configs.num_perf_trials,
            build_dir=build_dir,
            device=device,
        )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} "
        )
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {
                "other_error": f"error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # for debugging
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result


def evaluate_single_sample_modal_direct(
    problem_id: int,
    sample_id: int,
    ref_arch_src: str,
    kernel_src: str,
    gpu: str,
    configs: EvalConfig,
):
    """
    Evaluate a single sample using Modal
    """
    gpu_arch = gpu_arch_mapping.get(gpu, ["Ada"])
    
    try:
        evaluator = ModalEvaluator()
        eval_result = evaluator.evaluate_single_sample_modal.remote(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            gpu_arch=gpu_arch,
            num_correct_trials=configs.num_correct_trials,
            num_perf_trials=configs.num_perf_trials,
            measure_performance=configs.measure_performance,
            verbose=configs.verbose,
        )
        return eval_result
    except Exception as e:
        print(f"[ERROR] Modal evaluation failed for problem {problem_id} sample {sample_id}: {e}")
        return None


def cuda_single_eval_wrapper(curr_work: WorkArgs, configs: dict, dataset, run_dir: str):
    """
    Wrapper to handle timeout and keyboard interrupt
    """

    with mp.Pool(1) as pool:
        try:
            result = pool.apply_async(
                evaluate_single_sample,
                args=(curr_work, configs, dataset, run_dir),
            ).get(timeout=configs.timeout)
        except KeyboardInterrupt:
            print("\n [Terminate] Caught KeyboardInterrupt, terminating workers...")
            pool.terminate()
            pool.join()
            raise
        except mp.TimeoutError:
            print(
                f"[WARNING] Evaluation TIMED OUT for Problem ID: {curr_work.problem_id},"
                f" Sample ID: {curr_work.sample_id}"
            )

        print(
            f"[Eval Result] Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}: {result}"
        )
        return result


def remove_cache_dir(cache_dir: str, run_name: str, problem_id, sample_id):
    """
    Remove the cached folder for sample compilation so it can start a clean build next time
    useful for time out, failed build, etc.
    """
    problem_cache_dir = os.path.join(
        cache_dir, run_name, f"{problem_id}", f"{sample_id}"
    )
    print(f"cache_dir to remove: {problem_cache_dir}")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(
                f"\n[INFO] Removed cached folder for Problem ID: {problem_id}, Sample ID: {sample_id}"
            )
        except Exception as e:
            print(f"\n[WARNING] Failed to remove cache directory {cache_dir}: {str(e)}")


def batch_eval_modal(
    total_work: list[tuple[int, int]],
    config: EvalConfig,
    curr_level_dataset,
    run_dir: str,
    eval_file_path: str,
):
    print(f"[Modal] Starting batch evaluation on {config.gpu} GPUs")
    print(f"[Modal] Processing {len(total_work)} samples in parallel batches of {config.num_gpu_devices}")
    
    with app.run():
        with tqdm(total=len(total_work), desc="Modal Evaluation Progress") as pbar:
            batch_size = config.num_gpu_devices
            
            while len(total_work) > 0:
                curr_work_batch = total_work[:batch_size]
                total_work = total_work[batch_size:]
                
                print(f"\n[Modal Batch] Processing {len(curr_work_batch)} samples; {len(total_work)} remaining")
                
                start_time = time.time()
                
                # Prepare work items - fetch all data first
                work_items = []
                for problem_id, sample_id in curr_work_batch:
                    ref_arch_src = fetch_ref_arch_from_problem_id(
                        curr_level_dataset, problem_id, config.dataset_src
                    )
                    kernel_src = fetch_kernel_from_disk(run_dir, config.level, problem_id, sample_id)
                    
                    if kernel_src is None:
                        print(f"[WARNING] Kernel not found for problem {problem_id} sample {sample_id}")
                        work_items.append(None)
                    else:
                        work_items.append({
                            'problem_id': problem_id,
                            'sample_id': sample_id,
                            'ref_arch_src': ref_arch_src,
                            'kernel_src': kernel_src,
                        })
                
                # Submit all evaluations in parallel using Modal
                gpu_arch = gpu_arch_mapping.get(config.gpu, ["Ada"])
                
                # Override GPU if different from default in decorator
                # .with_options() overrides the decorator's parameters
                evaluator_cls = ModalEvaluator.with_options(gpu=config.gpu) if config.gpu != "A10G" else ModalEvaluator
                
                # Spawn all tasks in parallel
                # Each spawn creates a NEW container instance with a GPU
                futures = []
                for item in work_items:
                    if item is None:
                        futures.append(None)
                    else:
                        future = evaluator_cls().evaluate_single_sample_modal.spawn(
                            ref_arch_src=item['ref_arch_src'],
                            kernel_src=item['kernel_src'],
                            gpu_arch=gpu_arch,
                            num_correct_trials=config.num_correct_trials,
                            num_perf_trials=config.num_perf_trials,
                            measure_performance=config.measure_performance,
                            verbose=config.verbose,
                        )
                        futures.append(future)
                
                # Collect results from all futures
                results = []
                for i, future in enumerate(futures):
                    problem_id, sample_id = curr_work_batch[i]
                    
                    if future is None:
                        results.append((problem_id, sample_id, None))
                    else:
                        try:
                            result = future.get()
                            results.append((problem_id, sample_id, result))
                        except Exception as e:
                            error_msg = str(e)
                            # Check if it's a GPU attachment failure that exhausted retries
                            if "GPU not attached" in error_msg or "CUDA is not available" in error_msg:
                                print(f"[ERROR] Modal GPU attachment FAILED after retries for Problem ID: {problem_id}, Sample ID: {sample_id}")
                                print(f"        This is a Modal infrastructure issue. Sample will be skipped.")
                            else:
                                print(f"[ERROR] Modal evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {error_msg}")
                            results.append((problem_id, sample_id, None))
                
                end_time = time.time()
                
                # Save results
                for problem_id, sample_id, result in results:
                    print("-" * 128)
                    print(f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}")
                    print(result)
                    
                    if result is not None:
                        print(f"Adding Eval Result to file for problem {problem_id} sample {sample_id}")
                        add_to_eval_results_file(
                            problem_id, sample_id, result, eval_file_path
                        )
                
                print("-" * 128)
                print(f"[Modal Batch] Evaluation took {end_time - start_time:.2f} seconds")
                
                pbar.update(len(curr_work_batch))


def batch_eval(
    total_work: list[tuple[int, int]],
    config: EvalConfig,
    curr_level_dataset,
    run_dir: str,
    eval_file_path: str,
):
    """
    Batch evaluation across multiple GPUs (local or Modal)
    We put in time out for each batch, consider trying again with larger time out if it didn't finish building.
    Cache directory is removed if evaluation times out or fails
    """
    
    # Use Modal-based evaluation if eval_mode is "modal"
    if config.eval_mode == "modal":
        return batch_eval_modal(total_work, config, curr_level_dataset, run_dir, eval_file_path)
    
    # Original local GPU evaluation
    # construct a list of work args
    batch_size = config.num_gpu_devices

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {config.num_gpu_devices} GPUs; [Total Work left] {len(total_work)}"
            )
            assert (
                len(curr_work_batch) <= batch_size
            ), f"Current batch size {len(curr_work_batch)} is greater than the number of GPUs {batch_size}"

            with mp.Pool(batch_size) as pool:

                work_args = [
                    (
                        WorkArgs(
                            problem_id=p_id,
                            sample_id=s_idx,
                            device=torch.device(f"cuda:{i%batch_size}"),
                        ),
                        config,
                        curr_level_dataset,
                        run_dir,
                    )
                    for i, (p_id, s_idx) in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(evaluate_single_sample, work_arg)
                    )

                # Collect results with a batch timeout
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id = curr_work_batch[i]

                    try:
                        elapsed_time = time.time() - start_time
                        remaining_time = max(0, batch_timeout - elapsed_time)
                        result = async_result.get(timeout=remaining_time)
                        results.append((problem_id, sample_id, result))

                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        results.append((problem_id, sample_id, None))

                        remove_cache_dir(
                            config.kernel_eval_build_dir,
                            config.run_name,
                            problem_id,
                            sample_id,
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        results.append((problem_id, sample_id, None))
                        remove_cache_dir(
                            config.kernel_eval_build_dir,
                            config.run_name,
                            problem_id,
                            sample_id,
                        )

                end_time = time.time()

                # current batch summary
                for problem_id, sample_id, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}"
                    )
                    print(result)

                    # add all the batch results here to avoid file race condition
                    # add to eval result if valid result
                    if result is not None:
                        print(
                            f"Adding Eval Result to file for problem {problem_id} sample {sample_id}"
                        )
                        add_to_eval_results_file(
                            problem_id, sample_id, result, eval_file_path
                        )

                print("-" * 128)
                print(
                    f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))


def check_if_eval_exists_local(
    problem_id: int, sample_id: int, eval_file_path: str
) -> bool:
    """
    Check if evaluation result already exists in eval results file
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as f:
            eval_results = json.load(f)
        return str(problem_id) in eval_results
    return False


def add_to_eval_results_file(
    problem_id: int, sample_id: int, eval_result: KernelExecResult, eval_file_path: str
):
    """
    Add evaluation result to eval results file
    TODO: migrate database support
    """
    # Load existing results if file exists
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as f:
            eval_results = json.load(f)
            eval_results = defaultdict(lambda: [], eval_results)
    else:
        eval_results = defaultdict(lambda: [])

    # Add new result
    eval_results[str(problem_id)].append(
        {
            "sample_id": sample_id,
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "metadata": check_metadata_serializable_all_types(eval_result.metadata),
            "runtime": eval_result.runtime,
            "runtime_stats": eval_result.runtime_stats,
        }
    )

    # Write updated results back to file
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)

    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f)


def single_eval_example(
    config: EvalConfig, curr_level_dataset: list[str], run_dir: str, eval_file_path
):
    device = torch.device("cuda:0")
    example_work = WorkArgs(problem_id=1, sample_id=0, device=device)
    # example_eval_result = evaluate_single_sample(example_work, config, curr_level_dataset, run_dir)
    example_eval_result = cuda_single_eval_wrapper(
        example_work, config, curr_level_dataset, run_dir
    )
    print(example_eval_result)
    if not check_if_eval_exists_local(1, 0, eval_file_path):
        add_to_eval_results_file(1, 0, example_eval_result, eval_file_path)


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Batch Eval Samples from Particular Run
    Store Eval Results in specified eval results file
    """
    print(f"Starting Batch Eval with config: {config}")

    # Check if CUDA is available (only for local mode)
    if config.eval_mode == "local":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available. Local evaluation requires GPU.")
        
        # set GPU arch to configure what target to build for
        set_gpu_arch(config.gpu_arch)
        assert (
            config.num_gpu_devices <= torch.cuda.device_count()
        ), f"Number of GPUs requested ({config.num_gpu_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"
    else:
        print(f"[Modal] Using Modal for evaluation with GPU: {config.gpu}")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # Dataset Configurations
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert (
            config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level
        ), f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(
        f"Evaluating {config.num_samples_per_problem} sample(s) each for level {config.level} problems: {problem_id_range}"
    )

    run_dir = os.path.join(config.runs_dir, config.run_name)
    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    # To Debug
    # single_eval_example(config, curr_level_dataset, run_dir, eval_file_path)

    total_work = []
    for problem_id in range(
        problem_id_range.start, problem_id_range.stop + 1
    ):  # end index is inclusive
        for sample_id in range(config.num_samples_per_problem):
            if not check_if_eval_exists_local(problem_id, sample_id, eval_file_path):
                total_work.append((problem_id, sample_id))

    print(
        f"Start evaluation on {len(total_work)} unevaluated samples"
        f" in range: {problem_id_range}"
    )
    # Build Cache on CPU as that is faster (only for local mode)
    if config.build_cache and config.eval_mode == "local":
        compile.batch_compile(total_work, config.to_dict())

    # Batch Eval on multiple GPUs in parallel
    batch_eval(total_work, config, curr_level_dataset, run_dir, eval_file_path)

    # Calculate pass@k metrics if multiple samples per problem were evaluated
    if config.num_samples_per_problem > 1:
        calculate_pass_at_k(eval_file_path, config.pass_at_k_values)


def calc_pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calculate_pass_at_k(eval_file_path: str, k_values: list[int]) -> dict:
    """
    Calculate pass@k metrics from evaluation results.

    pass@k is the probability that at least one of k samples passes (is correct).
    Formula: 1 - (1 - c/n)^k, where c is number of correct samples and n is total samples evaluated.

    Args:
        eval_file_path: Path to evaluation results file
        k_values: List of k values to calculate pass@k for

    Returns:
        Dictionary mapping problem_id to pass@k metrics for each k value
    """
    if not os.path.exists(eval_file_path):
        print(
            f"[WARNING] Evaluation file {eval_file_path} does not exist. Cannot calculate pass@k."
        )
        return {}

    with open(eval_file_path, "r") as f:
        eval_results = json.load(f)

    # Group results by problem_id
    results_by_problem = {}
    for problem_id, result in eval_results.items():
        results_by_problem[problem_id] = result

    # Calculate pass@k for each problem
    pass_at_k_results = {}
    for problem_id, results in results_by_problem.items():
        # Count correct samples
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r["correctness"] and r["compiled"])

        # Calculate pass@k for each k value
        pass_at_k_metrics = {}
        for k in k_values:
            if k > total_samples:
                print(
                    f"[WARNING] k={k} is greater than total samples {total_samples} for problem {problem_id}. Using k={total_samples}."
                )
                k = total_samples

            pass_at_k = calc_pass_at_k(total_samples, correct_samples, k)
            pass_at_k_metrics[f"pass@{k}"] = pass_at_k

        pass_at_k_results[problem_id] = {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            **pass_at_k_metrics,
        }

    # Calculate average pass@k metrics across all problems
    avg_pass_at_k = {}
    total_problems = len(pass_at_k_results)
    if total_problems > 0:
        for k in k_values:
            filtered_results = {
                p: r for p, r in pass_at_k_results.items() if f"pass@{k}" in r
            }
            avg_pass_at_k[f"avg_pass@{k}"] = float(
                sum(result[f"pass@{k}"] for result in filtered_results.values())
                / total_problems
            )

    # Add metadata about the evaluation
    metadata = {
        "total_problems": total_problems,
        "problems_with_samples": len(
            [p for p, r in pass_at_k_results.items() if r["total_samples"] > 0]
        ),
        "total_evaluated_samples": sum(
            r["total_samples"] for r in pass_at_k_results.values()
        ),
        "total_correct_samples": sum(
            r["correct_samples"] for r in pass_at_k_results.values()
        ),
    }

    # Add pass@k metadata
    for k in k_values:
        filtered_results = {
            p: r for p, r in pass_at_k_results.items() if f"pass@{k}" in r
        }
        metadata[f"pass@{k}_count"] = len(filtered_results)

    # Construct the final result with averages, individual problem results, and metadata
    final_results = {
        "averages": avg_pass_at_k,
        "metadata": metadata,
        "problems": pass_at_k_results,
    }

    # Write pass@k results to file
    pass_at_k_file_path = os.path.join(
        os.path.dirname(eval_file_path), "pass_at_k_results.json"
    )
    with open(pass_at_k_file_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print the average pass@k metrics
    print(f"Pass@k Correctness metrics calculated and saved to {pass_at_k_file_path}")
    print(f"Evaluation metadata: {metadata}")
    print(f"Average pass@k metrics: {avg_pass_at_k}")

    return final_results


if __name__ == "__main__":
    main()
