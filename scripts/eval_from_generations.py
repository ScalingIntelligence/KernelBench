from dataclasses import dataclass
import time
import pydra
from pydra import REQUIRED, Config

import json
from tqdm import tqdm
from src import eval, utils
import torch
import os
import multiprocessing as mp


from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref, KernelExecResult, check_metadata_serializable
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch, read_file, create_inference_server_from_presets, maybe_multithread

"""
Batch Eval from Existing Generations

Usually with eval, we check
- correctness: 5 randomized input trials
- performance: 100 randomized input trials

TODO: add CPU Cache building (already exist, need to migrate)

You can increase the number of trials
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class EvalConfig(Config):
    def __init__(self):

        self.run_name = REQUIRED # name of the run to evaluate

        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED

        # subset of problems to evaluate
        self.subset = (None, None) # (problem_id, problem_name), these are the logical index

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local"

        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]


        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")
        # TODO: migrate CPU build cache code to speed up eval

        self.verbose = False

        # Eval settings
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 200
        self.measure_performance = True

        
        # number of GPUs to do batch evaluation
        self.num_gpu_devices = 1


    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int
    sample_id: int
    # run_name: str
    # dataset: list[str]
    device: torch.device


def fetch_ref_arch_from_problem_id(dataset, problem_id: int, dataset_src: str) -> str | None:
    """
    Fetch reference architecture from problem directory
    Either from Hugging Face or Local Dataset
    """
    if dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    
    elif dataset_src == "local":
        problem_idx_in_dataset = problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # verify
        # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    
    return ref_arch_src




def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> str | None:
    """
    Fetch kernel file from disk (stored in runs/{run_name})
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    
    if os.path.exists(kernel_path):
        return read_file(kernel_path)
    else:
        return None

def evaluate_single_sample(work_args: WorkArgs, configs: EvalConfig, dataset, run_dir: str) -> KernelExecResult | None:
    """
    Evaluate a single sample on a single GPU
    """
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    # fetch reference architecture from problem directory
    ref_arch_src = fetch_ref_arch_from_problem_id(dataset, problem_id, configs.dataset_src)

    # fetch kernel from disk
    # Add database support in the future
    kernel_src = fetch_kernel_from_disk(run_dir, configs.level, problem_id, sample_id)

    assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

    build_dir = os.path.join(configs.kernel_eval_build_dir, configs.run_name, f"{problem_id}", f"{sample_id}")

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
                "device": device,
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
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
            print(
                "\n [Terminate] Caught KeyboardInterrupt, terminating workers..."
            )
            pool.terminate()
            pool.join()
            raise
        except mp.TimeoutError as e:
            print(f"[WARNING] Evaluation TIMED OUT for Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}")

        print(f"[Eval Result] Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}: {result}")
        return result

def batch_eval(
    total_work: list[tuple[int, int, int]],
    run_name: str,
    dataset: list[str],
    configs: dict,
):
    """
    Batch evaluation across multiple GPUs
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # construct a list of work args
    num_gpu_devices = configs.get("num_gpu_devices", torch.cuda.device_count())
    batch_size = num_gpu_devices

    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]  # pop the first batch_size elements
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks over {num_gpu_devices} GPUs; [Total Work left] {len(total_work)}"
            )

            assert len(curr_work_batch) <= num_gpu_devices

            with mp.Pool(num_gpu_devices) as pool:

                work_args = [
                    (
                        WorkArgs(
                            problem_id=p_id,
                            sample_id=s_idx,
                            run_name=run_name,
                            dataset=dataset,
                            device=torch.device(f"cuda:{i%batch_size}"),
                        ),
                        configs,
                    )
                    for i, (p_id, s_idx, k_id) in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(
                        pool.apply_async(evaluate_single_sample, work_arg)
                    )

                # Collect results with individual timeouts
                results = []
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id, kernel_id = curr_work_batch[i]

                    try:
                        result = async_result.get(
                            timeout=configs["timeout"]
                        )  # 5 minutes timeout per evaluation
                        results.append((problem_id, sample_id, kernel_id, result))
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_id}"
                        )
                        results.append((problem_id, sample_id, kernel_id, None))
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}"
                        )
                        results.append((problem_id, sample_id, kernel_id, None))

                        # results.append(None)
                # results = pool.starmap(
                #     evaluate_single_sample,
                #     work_args
                # )
                end_time = time.time()

                for problem_id, sample_id, kernel_id, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_id}, Kernel ID: {kernel_id}"
                    )
                    print(result)
                print("-" * 128)
                print(
                    f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))


def check_if_eval_exists_local(problem_id: int, sample_id: int, eval_file_path: str) -> bool:
    """
    Check if evaluation result already exists in eval results file
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
        return str(problem_id) in eval_results
    return False

def add_to_eval_results_file(problem_id: int, sample_id: int, eval_result: KernelExecResult, eval_file_path: str):
    """
    Add evaluation result to eval results file
    TODO: migrate database support
    """
    # Load existing results if file exists
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
    
    # Add new result
    eval_results[str(problem_id)] = {
        # assume 1 sample for now, will think about how to do this better for more samples
        'sample_id': sample_id,
        'compiled': eval_result.compiled,
        'correctness': eval_result.correctness,
        'metadata': check_metadata_serializable(eval_result.metadata),
        'runtime': eval_result.runtime,
        'runtime_stats': eval_result.runtime_stats,
    }
    
    # Write updated results back to file
    if not os.path.exists(eval_file_path):
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f)

def single_eval_example(config: EvalConfig, curr_level_dataset: list[str], run_dir: str, eval_file_path ):
    device = torch.device("cuda:0")
    example_work = WorkArgs(problem_id=1, sample_id=0, device=device)
    # example_eval_result = evaluate_single_sample(example_work, config, curr_level_dataset, run_dir)
    example_eval_result = cuda_single_eval_wrapper(example_work, config, curr_level_dataset, run_dir)
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
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

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
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Evaluating 1 sample each for level {config.level} problems: {problem_id_range}")

    run_dir = os.path.join(config.runs_dir, config.run_name)
    eval_file_path = os.path.join(run_dir, f"eval_results.json")


    # set GPU arch to configure what target to build for
    set_gpu_arch(config.gpu_arch)

    # To Debug
    # single_eval_example(config, curr_level_dataset, run_dir, eval_file_path)

    total_work = []
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        sample_id = 0 # only evaluate 1 sample for now
        if not check_if_eval_exists_local(problem_id, sample_id, eval_file_path):
            total_work.append((problem_id, sample_id))

    print(f"Start evaluation on {len(total_work)} unevaluated samples in range: {problem_id_range}")
    
    # Batch Eval on multiple GPUs in parallel
    # batch_eval(total_work, RUN_NAME, dataset, configs)


if __name__ == "__main__":
    main()
  