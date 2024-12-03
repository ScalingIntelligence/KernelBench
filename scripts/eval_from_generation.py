from dataclasses import dataclass
import time
import pydra
from pydra import REQUIRED, Config

from tqdm import tqdm
from src import eval, utils
import torch
import os
import multiprocessing as mp

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
    problem_id: str
    sample_idx: int
    run_name: str
    dataset: list[str]
    device: torch.device



def evaluate_single_sample(work_args: WorkArgs, configs: dict):
    """
    Evaluate a single sample
    """
    # problem_id, sample_id, run_name, dataset, device
    problem_id, sample_id, run_name, dataset, device = (
        work_args.problem_id,
        work_args.sample_idx,
        work_args.run_name,
        work_args.dataset,
        work_args.device,
    )
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    # fetch reference architecture from problem directory
    ref_arch_src = eval.fetch_ref_arch_from_problem_id(problem_id, dataset)

    # fetch kernel code from database
    kernel = eval.fetch_kernel_from_database(
        run_name, problem_id, sample_id, SERVER_URL
    )
    kernel_src = kernel["kernel"]
    kernel_hash = kernel["kernel_hash"]
    assert kernel_src is not None, f"Kernel not found for sample {sample_id}"
    try:
        eval_result = eval.eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            custom_model_hash=kernel_hash,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            # move this to config in monkeys
            build_dir=f"/matx/u/simonguo/kernel_eval_build/{run_name}/{problem_id}/{sample_id}",
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
            }  # for debugging
            eval_result = eval.KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        return None

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
                            sample_idx=s_idx,
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
                    problem_id, sample_idx, kernel_id = curr_work_batch[i]

                    try:
                        result = async_result.get(
                            timeout=configs["timeout"]
                        )  # 5 minutes timeout per evaluation
                        results.append((problem_id, sample_idx, kernel_id, result))
                    except mp.TimeoutError:
                        print(
                            f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_idx}"
                        )
                        results.append((problem_id, sample_idx, kernel_id, None))
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_idx}: {str(e)}"
                        )
                        results.append((problem_id, sample_idx, kernel_id, None))

                        # results.append(None)
                # results = pool.starmap(
                #     evaluate_single_sample,
                #     work_args
                # )
                end_time = time.time()

                for problem_id, sample_idx, kernel_id, result in results:
                    print("-" * 128)
                    print(
                        f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_idx}, Kernel ID: {kernel_id}"
                    )
                    print(result)
                print("-" * 128)
                print(
                    f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds"
                )

                pbar.update(len(curr_work_batch))



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

    


    # problem_range = (3, 4)
    # samples_range = (0,)

    # # this works great, launch process one at a time
    # # cuda_eval_process(problem_range, samples_range, configs)

    # # batch eval, in our experiment server it will be replaced by fetching from database
    # total_work = []  # a list of (problem_id, sample_id)
    # for problem_id in range(*problem_range):
    #     for sample_id in range(*samples_range):
    #         kernel_id = -1  # fake example
    #         total_work.append((problem_id, sample_id, kernel_id))

    # # # this does it in a batch manner
    # batch_eval(total_work, RUN_NAME, dataset, configs)

    # use this to debug (e.g. pdb)
    # device = torch.device("cuda:1")
    # evaluate_single_sample(WorkArgs(problem_id=15, sample_idx=0, run_name=RUN_NAME, dataset=dataset, device=device, num_correct_trials=5))

    # this doesn't work fully yet
    # monkey_style_parallal_process_eval(problem_id, samples_range)



if __name__ == "__main__":
    main()
  