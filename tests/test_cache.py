from dataclasses import dataclass
import time

from tqdm import tqdm
from src import eval, utils
import torch
import os
import multiprocessing as mp


# Global Configs
 
MEASURE_PERFORMANCE = True

# RUN_NAME = "level2_run_10_28"
RUN_NAME = "level1_trial_11_01_sonnet"
# RUN_NAME = "kernelbench_prompt_v2_level_2"
# RUN_NAME = "level2_run_10_28"
PROBLEM_DIR = "KernelBench/level1"
# query from database, make sure the server is up
SERVER_URL = "http://matx3.stanford.edu:9091" 
# SERVER_URL = "http://localhost:9091"

NUM_CPU_WORKERS = 50
# NUM_GPU_DEVICES = 6

torch.set_printoptions(precision=4, threshold=10)

dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)


@dataclass
class WorkArgs:
    problem_id: str
    sample_idx: int
    run_name: str
    dataset: list[str]
    device: torch.device


def compile_single_sample(work_args: WorkArgs, configs: dict):
    # problem_id, sample_id, run_name, dataset, device
    problem_id = work_args.problem_id
    sample_id = work_args.sample_idx
    run_name = work_args.run_name
    # dataset = work_args.dataset
    # device = work_args.device
    verbose = configs["verbose"]


    # fetch kernel code from database
    kernel = eval.fetch_kernel_from_database(run_name, problem_id, sample_id, SERVER_URL)
    kernel_src = kernel["kernel"]
    kernel_hash = kernel["kernel_hash"]
    assert kernel_src is not None, f"Kernel not found for sample {sample_id}"
    try:
        compiled_and_cached = eval.build_compile_cache(custom_model_src=kernel_src,
                                                       custom_model_hash=kernel_hash, 
            verbose=verbose, 
            build_dir=f"/matx/u/simonguo/kernel_eval_build/{run_name}/{problem_id}/{sample_id}")

        return compiled_and_cached
    except Exception as e:
        print(f"[WARNING] Last level catch on {sample_id}: Some issue while compiling and attempting to cache for kernel: {e} ")
        return None


def batch_compile(problem_range: tuple[int, int], samples_range: tuple[int, int], configs: dict):
    """
    Batch compile cache across CPUs
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')


    # construct a list of work args
    total_work = []
    for problem_id in range(*problem_range):
        for sample_id in range(*samples_range):
            total_work.append((problem_id, sample_id))
    try: 
        with mp.Pool(NUM_CPU_WORKERS) as pool:
            # Create work args for each task
            work_args = [
                (WorkArgs(problem_id=p_id, sample_idx=s_idx, run_name=RUN_NAME, dataset=dataset, device=None), configs)
                for p_id, s_idx in total_work
            ]

            # Start async tasks
            async_results = []
            for work_arg in work_args:
                async_results.append(pool.apply_async(compile_single_sample, work_arg))

            # Collect results with timeouts
            results = []
            for i, async_result in enumerate(tqdm(async_results, desc="Compile & Cache Progress")):
                try:
                    result = async_result.get(timeout=configs["timeout"])
                    # TODO: do something with this result?
                    results.append(result)
                except mp.TimeoutError:
                    problem_id, sample_id = total_work[i]
                    print(f"[WARNING] Compilation timed out for Problem ID: {problem_id}, Sample ID: {sample_id}. Cannot cache this kernel")
                    results.append(None)
                except Exception as e:
                    problem_id, sample_id = total_work[i]
                    print(f"[ERROR] Compilation failed for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}")
                    results.append(None)

            return results

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Terminating workers...")
        pool.terminate()
        pool.join()
        raise
    finally:
        pool.close()
    # while len(total_work) > 0:
    #     curr_work_batch = total_work[:NUM_GPU_DEVICES]
    #     total_work = total_work[NUM_GPU_DEVICES:]
    #     print(f"[Total Work] {len(total_work)}")

    #     assert len(curr_work_batch) <= NUM_GPU_DEVICES

    #     with mp.Pool(NUM_GPU_DEVICES) as pool:

    #         work_args = [
    #             (WorkArgs(problem_id=p_id, sample_idx=s_idx, run_name=RUN_NAME, dataset=dataset, device=torch.device(f"cuda:{i%NUM_GPU_DEVICES}")), configs)
    #             for i, (p_id, s_idx) in enumerate(curr_work_batch)
    #         ]


    #         start_time = time.time()

    #         async_results = []
    #         for work_arg in work_args:
    #             async_results.append(pool.apply_async(evaluate_single_sample, work_arg))

    #         # Collect results with individual timeouts
    #         results = []
    #         for i, async_result in enumerate(async_results):
    #             try:
    #                 result = async_result.get(timeout=configs["timeout"])  # 5 minutes timeout per evaluation
    #                 results.append(result)
    #             except mp.TimeoutError:
    #                 problem_id, sample_idx = curr_work_batch[i]
    #                 print(f"[WARNING] Evaluation timed out for Problem ID: {problem_id}, Sample ID: {sample_idx}")
    #                 results.append(None)
    #             except Exception as e:
    #                 problem_id, sample_idx = curr_work_batch[i]
    #                 print(f"[ERROR] Evaluation failed for Problem ID: {problem_id}, Sample ID: {sample_idx}: {str(e)}")
    #                 results.append(None)
    #         # results = pool.starmap(
    #         #     evaluate_single_sample,
    #         #     work_args
    #         # )
    #         end_time = time.time()

    #         for result in results:
    #             print("-" * 128)
    #             problem_id, sample_idx = curr_work_batch[results.index(result)]
    #             print(f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_idx}")
    #             print(result)
    #         print("-" * 128)
    #         print(f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds")




if __name__ == "__main__":
    # problem_id = 7
    # samples_range = (4, 5)
    # problem_range = (15, 54)
    # samples_range = (2, 10) # 30 samples
    
    # problem_range = (15, 16)
    # samples_range = (0, 1)
    # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA device not available. This test requires a GPU.")
    
    # these will go into pydra in the future
    configs = {"timeout": 900, "verbose": False}

    # problem_range = (3, 4)
    # samples_range = (0, 15)

    # problem_range = (0, 45)
    # problem_range = (45, 87)
    # problem_range = (39, 54)
    # samples_range = (0, 30)
    problem_range = (0, len(dataset))
    samples_range = (0, 10)

    # compile_single_sample(WorkArgs(problem_id=40, sample_idx=0, run_name=RUN_NAME, dataset=dataset, device=None), configs)
    # evaluate_single_sample(WorkArgs(problem_id=15, sample_idx=0, run_name=RUN_NAME, dataset=dataset, device=device, num_correct_trials=5))

    batch_compile(problem_range, samples_range, configs)