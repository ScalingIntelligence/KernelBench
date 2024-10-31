from dataclasses import dataclass
import time

from tqdm import tqdm
from src import eval, utils
import torch
import os
import multiprocessing as mp


# Global Configs
 
MEASURE_PERFORMANCE = True

RUN_NAME = "level2_run_10_28"
# RUN_NAME = "kernelbench_prompt_v2_level_2"
# RUN_NAME = "level2_run_10_28"
PROBLEM_DIR = "KernelBench/level2"
# query from database, make sure the server is up
SERVER_URL = "http://matx3.stanford.edu:9091" 
# SERVER_URL = "http://localhost:9091"

NUM_GPU_DEVICES = 6

torch.set_printoptions(precision=4, threshold=10)

dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

# NOTE: If you run this this will have cascading errors from previous iterations
# # evaluate kernel (for some samples)
# for sample_id in range(1,5):
#     print(f"Evaluating for sample {sample_id}")
#     print(f"[Curr Eval] Evaluating Kernel for Run {RUN_NAME} on Problem {problem_id} with Sample {sample_id} on CUDA device {device}: {torch.cuda.get_device_name(device)}")

#     # fetch reference architecture from problem directory
#     ref_arch_src = eval.fetch_ref_arch_from_problem_id(problem_id, dataset)

#     # fetch kernel code from database
#     kernel_src = eval.fetch_kernel_from_database(RUN_NAME, problem_id, sample_id, SERVER_URL)
#     assert kernel_src is not None, f"Kernel not found for sample {sample_id}"

#     try:
#         eval_result = eval.eval_kernel_against_ref(original_model_src=ref_arch_src, 
#                                                custom_model_src=kernel_src, 
#                                                measure_performance=MEASURE_PERFORMANCE,
#                                                verbose=True,
#                                                device=device)
#         print("-" * 32)
#         print(f"Eval result for sample {sample_id}: {eval_result}")
#         print("-" * 32)
#     except Exception as e:
#         print(f"THIS SHOULD NOT PRINT for sample {sample_id}: Some issue evaluating for kernel: {e} ")
#     finally:
#         torch.cuda.empty_cache()


@dataclass
class WorkArgs:
    problem_id: str
    sample_idx: int
    run_name: str
    dataset: list[str]
    device: torch.device


def run(work, config=None, coordinator=None):
    """
    Matching Monkey API, took out some args for config and coordinator
    """
    run_inner = evaluate_single_sample
    # if config.testing: return run_inner(work)
    try:
        eval_result = run_inner(work)
        with open(f"results/eval_result_problem_{work.problem_id}.txt", "a") as f:
            f.write("-" * 128 + "\n")
            f.write(f"Eval result for sample {work.sample_idx}: {eval_result}\n") 

        print("-" * 32)
        print(f"Eval result for problem {work.problem_id} sample {work.sample_idx}: {eval_result}")
        print("-" * 32)
    except Exception as e:
        print("Error", e, work.problem, work.problem_id, work.sample_idx)
        return None


def evaluate_single_sample(work_args: WorkArgs, configs: dict):
    # problem_id, sample_id, run_name, dataset, device
    problem_id, sample_id, run_name, dataset, device = work_args.problem_id, work_args.sample_idx, work_args.run_name, work_args.dataset, work_args.device
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]    
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    # fetch reference architecture from problem directory
    ref_arch_src = eval.fetch_ref_arch_from_problem_id(problem_id, dataset)
    
    # fetch kernel code from database
    kernel = eval.fetch_kernel_from_database(run_name, problem_id, sample_id, SERVER_URL)
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
            device=device
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": device
                        } # for debugging
            eval_result = eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        return None



def monkey_style_parallal_process_eval(problem_id: int, samples_range: tuple[int, int]):
    """"
    Same API as monkey maybe_multi_processing
    This doesn't work yet, suffer the same cascading errors as before
    """
    to_run =  []
    for sample_idx in range(*samples_range):
        
        to_run.append(
            WorkArgs(
                problem_id=problem_id,
                sample_idx=sample_idx,
                run_name=RUN_NAME,
                dataset=dataset,
                # device=device, 
                num_correct_trials=5
            )
        )

    # WHY DOES THIS !!!NOT!!! WORK?
    utils.maybe_multiprocess_cuda(
        func=run,
        instances=to_run,
        # only limited to 1 worker for now, don't worry about concurrecy
        num_workers=1, 
    )

def cuda_eval_process(problem_range: tuple[int, int], samples_range: tuple[int, int], configs: dict):
    """
    This works, but one at at time
    """
    # THIS WORKS
    # Set start method to spawn to work with CUDA
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    device = torch.device("cuda:1")
    print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")
    
    timeout = configs["timeout"]

    for problem_id in tqdm(range(*problem_range)):
    # Main evaluation loop

        os.makedirs("results", exist_ok=True)
        with open(f"results/eval_result_problem_{problem_id}.txt", "a") as f:
            f.write(f"Evaluating for problem {problem_id} over sample range {samples_range} \n")

        for sample_id in tqdm(range(*samples_range)):

            
            print(f"Evaluating for problem {problem_id} sample {sample_id}")
            curr_work = WorkArgs(problem_id=problem_id, sample_idx=sample_id, run_name=RUN_NAME, dataset=dataset, device=device)

            # Create a new process for each evaluation
            with mp.Pool(1) as pool:
                try:
                    result = pool.apply_async(
                        evaluate_single_sample,
                        args=(curr_work, configs),
                    ).get(timeout=timeout)
                except KeyboardInterrupt:
                    print("\n [Terminate] Caught KeyboardInterrupt, terminating workers...")
                    pool.terminate()
                    pool.join()
                    raise
                except mp.TimeoutError as e:
                    with open(f"results/eval_result_problem_{problem_id}.txt", "a") as f:
                        f.write("-" * 128 + "\n")
                        f.write(f"Eval result for sample {sample_id}: timed out\n") 
                    continue

            with open(f"results/eval_result_problem_{problem_id}.txt", "a") as f:
                f.write("-" * 128 + "\n")
                f.write(f"Eval result for sample {sample_id}: {result}\n") 


def batch_eval(total_work: list[tuple[int, int, int]], run_name: str, dataset: list[str], configs: dict):
    """
    Batch evaluation across multiple GPUs
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # construct a list of work args
    num_gpu_devices = configs.get("num_gpu_devices", torch.cuda.device_count())
    batch_size = num_gpu_devices
    
    with tqdm(total=len(total_work), desc="Processing batches") as pbar:

        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:] # pop the first batch_size elements
            print(f"[Curr Batch] {len(curr_work_batch)} tasks over {num_gpu_devices} GPUs; [Total Work left] {len(total_work)}")

            assert len(curr_work_batch) <= num_gpu_devices

            with mp.Pool(num_gpu_devices) as pool:

                work_args = [
                    (WorkArgs(problem_id=p_id, sample_idx=s_idx, run_name=run_name, dataset=dataset, device=torch.device(f"cuda:{i%batch_size}")), configs)
                    for i, (p_id, s_idx, k_id) in enumerate(curr_work_batch)
                ]

                start_time = time.time()

                async_results = []
                for work_arg in work_args:
                    async_results.append(pool.apply_async(evaluate_single_sample, work_arg))

                # Collect results with individual timeouts
                results = []
                for i, async_result in enumerate(async_results):
                    problem_id, sample_idx, kernel_id = curr_work_batch[i]

                    try:
                        result = async_result.get(timeout=configs["timeout"])  # 5 minutes timeout per evaluation
                        results.append((problem_id, sample_idx, kernel_id, result))
                    except mp.TimeoutError:
                        print(f"[WARNING] Evaluation TIMED OUT for Problem ID: {problem_id}, Sample ID: {sample_idx}")
                        results.append((problem_id, sample_idx, kernel_id, None))
                    except Exception as e:
                        print(f"[ERROR] Evaluation FAILED for Problem ID: {problem_id}, Sample ID: {sample_idx}: {str(e)}")
                        results.append((problem_id, sample_idx, kernel_id, None))

                        # results.append(None)
                # results = pool.starmap(
                #     evaluate_single_sample,
                #     work_args
                # )
                end_time = time.time()

                for problem_id, sample_idx, kernel_id, result in results:
                    print("-" * 128)
                    print(f"[Eval Result] Problem ID: {problem_id}, Sample ID: {sample_idx}, Kernel ID: {kernel_id}")
                    print(result)
                print("-" * 128)
                print(f"[Curr batch] Evaluation took {end_time - start_time:.2f} seconds")

                pbar.update(len(curr_work_batch))


if __name__ == "__main__":
    # problem_id = 7
    # samples_range = (4, 5)
    # problem_range = (15, 54)
    # samples_range = (2, 10) # 30 samples
    
    # problem_range = (15, 16)
    # samples_range = (0, 1)
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. This test requires a GPU.")
    
    # these will go into pydra in the future
    configs = {"num_correct_trials": 5, "num_perf_trials": 100, "timeout": 20, "verbose": False, "num_gpu_devices": NUM_GPU_DEVICES, "measure_performance": True}

    problem_range = (3, 4)
    samples_range = (0, 6)

    # this works great, launch process one at a time
    # cuda_eval_process(problem_range, samples_range, configs)

    # batch eval, in our experiment server it will be replaced by fetching from database
    total_work = [] # a list of (problem_id, sample_id)
    for problem_id in range(*problem_range):
        for sample_id in range(*samples_range):
            kernel_id = -1 # fake example
            total_work.append((problem_id, sample_id, kernel_id))
    
    
    # # this does it in a batch manner
    batch_eval(total_work, RUN_NAME, dataset, configs)

    
    # use this to debug (e.g. pdb)
    # device = torch.device("cuda:1")
    # evaluate_single_sample(WorkArgs(problem_id=15, sample_idx=0, run_name=RUN_NAME, dataset=dataset, device=device, num_correct_trials=5))





    # this doesn't work fully yet
    # monkey_style_parallal_process_eval(problem_id, samples_range)