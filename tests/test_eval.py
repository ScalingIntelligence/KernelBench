from dataclasses import dataclass

from tqdm import tqdm
from src import eval, utils
import torch
import os
import multiprocessing as mp

MEASURE_PERFORMANCE = False

RUN_NAME = "level2_run_10_28"
# RUN_NAME = "kernelbench_prompt_v2_level_2"
RUN_NAME = "level2_run_10_28"
PROBLEM_DIR = "KernelBench/level2"
# query from database, make sure the server is up
SERVER_URL = "http://matx3.stanford.edu:9091" 
# SERVER_URL = "http://localhost:9091"

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
    num_times: int


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


def evaluate_single_sample(work_args: WorkArgs):
    # problem_id, sample_id, run_name, dataset, device
    problem_id = work_args.problem_id
    sample_id = work_args.sample_idx
    run_name = work_args.run_name
    dataset = work_args.dataset
    device = work_args.device
    num_times = work_args.num_times

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
            measure_performance=MEASURE_PERFORMANCE,
            verbose=True,
            num_times=num_times,
            # move this to config in monkeys
            build_dir=f"/matx/u/simonguo/kernel_eval_build/{run_name}/{problem_id}/{sample_id}",
            device=device
        )
        return eval_result
    except Exception as e:
        print(f"THIS SHOULD NOT PRINT for sample {sample_id}: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e):
            # TODO: should we count it like this?
            eval_result = eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata={"cuda_error": f"CUDA Error: {str(e)}"})
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
                num_times=5
            )
        )

    # WHY DOES THIS !!!NOT!!! WORK?
    utils.maybe_multiprocess_cuda(
        func=run,
        instances=to_run,
        # only limited to 1 worker for now, don't worry about concurrecy
        num_workers=1, 
    )

def multiprocess_cuda_eval(problem_range: tuple[int, int], samples_range: tuple[int, int]):
    """
    This works
    """
    
    # THIS WORKS
    # Set start method to spawn to work with CUDA
    mp.set_start_method('spawn')

    device = torch.device("cuda:1")
    print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")


    for problem_id in tqdm(range(*problem_range)):
    # Main evaluation loop

        os.makedirs("results", exist_ok=True)
        with open(f"results/eval_result_problem_{problem_id}.txt", "a") as f:
            f.write(f"Evaluating for problem {problem_id} over sample range {samples_range} \n")

        for sample_id in tqdm(range(*samples_range)):

            
            print(f"Evaluating for problem {problem_id} sample {sample_id}")
            curr_work = WorkArgs(problem_id=problem_id, sample_idx=sample_id, run_name=RUN_NAME, dataset=dataset, device=device, num_times=5)

            # Create a new process for each evaluation
            with mp.Pool(1) as pool:
                try:
                    result = pool.apply_async(
                        evaluate_single_sample,
                        args=(curr_work,),
                    ).get(timeout=300)
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

if __name__ == "__main__":
    # problem_id = 7
    # samples_range = (4, 5)
    problem_range = (9, 54)
    samples_range = (0, 2) # 30 samples
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. This test requires a GPU.")

    # for problem_id in range(8, 54):
    multiprocess_cuda_eval(problem_range, samples_range)
    # monkey_style_parallal_process_eval(problem_id, samples_range)



