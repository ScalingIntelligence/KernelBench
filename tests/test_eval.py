from dataclasses import dataclass
from src import eval, utils
import torch

import multiprocessing as mp

MEASURE_PERFORMANCE = False

# RUN_NAME = "kernelbench_prompt_v2_level_2"
RUN_NAME = "level2_run_10_28"
PROBLEM_DIR = "KernelBench/level2"
# query from database, make sure the server is up
SERVER_URL = "http://mkt1.stanford.edu:9091" 

# sample_id = 2


# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available. This test requires a GPU.")

device = torch.cuda.current_device()
print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")


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

# NOTE: this should isolate each run
# however, let me mirgate the monkey codebase code to do this instaed of my little hack
# DON'T USE THIS YET! just run and see how it would be 


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
    kernel_src = eval.fetch_kernel_from_database(run_name, problem_id, sample_id, SERVER_URL)
    assert kernel_src is not None, f"Kernel not found for sample {sample_id}"
    try:
        eval_result = eval.eval_kernel_against_ref(
        original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=MEASURE_PERFORMANCE,
            verbose=True,
            num_times=5,
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



def monkey_style_parallal_process_eval():
    """
    Same API as monkey maybe_multi_processing
    This doesn't work yet, suffer the same cascading errors as before
    """
    to_run =  []
    for sample_idx in range(5, 10):
        
        to_run.append(
            WorkArgs(
                problem_id=problem_id,
                sample_idx=sample_idx,
                run_name=RUN_NAME,
                dataset=dataset,
                device=device
            )
        )

    # WHY DOES THIS !!!NOT!!! WORK?
    utils.maybe_multiprocess(
        func=run,
        instances=to_run,
        # only limited to 1 worker for now, don't worry about concurrecy
        num_workers=1, 
    )

def multiprocess_eval(problem_id: int, samples_range: tuple[int, int]):
    """
    This works
    """
    
    # THIS WORKS
    # Set start method to spawn to work with CUDA
    mp.set_start_method('spawn')

    with open(f"results/eval_result_problem_{problem_id}.txt", "a") as f:
        f.write(f"Evaluating for problem {problem_id} over sample range {samples_range} \n")

    # Main evaluation loop
    for sample_id in range(*samples_range):

        print(f"Evaluating for sample {sample_id}")
        curr_work = WorkArgs(problem_id=problem_id, sample_idx=sample_id, run_name=RUN_NAME, dataset=dataset, device=device, num_times=5)

        # Create a new process for each evaluation
        with mp.Pool(1) as pool:
            result = pool.apply(
                evaluate_single_sample,
                args=(curr_work,)
                # (problem_id, sample_id, RUN_NAME, dataset, device)
            )
        with open(f"results/eval_result_problem_{problem_id}.txt", "a") as f:
            f.write("-" * 128 + "\n")
            f.write(f"Eval result for sample {sample_id}: {result}\n") 

if __name__ == "__main__":
    problem_id = 2
    # samples_range = (4, 5)
    samples_range = (0, 30)
    
    multiprocess_eval(problem_id, samples_range)
   



