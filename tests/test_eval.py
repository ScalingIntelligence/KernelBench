from src import eval, utils
import torch

import multiprocessing as mp

MEASURE_PERFORMANCE = False

RUN_NAME = "kernelbench_prompt_v2_level_2"
PROBLEM_DIR = "KernelBench/level2"
# query from database, make sure the server is up
SERVER_URL = "http://mkt1.stanford.edu:9091" 

problem_id = 1
sample_id = 2


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
def evaluate_single_sample(problem_id, sample_id, run_name, dataset, device):
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

if __name__ == "__main__":
    
    # Set start method to spawn to work with CUDA
    mp.set_start_method('spawn')

    # Main evaluation loop
    for sample_id in range(5, 7):
        print(f"Evaluating for sample {sample_id}")
        
        # Create a new process for each evaluation
        with mp.Pool(1) as pool:
            result = pool.apply(
                evaluate_single_sample,
                args=(problem_id, sample_id, RUN_NAME, dataset, device)
            )
    
        print("-" * 32)
        print(f"Eval result for sample {sample_id}: {result}")
        print("-" * 32)




