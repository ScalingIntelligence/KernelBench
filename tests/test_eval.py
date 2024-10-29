from src import eval, utils
import torch

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

print(f"[Curr Eval] Evaluating Kernel for Run {RUN_NAME} on Problem {problem_id} with Sample {sample_id} on CUDA device {device}: {torch.cuda.get_device_name(device)}")

# fetch reference architecture from problem directory
dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
ref_arch_src = eval.fetch_ref_arch_from_problem_id(problem_id, dataset)

# fetch kernel code from database
kernel_src = eval.fetch_kernel_from_database(RUN_NAME, problem_id, sample_id, SERVER_URL)
assert kernel_src is not None, "Kernel not found"

# evaluate kernel

try:
    eval_result = eval.eval_kernel_against_ref(original_model_src=ref_arch_src, 
                                               custom_model_src=kernel_src, 
                                               measure_performance=MEASURE_PERFORMANCE,
                                               verbose=True,
                                               device=device)
except Exception as e:
    print(f"Some issue evaluating for kernel: {e} ")






