import torch
import numpy as np
from src.eval import load_original_model_and_inputs, time_execution_with_cuda_event, get_timing_stats, set_seed, fetch_ref_arch_from_problem_id
from src.utils import construct_problem_dataset_from_problem_dir
import os

device = torch.device("cuda:0")

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

def fetch_ref_arch_from_level_problem_id(level_num, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, 'level'+str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)

def get_time(level_num, problem_id, num_trials=50):
    ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(level_num, problem_id, with_name=True)
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_arch_src, context)
    try: 
        torch.cuda.synchronize(device=device)
        set_seed(42)
        inputs = get_inputs()
        set_seed(42)
        init_inputs = get_init_inputs()
        inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
        model = Model(*init_inputs)
        model = model.cuda(device=device)
        torch.cuda.synchronize(device=device)
        elapsed_times = time_execution_with_cuda_event(model, *inputs, num_trials=num_trials, verbose=False, device=device)
        runtime_stats = get_timing_stats(elapsed_times, device=device)
        print(f"Level {level_num} Problem {problem_id} Runtime Stats: {runtime_stats}")
        # save stats to results/baseline_time.txt
        with open(f"results/baseline_time.txt", "a") as f:
            f.write(f"Level {level_num} Problem {problem_id} Runtime Stats: {runtime_stats}\n")
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")

if __name__ == "__main__":

    REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
    KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

    PROBLEM_DIR_LEVEL1 = "KernelBench/level1"
    dataset_level1 = construct_problem_dataset_from_problem_dir(PROBLEM_DIR_LEVEL1)
    for problem_id in range(len(dataset_level1)):
        get_time(1, problem_id)

    PROBLEM_DIR_LEVEL2 = "KernelBench/level2"
    dataset_level2 = construct_problem_dataset_from_problem_dir(PROBLEM_DIR_LEVEL2)
    for problem_id in range(len(dataset_level2)):
        get_time(2, problem_id)

    PROBLEM_DIR_LEVEL3 = "KernelBench/level3"
    dataset_level3 = construct_problem_dataset_from_problem_dir(PROBLEM_DIR_LEVEL3)
    for problem_id in range(len(dataset_level3)):
        get_time(3, problem_id)

