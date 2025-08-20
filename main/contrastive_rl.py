"""
Contrastive RL implementation (from CUDA-L1 paper)
1. Sample from collection of correct kernels with runtime
2. Prompt model to do comparative analysis and generate new kernel (n times)
3. Evaluate new kernels and GRPO to update model
"""
import os
import sys
import json
import random
import torch
import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from verl.utils.dataset.rl_dataset import RLHFDataset

from src.run_utils import find_highest_sample_id, fetch_baseline_results, write_kernel_to_disk, write_eval_result_to_separate_file
from src.utils import extract_last_code, read_file
from src.eval import KernelExecResult
from src.reward_hacking import is_generated_kernel_used, torch_function_used
from src.dataset import fetch_ref_arch_from_level_problem_id
from src.prompt_constructor import prompt_base

from main.evaluation_utils import send_batch_evaluation_request, EvaluationWorkArgs, serialize_work_args



########################################################
# TODO: change config before each run
# RUNS_DIR = "/data/user_data/gyeongwk/KernelBench/grpo/runs"
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
RUN_NAME = "grpo_contrastive_rl_correct_Qwen2.5-Coder-7B-Instruct-SFT"
EVAL_SERVER_HOST = "babel-11-9"
EVAL_SERVER_PORT = 8085
NUM_GENERATIONS = 8
HARDWARE = "A6000_babel"

DATABASE_PATH = os.path.join(RUNS_DIR, RUN_NAME, "contrastive_rl_database.json")
########################################################
os.makedirs(os.path.join(RUNS_DIR, RUN_NAME), exist_ok=True)
DEP_RUNS_DIR = os.path.join(REPO_ROOT, "runs_v0.0")

# Set up database of correct kernels
def construct_database():
    database = {"1": {}, "2": {}}
    for level in [1, 2]:
        best_k_path = os.path.join(REPO_ROOT, "KernelBench", f"best_k_kernels_level{level}.json")
        with open(best_k_path, "r") as f:
            level_kernels = json.load(f)
        for prob_id, kernels in level_kernels.items():
            baseline_time = fetch_baseline_results(level, prob_id, HARDWARE)["mean"]
            data = []
            for kernel in kernels:
                kernel_path = os.path.join(DEP_RUNS_DIR, kernel["run_name"], f"level_{level}_problem_{prob_id}_sample_{kernel['sample_id']}_kernel.py")
                kernel_src = read_file(kernel_path)
                data.append({"speedup": baseline_time / kernel["runtime"], "kernel_src": kernel_src})
            database[str(level)][prob_id] = data
    
    with open(DATABASE_PATH, "w") as f:
        json.dump(database, f, indent=2)


def sample_from_database(level, problem):
    with open(DATABASE_PATH, "r") as f:
        database = json.load(f)
    
    kernels = database[str(level)][str(problem)]

    # Sample 2 kernels weighted by speedup
    # Prepare weights based on speedup
    speedups = [k["speedup"] for k in kernels]
    total_speedup = sum(speedups)
    if total_speedup == 0:
        # fallback to uniform if all speedups are zero
        weights = [1.0 for _ in kernels]
    else:
        weights = [s / total_speedup for s in speedups]

    # Deal with edge cases
    if len(kernels) == 0:
        print(f"[DB] Problem {problem} has 0 correct kernels. Using reference.")
        ref_kernel = fetch_ref_arch_from_level_problem_id(level, problem, "local")
        return ref_kernel, 1.0, ref_kernel, 1.0

    if len(kernels) == 1:
        print(f"[DB] Problem {problem} has 1 correct kernel. Duplicating.")
        return kernels[0]["kernel_src"], kernels[0]["speedup"], kernels[0]["kernel_src"], kernels[0]["speedup"]

    # Sample 2 distinct indices without replacement, weighted by speedup
    indices = random.choices(range(len(kernels)), weights=weights, k=2)
    # Ensure distinctness
    while indices[0] == indices[1]:
        indices[1] = random.choices(range(len(kernels)), weights=weights, k=1)[0]

    kernel1 = kernels[indices[0]]
    kernel2 = kernels[indices[1]]
    return kernel1["kernel_src"], kernel1["speedup"], kernel2["kernel_src"], kernel2["speedup"]



def add_to_database(batch_result):
    with open(DATABASE_PATH, "r") as f:
        database = json.load(f)

    for key, reward in batch_result.items():
        level, problem, sample_id = key.split("_")
        level = int(level)
        problem = int(problem)
        sample_id = int(sample_id)
        kernel_src = read_file(os.path.join(RUNS_DIR, RUN_NAME, f"level_{level}_problem_{problem}_sample_{sample_id}_kernel.py"))
        speedup = reward[0] - 0.3

        database[str(level)][str(problem)].append({"speedup": speedup, "kernel_src": kernel_src})

    with open(DATABASE_PATH, "w") as f:
        json.dump(database, f)


construct_database()
    

def prompt_contrastive_rl(ref_arch_src: str, kernel_src1, speedup1, kernel_src2, speedup2):
    prompt = "# Task for CUDA optimization \nYou are a CUDA programming expert specializing in GPU kernel optimization. Given a reference CUDA implementation, your objective is to create an accelerated version that maintains identical functionality. You will receive previous CUDA implementations accompanied by their performance metrics. Conduct a comparative analysis of these implementations and use the insights to develop optimized and correct CUDA code that delivers superior performance"

    prompt += f"""# Reference Code
    {ref_arch_src}

# Previous CUDA Implementations with Scores
// code1 ({speedup1})
{kernel_src1}

// code2 ({speedup2})
{kernel_src2}

# Generation Protocol
You MUST use exactly two hash symbols (##) at the beginning of each section.
## Performance Analysis: Compare code snippets and articulate on how to improve the code.
    1. Which implementations demonstrate superior performance and why?
    2. What particular optimization strategies exhibit the greatest potential for improvement?
    3. What are the primary performance limitations in the implementation?
    4. What CUDA-specific optimization techniques remain unexploited?
    5. Where do the most significant acceleration opportunities exist?
## Algorithm Design: Describe your optimization approach
## Code Implementation: Provide your improved CUDA kernel

# Requirements and Restrictions
## Critical Requirements:
    1. Functionality must match the reference implementation exactly. Failure to do so will result in a score of 0.
    2. Code must compile and run properly on modern NVIDIA GPUs
## Key Restrictions:
    1. Do not cache or reuse previous results — the code must execute fully on each run.
    2. Keep hyperparameters unchanged (e.g., batch size, dimensions, etc.) as specified in the reference.
    """

    return prompt


class ContrastiveRLDataset(RLHFDataset):
    def _build_messages(self, example: dict):
        level, problem = example["extra_info"]["level"], example["extra_info"]["problem"]
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, "local")

        kernel_src1, speedup1, kernel_src2, speedup2 = sample_from_database(level, problem)
        message = prompt_contrastive_rl(ref_arch_src, kernel_src1, speedup1, kernel_src2, speedup2)

        return [
            {
                "role": "user",
                "content": message
            }
        ]

def reward_from_exec_result(level, problem, exec_result):
    if exec_result.correctness:
        baseline_results = fetch_baseline_results(level, problem, HARDWARE)
        speedup = baseline_results["mean"] / exec_result.runtime
        return 0.3 + float(speedup)
    else:
        return 0.0


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    work_args_list = []
    job_list = []
    thread_ids = {}
    rewards = {}
    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        level = extra_info['level']
        problem = extra_info['problem']
        key = f"{level}_{problem}"
        if key not in thread_ids:
            thread_ids[key] = find_highest_sample_id(run_dir, level, problem, 0, NUM_GENERATIONS)
        else:
            thread_ids[key] += 1
        sample_id = thread_ids[key]
        work_args_list.append((level, problem, sample_id))

        response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
        with open(response_path, "w") as f:
            f.write(solution_str)

        kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"
        kernel_src = extract_last_code(solution_str, ["python", "cpp"])

        if kernel_src is not None:
            write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src) # for debugging

            # Level 1 kernels should not use torch functions
            if level == 1 and torch_function_used(kernel_src):
                rewards[f"{level}_{problem}_{sample_id}"] = (0.0, "Torch function used")
                exec_result = KernelExecResult(correctness=False, compiled=False, metadata={"other_error": "Torch function used"})
                write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
                continue

            # CUDA kernels must be used
            if not is_generated_kernel_used(kernel_src):
                rewards[f"{level}_{problem}_{sample_id}"] = (0.0, "Kernel not used")
                exec_result = KernelExecResult(correctness=False, compiled=False, metadata={"other_error": "Kernel not used"})
                write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
                continue

            work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda"))
            job_list.append({
                "work_args": serialize_work_args(work_args),
                "run_name": RUN_NAME,
                "kernel_src": kernel_src,
                "kernel_name": kernel_name
            })
        else:
            rewards[f"{level}_{problem}_{sample_id}"] = (0.0, "No kernel")
    
    results = send_batch_evaluation_request(EVAL_SERVER_HOST, EVAL_SERVER_PORT, job_list)

    for result, job in zip(results, job_list):
        level = job['work_args']["level"]
        problem = job['work_args']["problem_id"]
        sample_id = job['work_args']["sample_id"]
        write_eval_result_to_separate_file(level, problem, sample_id, result, run_dir)
        reward = reward_from_exec_result(level, problem, result)
        rewards[f"{level}_{problem}_{sample_id}"] = (reward, "Kernel evaluated")
    
    log_reward_info(rewards, len(rewards) != 64)
    add_to_database(rewards)
    
    # Turn rewards into list in the same order as input
    rewards_list = [rewards[f"{level}_{problem}_{sample_id}"][0] for level, problem, sample_id in work_args_list]
    return rewards_list

def log_reward_info(rewards, eval=False):
    """
    Given dictionary of rewards, log different metrics about the rewards.
    - percent of correct samples (per problem)
    - percent of correct samples with speedup
    - avg speedup
    - 
    """
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)
    
    batch_num = 1
    while os.path.exists(os.path.join(run_dir, f"batch_{batch_num}_rewards.json")):
        batch_num += 1

    if eval:
        batch_num -= 1
    
    save_path = os.path.join(run_dir, f"eval_batch_{batch_num}_rewards.json" if eval else f"batch_{batch_num}_rewards.json")

    # Calculate metrics
    metrics = {}
    for key, value in rewards.items():
        reward, reason = value
        level, problem, sample_id = key.split("_")
        prob_key = f"{level}_{problem}"
        if prob_key not in metrics:
            metrics[prob_key] = {"correct": 0, "correct_with_speedup": 0, "num_samples": 0, "speedups": [], "no_kernel": 0, "torch_function_used": 0, "rewards": []}
        metrics[prob_key]["num_samples"] += 1
        metrics[prob_key]["rewards"].append(reward)
        if reward >= 0.3:
            metrics[prob_key]["correct"] += 1
            metrics[prob_key]["speedups"].append(reward - 0.3)
            if reward >= 1.3:
                metrics[prob_key]["correct_with_speedup"] += 1
        elif reason == "No kernel":
            metrics[prob_key]["no_kernel"] += 1
        elif reason == "Torch function used":
            metrics[prob_key]["torch_function_used"] += 1
        
    for prob_key in metrics:
        # assert metrics[prob_key]["num_samples"] == NUM_GENERATIONS, f"Number of samples for {prob_key} is {metrics[prob_key]['num_samples']} but should be {NUM_GENERATIONS}"
        metrics[prob_key]["percent_correct"] = metrics[prob_key]["correct"] / metrics[prob_key]["num_samples"]
        metrics[prob_key]["percent_correct_with_speedup"] = metrics[prob_key]["correct_with_speedup"] / metrics[prob_key]["num_samples"]
        if len(metrics[prob_key]["speedups"]) > 0:
            metrics[prob_key]["avg_speedup"] = sum(metrics[prob_key]["speedups"]) / len(metrics[prob_key]["speedups"])
        else:
            metrics[prob_key]["avg_speedup"] = 0.0
        metrics[prob_key]["percent_no_kernel"] = metrics[prob_key]["no_kernel"] / metrics[prob_key]["num_samples"]
        metrics[prob_key]["percent_torch_function_used"] = metrics[prob_key]["torch_function_used"] / metrics[prob_key]["num_samples"]
    

    agg_metrics = {}
    agg_metrics["percent_correct"] = sum([metrics[prob_key]["correct"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    agg_metrics["percent_correct_with_speedup"] = sum([metrics[prob_key]["correct_with_speedup"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    # Fix avg_speedup calculation: flatten all speedups and compute mean, handle empty case
    all_speedups = [speedup for prob_key in metrics for speedup in metrics[prob_key]["speedups"]]
    agg_metrics["avg_speedup"] = sum(all_speedups) / len(all_speedups) if len(all_speedups) > 0 else 0.0
    agg_metrics["percent_no_kernel"] = sum([metrics[prob_key]["no_kernel"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    agg_metrics["percent_torch_function_used"] = sum([metrics[prob_key]["torch_function_used"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    all_rewards = [reward for prob_key in metrics for reward in metrics[prob_key]["rewards"]]
    agg_metrics["avg_reward"] = sum(all_rewards) / len(all_rewards) if len(all_rewards) > 0 else 0.0

    # Save to file
    with open(save_path, "w") as f:
        json.dump({'by_problem': metrics, 'aggregated': agg_metrics}, f, indent=2)
    
    # Log to wandb
    prefix = "reward_info_eval" if eval else "reward_info"
    wandb.log({
        f"{prefix}/percent_correct": agg_metrics["percent_correct"],
        f"{prefix}/percent_correct_with_speedup": agg_metrics["percent_correct_with_speedup"],
        f"{prefix}/avg_speedup": agg_metrics["avg_speedup"],
        f"{prefix}/percent_no_kernel": agg_metrics["percent_no_kernel"],
        f"{prefix}/percent_torch_function_used": agg_metrics["percent_torch_function_used"],
        f"{prefix}/avg_reward": agg_metrics["avg_reward"]
    }, step=batch_num)
    
