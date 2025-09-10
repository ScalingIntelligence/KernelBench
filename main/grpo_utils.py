import os
import json
import wandb
from math import prod

from src.run_utils import fetch_baseline_results
from src.reward_hacking import is_generated_kernel_used, torch_function_used


HARDWARE = "A6000_babel"

# Reward utils
def get_speedup_from_exec_result(level, problem, exec_result):
    if not exec_result.correctness:
        return 0.0
    baseline_results = fetch_baseline_results(level, problem, HARDWARE)
    speedup = baseline_results["mean"] / exec_result.runtime
    return speedup

def kevin_reward_from_exec_result(level, problem, exec_result):
    if exec_result.correctness:
        return 0.3 + get_speedup_from_exec_result(level, problem, exec_result)
    else:
        return 0.0

def binary_reward_from_exec_result(level, problem, exec_result): 
    if exec_result.correctness:
        return 1.0
    else:
        return 0.0


def is_format_satisfied(kernel_src, level):
    # Level 1 kernels should not use torch functions
    if level == 1 and torch_function_used(kernel_src):
        return False, "Torch function used"

    # CUDA kernels must be used
    if not is_generated_kernel_used(kernel_src):
        return False, "Kernel not used"

    return True, "Format satisfied"




# Logging utils
def log_reward_info(rewards, run_dir, eval=False):
    """
    Given dictionary of rewards, log different metrics about the rewards.
    - percent of correct samples (per problem)
    - percent of correct samples with speedup
    - avg speedup
    - 
    """ 
    batch_num = 1
    while os.path.exists(os.path.join(run_dir, f"batch_{batch_num}_rewards.json")):
        batch_num += 1

    if eval:
        batch_num -= 1
    
    save_path = os.path.join(run_dir, f"eval_batch_{batch_num}_rewards.json" if eval else f"batch_{batch_num}_rewards.json")

    # Calculate metrics
    metrics = {}
    for key, value in rewards.items():
        level, problem, sample_id = key.split("_")
        prob_key = f"{level}_{problem}"
        reward_info, exec_result = value

        if prob_key not in metrics:
            metrics[prob_key] = {
                "num_samples": 0, 
                "rewards": [],
                "reward_info": {
                    "reward": []
                },
                "correct": 0, 
                "correct_with_speedup": 0, 
                "speedups": [], 
                "format_error": {
                    "total": 0,
                },
                "correctness_issue": {
                    "total": 0,
                }
            }

        metrics[prob_key]["num_samples"] += 1
        metrics[prob_key]["rewards"].append(reward_info["reward"])
        for reward_key in reward_info:
            if reward_key not in metrics[prob_key]["reward_info"]:
                metrics[prob_key]["reward_info"][reward_key] = []
            metrics[prob_key]["reward_info"][reward_key].append(reward_info[reward_key])

        if exec_result.correctness:
            metrics[prob_key]["correct"] += 1
            speedup = reward_info["speedup"]
            metrics[prob_key]["speedups"].append(speedup)
            if speedup >= 1.0:
                metrics[prob_key]["correct_with_speedup"] += 1
        elif "format_error" in exec_result.metadata:
            metrics[prob_key]["format_error"] += 1
            error_type = exec_result.metadata["format_error"]
            if error_type not in metrics[prob_key]["format_error"]:
                metrics[prob_key]["format_error"][error_type] = 0
            metrics[prob_key]["format_error"][error_type] += 1
        else: # incorrect kernel
            metrics[prob_key]["correctness_issue"]["total"] += 1
            if "correctness_issue" in exec_result.metadata:
                error_type = exec_result.metadata["correctness_issue"]
                if error_type not in metrics[prob_key]["correctness_issue"]:
                    metrics[prob_key]["correctness_issue"][error_type] = 0
                metrics[prob_key]["correctness_issue"][error_type] += 1

    for prob_key in metrics:
        metrics[prob_key]["percent_correct"] = metrics[prob_key]["correct"] / metrics[prob_key]["num_samples"]
        metrics[prob_key]["percent_correct_with_speedup"] = metrics[prob_key]["correct_with_speedup"] / metrics[prob_key]["num_samples"]
        metrics[prob_key]["percent_format_error"] = metrics[prob_key]["format_error"]["total"] / metrics[prob_key]["num_samples"]
        metrics[prob_key]["percent_correctness_issue"] = metrics[prob_key]["correctness_issue"]["total"] / metrics[prob_key]["num_samples"]

        # Calculate geometric mean of speedups
        safe_speedups = [s for s in metrics[prob_key]["speedups"] if s > 0]
        if len(safe_speedups) > 0:
            metrics[prob_key]["mean_speedup"] = prod(safe_speedups) ** (1 / len(safe_speedups))
        else:
            metrics[prob_key]["mean_speedup"] = 0.0
    

    agg_metrics = {}
    # Reward calculations
    all_rewards = [reward for prob_key in metrics for reward in metrics[prob_key]["rewards"]]
    agg_metrics["avg_reward"] = sum(all_rewards) / len(all_rewards) if len(all_rewards) > 0 else 0.0
    all_exec_rewards = []
    all_autorule_rewards = []
    for prob_key in metrics:
        if "exec_reward" in metrics[prob_key]["reward_info"]:
            all_exec_rewards.extend(metrics[prob_key]["reward_info"]["exec_reward"])
        if "autorule_reward" in metrics[prob_key]["reward_info"]:
            all_autorule_rewards.extend(metrics[prob_key]["reward_info"]["autorule_reward"])
    agg_metrics["avg_exec_reward"] = sum(all_exec_rewards) / len(all_exec_rewards) if len(all_exec_rewards) > 0 else 0.0
    agg_metrics["avg_autorule_reward"] = sum(all_autorule_rewards) / len(all_autorule_rewards) if len(all_autorule_rewards) > 0 else 0.0

    agg_metrics["percent_correct"] = sum([metrics[prob_key]["correct"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    agg_metrics["percent_correct_with_speedup"] = sum([metrics[prob_key]["correct_with_speedup"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    agg_metrics["percent_format_error"] = sum([metrics[prob_key]["format_error"]["total"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    agg_metrics["percent_correctness_issue"] = sum([metrics[prob_key]["correctness_issue"]["total"] for prob_key in metrics]) / sum([metrics[prob_key]["num_samples"] for prob_key in metrics])
    all_speedups = [speedup for prob_key in metrics for speedup in metrics[prob_key]["speedups"]]
    safe_speedups = [s for s in all_speedups if s > 0]
    agg_metrics["mean_speedup"] = prod(safe_speedups) ** (1 / len(safe_speedups)) if len(safe_speedups) > 0 else 0.0

    # Save to file
    with open(save_path, "w") as f:
        json.dump({'aggregated': agg_metrics, 'by_problem': metrics}, f, indent=2)
    
    # Log to wandb
    prefix = "reward_info_eval" if eval else "reward_info"
    wandb.log({
        f"{prefix}/avg_reward": agg_metrics["avg_reward"],
        f"{prefix}/avg_exec_reward": agg_metrics["avg_exec_reward"],
        f"{prefix}/avg_autorule_reward": agg_metrics["avg_autorule_reward"],
        f"{prefix}/percent_correct": agg_metrics["percent_correct"],
        f"{prefix}/percent_correct_with_speedup": agg_metrics["percent_correct_with_speedup"],
        f"{prefix}/mean_speedup": agg_metrics["mean_speedup"],
        f"{prefix}/percent_format_error": agg_metrics["percent_format_error"],
        f"{prefix}/percent_correctness_issue": agg_metrics["percent_correctness_issue"],
    }, step=batch_num)
    
