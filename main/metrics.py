import yaml
from argparse import ArgumentParser
import os
import json
import numpy as np
from tabulate import tabulate

from dataset import construct_kernelbench_dataset, TEST_PROBLEM_IDS_LEVEL_1, TEST_PROBLEM_IDS_LEVEL_2, TRAIN_PROBLEM_IDS_LEVEL_1, TRAIN_PROBLEM_IDS_LEVEL_2
from src.score import *


BASELINES = ["torch"] # , "torch_compile_inductor_default", "torch_compile_inductor_reduce-overhead", "torch_compile_inductor_max-autotune", "torch_compile_inductor_max-autotune-no-cudagraphs"]


def compute_correctness_metrics(eval_results, subset=None):
    """
    Expects eval_results to be dict of problem_id -> exec_result
    """
    total = 0
    compiled = 0
    correct = 0
    runtime_error = 0
    output_mismatch = 0
    output_shape_mismatch = 0

    for k, res in eval_results.items():
        if subset is not None and int(k) not in subset:
            continue
        total += 1
        if res["compiled"]:
            compiled += 1
        if res["correctness"]:
            correct += 1
        if "runtime_error" in res["metadata"]:
            runtime_error += 1
        if "correctness_issue" in res["metadata"]:
            if res["metadata"]["correctness_issue"] == "Output mismatch":
                output_mismatch += 1
            elif "Output shape mismatch" in res["metadata"]["correctness_issue"]:
                output_shape_mismatch += 1

    return {
        "total": total,
        "compiled": compiled,
        "correct": correct,
        "runtime_error": runtime_error,
        "output_mismatch": output_mismatch,
        "output_shape_mismatch": output_shape_mismatch
    }


def compute_efficiency_metrics(eval_results, baseline_results, subset=None):
    """
    Expects eval_results and baseline_results to be dict (problem_id -> exec_result)
    """
    # Filter out problems where baseline_results is empty
    eval = []
    baseline = []
    for k, v in baseline_results.items():
        if v is None: 
            # print(f"Skipping {k} in baseline_results")
            continue
        problem_number = k.split("_")[0]
        if subset is not None and int(problem_number) not in subset:
            continue
        if problem_number not in eval_results: 
            # print(f"Problem {problem_number} not in eval_results")
            continue
        eval.append(eval_results[problem_number])
        baseline.append(v)

    is_correct = np.array([entry["correctness"] for entry in eval])
    baseline_speed = np.array([entry["mean"] for entry in baseline])
    actual_speed = np.array([entry["runtime"] for entry in eval])
    n = len(is_correct)

    assert len(baseline_speed) == n, "Baseline speedup values do not match the number of eval results"
    assert len(actual_speed) == n, "Actual speedup values do not match the number of eval results"

    # Calculate the metrics
    gmsr_correct = geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n)
    gmsr_correct_and_faster = geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n)

    # list of speedup thresholds p
    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = {p: fastp(is_correct, baseline_speed, actual_speed, n, p) for p in p_values}

    return {
        "mean_speedup_correct": gmsr_correct, # geometric mean of speedup for correct samples
        "mean_speedup_correct_and_faster": gmsr_correct_and_faster, # geometric mean of speedup for correct and faster samples
        "fast_p_results": results
    }

def compute_efficiency_metrics_all_baselines(config, hardware: str, eval_results: dict, subset=None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> exec_result
    """
    results = {}
    for baseline in BASELINES:
        try:
            baseline_file_path = f'results/timing/{hardware}/baseline_time_{baseline}.json'
            assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

            with open(baseline_file_path, 'r') as f:
                baseline_results = json.load(f)

            baseline_results = baseline_results[f'level{config["level"]}']

            comp_metrics = compute_efficiency_metrics(eval_results, baseline_results, subset)
            results[baseline] = comp_metrics
        except Exception as e:
            print(f"Error computing efficiency metrics for {baseline}: {e}")
            continue

    return results


def hardware_check(eval_results: dict, hardware_ref: str):
    print("Checking that results are on the same hardware")
    hardware = list(list(eval_results.values())[0].values())[0]["metadata"]["hardware"]
    for _, prob_res in eval_results.items():
        for _, sample_res in prob_res.items():
            if "metadata" not in sample_res or "hardware" not in sample_res["metadata"]:
                continue
            assert sample_res["metadata"]["hardware"] == hardware, f"Hardware mismatch: {sample_res['metadata']['hardware']} != {hardware}"
    print(f"Computing metrics for {hardware} with baseline {hardware_ref} (Should match)")


dummy_result = {
    "sample_id": 0, 
    "compiled": False, 
    "correctness": False, 
    "metadata": {},
    "runtime": -1.0, 
    "runtime_stats": {}
}

def patch(eval_results, dataset):
    """
    Patch the eval results with the dataset
    """
    for pid in range(1, len(dataset) + 1):
        if str(pid) not in eval_results:
            eval_results[str(pid)] = {
                "sample_id": 0, 
                "compiled": False, 
                "correctness": False, 
                "metadata": {},
                "runtime": -1.0, 
                "runtime_stats": {}
            }
    return eval_results


def compute_all_metrics(config, hardware, eval_results, subset=None):
    """
    Computes correctness and efficiency metrics for evaluation results of form dict of problem_id -> exec_result
    """
    correctness_metrics = compute_correctness_metrics(eval_results, subset)
    dataset = construct_kernelbench_dataset(config["level"])
    eval_results = patch(eval_results, dataset)
    efficiency_metrics = compute_efficiency_metrics_all_baselines(config, hardware, eval_results, subset)
    return {"correctness": correctness_metrics, "speedups": efficiency_metrics}


def compute_metrics_base(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    eval_results = {k: v["0"] for k, v in eval_results.items()}
    return compute_all_metrics(config, hardware, eval_results, subset)


def increasing_best_solution_metrics(config, hardware: str, eval_results: dict, num_steps, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    best_by_step = {}
    best_by_step[0] = {k: v["0"] if "0" in v else dummy_result for k, v in eval_results.items()}

    for step in range(1, num_steps):
        best_by_step[step] = {}
        for pid, prob_res in eval_results.items():
            prev_best = best_by_step[step - 1][pid]
            if str(step) not in prob_res:
                best_by_step[step][pid] = prev_best
                continue
            res = prob_res[str(step)]
            if not prev_best["correctness"] and res["correctness"]:
                best_by_step[step][pid] = res
            elif not prev_best["compiled"] and res["compiled"]:
                best_by_step[step][pid] = res
            elif prev_best["correctness"] and res["correctness"] and "runtime" in res and res["runtime"] < prev_best["runtime"]:
                best_by_step[step][pid] = res
            else:
                best_by_step[step][pid] = prev_best
            
    metrics = {}
    for step, step_results in best_by_step.items():
        metrics[step] = compute_all_metrics(config, hardware, step_results, subset)
    return metrics


def compute_metrics_best_of_n(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    by_sample_results = {}
    for pid, prob_res in eval_results.items():
        for sid, sample_res in prob_res.items():
            if sid not in by_sample_results:
                by_sample_results[sid] = {}
            
            by_sample_results[sid][pid] = sample_res
    
    metrics = {"by_sample": {}}
    for sid, sample_res in by_sample_results.items():
        metrics["by_sample"][sid] = compute_all_metrics(config, hardware, by_sample_results[sid], subset)
    metrics["best_by_sample"] = increasing_best_solution_metrics(config, hardware, eval_results, config["num_parallel"])
    return metrics


def compute_metrics_iterative_refinement(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    assert config["num_parallel"] == 1, "Iterative refinement is only supported for 1 parallel run"
    return increasing_best_solution_metrics(config, hardware, eval_results, config["num_iterations"], subset)


def compute_metrics_metr(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    for i in range(config["num_samples"]-1):
        for pid, prob_res in eval_results.items():
            if str(i+1) in prob_res:
                eval_results[pid][str(i)] = eval_results[pid][str(i+1)]
            else:
                eval_results[pid][str(i)] = dummy_result
    return increasing_best_solution_metrics(config, hardware, eval_results, config["num_samples"], subset)


def compute_metrics_stanford(config, hardware: str, eval_results: dict, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    pass


def compute_metrics_test_time_scaling(config, hardware: str, eval_results: dict, run_dir: str, subset: list[int] = None) -> dict:
    """
    Expects eval_results to be dict of problem_id -> sample_id -> exec_result
    """
    # hardware_check(eval_results, hardware)

    match config["method"]:
        case "base":
            metrics = compute_metrics_base(config, hardware, eval_results, subset)
        case "best-of-N":
            metrics = compute_metrics_best_of_n(config, hardware, eval_results, subset)
        case "iterative refinement":
            metrics = compute_metrics_iterative_refinement(config, hardware, eval_results, subset)
        case "METR":
            metrics = compute_metrics_metr(config, hardware, eval_results, subset)
        case "Stanford":
            metrics = compute_metrics_stanford(config, hardware, eval_results, subset)
        case _:
            raise ValueError(f"Invalid method: {config['method']}")

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    
    return metrics


def compute_metrics_grpo(config, hardware: str, eval_results: dict, run_dir: str) -> dict:
    # Eval subsets for Level 1 and 2
    eval_results_level_1 = eval_results["1"]
    eval_results_level_2 = eval_results["2"]

    config["num_parallel"] = 64 # TODO: make this match number of kernels generated in training
    metrics_train_level_1 = compute_metrics_best_of_n(config, hardware, eval_results_level_1, subset=TRAIN_PROBLEM_IDS_LEVEL_1)
    metrics_train_level_2 = compute_metrics_best_of_n(config, hardware, eval_results_level_2, subset=TRAIN_PROBLEM_IDS_LEVEL_2)
    metrics_eval_level_1 = compute_all_metrics(config, hardware, eval_results_level_1, subset=TEST_PROBLEM_IDS_LEVEL_1)
    metrics_eval_level_2 = compute_all_metrics(config, hardware, eval_results_level_2, subset=TEST_PROBLEM_IDS_LEVEL_2)

    metrics = {
        "1": {
            "train": metrics_train_level_1,
            "eval": metrics_eval_level_1
        },
        "2": {
            "train": metrics_train_level_2,
            "eval": metrics_eval_level_2
        }
    }

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    
    return metrics


def collate_eval_results(run_dir):
    """
    Go through every file in directory and find those like "level_1_problem_1_sample_0_eval_results.json".
    This file will contain a json object like {"1": {"1": {"0": eval_result}}}.
    Combine all of these into one json and save it to file.
    """
    import glob
    import re

    # Pattern to match eval result files
    pattern = os.path.join(run_dir, "level_*_problem_*_sample_*_eval_result.json")
    eval_files = glob.glob(pattern)
    
    # Combined results structure: {level: {problem: {sample: eval_result}}}
    combined_results = {}
    
    for file_path in eval_files:
        # Extract level, problem, and sample from filename
        filename = os.path.basename(file_path)
        match = re.match(r"level_(\d+)_problem_(\d+)_sample_(\d+)_eval_result\.json", filename)
        
        if match:
            level = match.group(1)
            problem = match.group(2)
            sample = match.group(3)
            
            # Read the eval result file
            with open(file_path, 'r') as f:
                eval_result = json.load(f)
            
            # Initialize nested structure if needed
            if level not in combined_results:
                combined_results[level] = {}
            if problem not in combined_results[level]:
                combined_results[level][problem] = {}
            
            # Add the eval result
            combined_results[level][problem][sample] = eval_result

    # Sort the combined results by level, problem_id, and sample_id
    sorted_results = {}
    for level in sorted(combined_results.keys(), key=int):
        sorted_results[level] = {}
        for problem in sorted(combined_results[level].keys(), key=int):
            sorted_results[level][problem] = {}
            for sample in sorted(combined_results[level][problem].keys(), key=int):
                sorted_results[level][problem][sample] = combined_results[level][problem][sample]
    
    combined_results = sorted_results
    
    # Save combined results to file
    output_path = os.path.join(run_dir, "eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print(f"Collated {len(eval_files)} eval result files into {output_path}")
    return combined_results


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--run_dir", type=str, required=True)
    argparser.add_argument("--hardware", type=str, required=True)
    argparser.add_argument("--grpo", action="store_true")
    args = argparser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    eval_file_path = os.path.join(args.run_dir, "eval_results.json")
    if not os.path.exists(eval_file_path):
        print("Collating eval results")
        collate_eval_results(args.run_dir)

    with open(eval_file_path, 'r') as f:
        eval_results = json.load(f)[f'{config["level"]}']

    if args.grpo:
        compute_metrics_grpo(config, args.hardware, eval_results, args.run_dir)
    else:
        compute_metrics_test_time_scaling(config, args.hardware, eval_results[f'{config["level"]}'], args.run_dir, subset=None)


if __name__ == "__main__":
    main()