import pydra
import os
import json
import numpy as np
from tabulate import tabulate

from src.score import *
from src.utils import read_file
from src.dataset import construct_kernelbench_dataset
from configs import TestTimeScalingConfig


BASELINES = ["torch", "torch_compile_inductor_default", "torch_compile_inductor_reduce-overhead", "torch_compile_inductor_max-autotune", "torch_compile_inductor_max-autotune-no-cudagraphs"]


def compute_correctness_metrics(eval_results):
    total_samples = len(eval_results)
    compiled = 0
    correct = 0

    for _, res in eval_results.items():
        if res["compiled"] == True:
            compiled += 1
        if res["correctness"] == True:
            correct += 1

    return {
        "total_samples": total_samples,
        "compiled": compiled,
        "correct": correct,
        "compilation_rate": compiled/total_samples,
        "correctness_rate": correct/total_samples,
    }


def compute_efficiency_metrics(eval_results, baseline_results):
    """
    expects eval_results and baseline_results to be dict (problem_id -> exec_result)
    """
    is_correct = np.array([entry["correctness"] for entry in eval_results.values()])
    baseline_speed = np.array([entry["mean"] for entry in baseline_results.values()])
    actual_speed = np.array([entry["runtime"] for entry in eval_results.values()])
    n = len(is_correct)

    assert len(baseline_speed) == n, "Baseline speedup values do not match the number of eval results"
    assert len(actual_speed) == n, "Actual speedup values do not match the number of eval results"

    # Calculate the metrics
    gmsr_correct = geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n)

    # list of speedup thresholds p
    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = {p: fastp(is_correct, baseline_speed, actual_speed, n, p) for p in p_values}

    return {
        "mean_speedup_correct": gmsr_correct, # geometric mean of speedup for correct samples
        "fast_p_results": results
    }

def compute_efficiency_metrics_all_baselines(config: TestTimeScalingConfig, eval_results: dict) -> dict:
    results = {}
    for baseline in BASELINES:
        try:
            baseline_file_path = f'results/timing/{config.hardware}/baseline_time_{baseline}.json'
            assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

            with open(baseline_file_path, 'r') as f:
                baseline_results = json.load(f)

            baseline_results = baseline_results[f'level{config.level}']

            comp_metrics = compute_efficiency_metrics(eval_results, baseline_results)
            results[baseline] = comp_metrics
        except Exception as e:
            print(f"Error computing efficiency metrics for {baseline}: {e}")
            continue

    return results


def hardware_check(eval_results: dict, config: TestTimeScalingConfig):
    hardware = list(list(eval_results.values())[0].values())[0]["metadata"]["hardware"]
    for _, prob_res in eval_results.items():
        for _, sample_res in prob_res.items():
            assert sample_res["metadata"]["hardware"] == hardware, f"Hardware mismatch: {sample_res['metadata']['hardware']} != {hardware}"
    print(f"Computing metrics for {hardware} with baseline {config.hardware} (Should match)")


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


def compute_metrics_base(config: TestTimeScalingConfig, eval_results: dict) -> dict:
    eval_results = {k: v["0"] for k, v in eval_results.items()}
    correctness_metrics = compute_correctness_metrics(eval_results)
    dataset = construct_kernelbench_dataset(config.level)
    eval_results = patch(eval_results, dataset)
    efficiency_metrics = compute_efficiency_metrics_all_baselines(config, eval_results)
    return {**correctness_metrics, "speedups": efficiency_metrics}

def compute_metrics_best_of_n(config: TestTimeScalingConfig, eval_results: dict) -> dict:
    pass

def compute_metrics_iterative_refinement(config: TestTimeScalingConfig, eval_results: dict) -> dict:
    pass

def compute_metrics_metr(config: TestTimeScalingConfig, eval_results: dict) -> dict:
    pass

def compute_metrics_stanford(config: TestTimeScalingConfig, eval_results: dict) -> dict:
    pass

def compute_metrics(config: TestTimeScalingConfig, eval_file_path: str, run_dir: str) -> dict:
    with open(eval_file_path, 'r') as f:
        eval_results = json.load(f)

    hardware_check(eval_results, config)

    match config.method:
        case "base":
            metrics = compute_metrics_base(config, eval_results)
        case "best-of-N":
            metrics = compute_metrics_best_of_n(config, eval_results)
        case "iterative refinement":
            metrics = compute_metrics_iterative_refinement(config, eval_results)
        case "METR":
            metrics = compute_metrics_metr(config, eval_results)
        case "Stanford":
            metrics = compute_metrics_stanford(config, eval_results)
        case _:
            raise ValueError(f"Invalid method: {config.method}")


    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics


@pydra.main(base=TestTimeScalingConfig)
def main(config: TestTimeScalingConfig):
    run_dir = os.path.join(config.runs_dir, config.run_name)
    eval_file_path = os.path.join(run_dir, "eval_results.json")

    compute_metrics(config, eval_file_path, run_dir)


if __name__ == "__main__":
    main()