import yaml
from argparse import ArgumentParser
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
    total = 0
    compiled = 0
    correct = 0
    runtime_error = 0
    output_mismatch = 0
    output_shape_mismatch = 0

    for _, res in eval_results.items():
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
    gmsr_correct_and_faster = geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n)

    # list of speedup thresholds p
    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = {p: fastp(is_correct, baseline_speed, actual_speed, n, p) for p in p_values}

    return {
        "mean_speedup_correct": gmsr_correct, # geometric mean of speedup for correct samples
        "mean_speedup_correct_and_faster": gmsr_correct_and_faster, # geometric mean of speedup for correct and faster samples
        "fast_p_results": results
    }

def compute_efficiency_metrics_all_baselines(config: TestTimeScalingConfig, hardware: str, eval_results: dict) -> dict:
    results = {}
    for baseline in BASELINES:
        try:
            baseline_file_path = f'results/timing/{hardware}/baseline_time_{baseline}.json'
            assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

            with open(baseline_file_path, 'r') as f:
                baseline_results = json.load(f)

            baseline_results = baseline_results[f'level{config["level"]}']

            comp_metrics = compute_efficiency_metrics(eval_results, baseline_results)
            results[baseline] = comp_metrics
        except Exception as e:
            print(f"Error computing efficiency metrics for {baseline}: {e}")
            continue

    return results

# best, average, individual (per sample)


def hardware_check(eval_results: dict, hardware_ref: str):
    hardware = list(list(eval_results.values())[0].values())[0]["metadata"]["hardware"]
    for _, prob_res in eval_results.items():
        for _, sample_res in prob_res.items():
            assert sample_res["metadata"]["hardware"] == hardware, f"Hardware mismatch: {sample_res['metadata']['hardware']} != {hardware}"
    print(f"Computing metrics for {hardware} with baseline {hardware_ref} (Should match)")


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


def compute_all_metrics(config, hardware, eval_results):
    """
    Expects eval_results to be dict of problem_id -> exec_result
    """
    correctness_metrics = compute_correctness_metrics(eval_results)
    dataset = construct_kernelbench_dataset(config["level"])
    eval_results = patch(eval_results, dataset)
    efficiency_metrics = compute_efficiency_metrics_all_baselines(config, hardware, eval_results)
    return {"correctness": correctness_metrics, "speedups": efficiency_metrics}


def compute_metrics_base(config: TestTimeScalingConfig, hardware: str, eval_results: dict) -> dict:
    eval_results = {k: v["0"] for k, v in eval_results.items()}
    return compute_all_metrics(config, hardware, eval_results)


dummy_result = {
    "sample_id": 0, 
    "compiled": False, 
    "correctness": False, 
    "metadata": {},
    "runtime": -1.0, 
    "runtime_stats": {}
}


def increasing_best_solution_metrics(config: TestTimeScalingConfig, hardware: str, eval_results: dict, num_steps) -> dict:
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
            elif prev_best["correctness"] and res["correctness"] and res["runtime"] < prev_best["runtime"]:
                best_by_step[step][pid] = res
            else:
                best_by_step[step][pid] = prev_best
            
    metrics = {}
    for step, step_results in best_by_step.items():
        metrics[step] = compute_all_metrics(config, hardware, step_results)
    return metrics


def compute_metrics_best_of_n(config: TestTimeScalingConfig, hardware: str, eval_results: dict) -> dict:
    by_sample_results = {}
    for pid, prob_res in eval_results.items():
        for sid, sample_res in prob_res.items():
            if sid not in by_sample_results:
                by_sample_results[sid] = {}
            
            by_sample_results[sid][pid] = sample_res
    
    metrics = {"by_sample": {}}
    for sid, sample_res in by_sample_results.items():
        metrics["by_sample"][sid] = compute_all_metrics(config, hardware, by_sample_results[sid])
    metrics["best_by_sample"] = increasing_best_solution_metrics(config, hardware, eval_results, config["num_parallel"])
    return metrics


def compute_metrics_iterative_refinement(config: TestTimeScalingConfig, hardware: str, eval_results: dict) -> dict:
    assert config["num_parallel"] == 1, "Iterative refinement is only supported for 1 parallel run"
    return increasing_best_solution_metrics(config, hardware, eval_results, config["num_iterations"])


def compute_metrics_metr(config: TestTimeScalingConfig, hardware: str, eval_results: dict) -> dict:
    return increasing_best_solution_metrics(config, hardware, eval_results, config["num_samples"])


def compute_metrics_stanford(config: TestTimeScalingConfig, hardware: str, eval_results: dict) -> dict:
    pass


def compute_metrics(config: TestTimeScalingConfig, hardware: str, eval_file_path: str, run_dir: str) -> dict:
    with open(eval_file_path, 'r') as f:
        eval_results = json.load(f)

    print("Checking that results are on the same hardware")
    hardware_check(eval_results, hardware)

    match config["method"]:
        case "base":
            metrics = compute_metrics_base(config, hardware, eval_results)
        case "best-of-N":
            metrics = compute_metrics_best_of_n(config, hardware, eval_results)
        case "iterative refinement":
            metrics = compute_metrics_iterative_refinement(config, hardware, eval_results)
        case "METR":
            metrics = compute_metrics_metr(config, hardware, eval_results)
        case "Stanford":
            metrics = compute_metrics_stanford(config, hardware, eval_results)
        case _:
            raise ValueError(f"Invalid method: {config['method']}")

    print("Computed all metrics")
    print(metrics)

    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_file}")
    
    return metrics


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--run_dir", type=str, required=True)
    argparser.add_argument("--hardware", type=str, required=True)
    args = argparser.parse_args()

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    eval_file_path = os.path.join(args.run_dir, "eval_results.json")

    compute_metrics(config, args.hardware, eval_file_path, args.run_dir)


if __name__ == "__main__":
    main()