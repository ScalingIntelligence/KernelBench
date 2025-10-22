import json, os

import pydra
from pydra import Config, REQUIRED
from src.dataset import construct_kernelbench_dataset
from tabulate import tabulate

"""
Benchmark Eval Analysis

This script shows how to conduct analysis for model performance on KernelBench

Given generations and eval results, this script will compute the following:
- Success rate (compiled and correctness)
- Geometric mean of speedup for correct samples
- Fast_p score for different speedup thresholds (we recommend and use this metric)

Usage:
```
python3 scripts/benchmark_eval_analysis.py run_name=<run_name> level=<level> hardware=<hardware> baseline=<baseline>
```
hardware + baseline should correspond to the results/timing/hardware/baseline.json file   

"""


class AnalysisConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED  # name of the run to evaluate
        self.level = REQUIRED  # level to evaluate

        self.hardware = REQUIRED  # hardware to evaluate
        self.baseline = REQUIRED  # baseline to compare against

    def __repr__(self):
        return f"AnalysisConfig({self.to_dict()})"


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
                "runtime_stats": {},
            }

    return eval_results


def analyze_greedy_eval(run_name, hardware, baseline, level):
    """
    Analyze the greedy eval results for a run of a particular level
    """

    dataset = construct_kernelbench_dataset(level)

    # load json
    eval_file_path = f"runs/{run_name}/eval_results.json"
    assert os.path.exists(
        eval_file_path
    ), f"Eval file does not exist at {eval_file_path}"

    # Check if pass@k results exist
    pass_at_k_file_path = f"runs/{run_name}/pass_at_k_results.json"
    has_pass_at_k_results = os.path.exists(pass_at_k_file_path)

    baseline_file_path = f"results/timing/{hardware}/{baseline}.json"
    assert os.path.exists(
        baseline_file_path
    ), f"Baseline file does not exist at {baseline_file_path}"

    with open(eval_file_path, "r") as f:
        eval_results = json.load(f)

    # Load pass@k results if available
    pass_at_k_results = None
    if has_pass_at_k_results:
        with open(pass_at_k_file_path, "r") as f:
            pass_at_k_results = json.load(f)

    with open(baseline_file_path, "r") as f:
        baseline_results = json.load(f)

    # Initialize counters
    total_count = len(dataset)
    total_eval = len(eval_results)
    compiled_count = 0
    correct_count = 0

    # todo: for now we only consider sample_id = 0 though we should change this later

    stripped_eval_results = {}
    for key, result in eval_results.items():
        entry = [r for r in result if r["sample_id"] == 0]
        assert len(entry) <= 1, "Multiple entries for sample_id = 0"
        if len(entry) == 1:
            stripped_eval_results[key] = entry[0]
    eval_results = stripped_eval_results

    # Patch the eval results
    eval_results = patch(eval_results, dataset)

    # Count results
    for entry in eval_results.values():
        if entry["compiled"] == True:
            compiled_count += 1
        if entry["correctness"] == True:
            correct_count += 1

    # Print results
    print("-" * 128)
    print(f"Eval Summary for {run_name}")
    print("-" * 128)
    print(f"Total test cases with Eval Results: {total_eval} out of {total_count}")
    print(f"Successfully compiled: {compiled_count}")
    print(f"Functionally correct: {correct_count}")

    print(f"\nSuccess rates:")
    print(f"Compilation rate: {compiled_count/total_count*100:.1f}%")
    print(f"Correctness rate: {correct_count/total_count*100:.1f}%")

    import numpy as np

    # Calculate speedup metrics
    from src.score import (
        fastp,
        geometric_mean_speed_ratio_correct_and_faster_only,
        geometric_mean_speed_ratio_correct_only,
    )

    # Extract the speedup values
    is_correct = np.array([entry["correctness"] for entry in eval_results.values()])
    baseline_speed = np.array(
        [entry["mean"] for entry in baseline_results[f"level{level}"].values()]
    )
    actual_speed = np.array([entry["runtime"] for entry in eval_results.values()])
    n = len(is_correct)

    assert (
        len(baseline_speed) == n
    ), "Baseline speedup values do not match the number of eval results"
    assert (
        len(actual_speed) == n
    ), "Actual speedup values do not match the number of eval results"

    # Calculate the metrics
    gmsr_correct = geometric_mean_speed_ratio_correct_only(
        is_correct, baseline_speed, actual_speed, n
    )

    # list of speedup thresholds p
    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = [
        [p, fastp(is_correct, baseline_speed, actual_speed, n, p)] for p in p_values
    ]

    # Print the results
    print("\nSpeedup Metrics:")
    print(f"Geometric mean of speedup for correct samples: {gmsr_correct:.4f}")

    # Print table
    print("\nFast_p Results:")
    print(
        tabulate(
            results, headers=["Speedup Threshold (p)", "Fast_p Score"], tablefmt="grid"
        )
    )

    # Display pass@k metrics if available
    if pass_at_k_results:
        print("\nPass@k Correctness Metrics:")

        # Print metadata
        metadata = pass_at_k_results.get("metadata", {})
        if metadata:
            print("\nEvaluation Metadata:")
            metadata_table = [[key, value] for key, value in metadata.items()]
            print(
                tabulate(metadata_table, headers=["Metric", "Value"], tablefmt="grid")
            )

        # Print average pass@k metrics
        averages = pass_at_k_results.get("averages", {})
        if averages:
            print("\nAverage Pass@k Metrics:")
            avg_table = [[k, v] for k, v in averages.items()]
            print(tabulate(avg_table, headers=["Metric", "Value"], tablefmt="grid"))


@pydra.main(base=AnalysisConfig)
def main(config: AnalysisConfig):
    analyze_greedy_eval(config.run_name, config.hardware, config.baseline, config.level)


if __name__ == "__main__":
    main()
