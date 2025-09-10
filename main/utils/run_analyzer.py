import os
import argparse
import json
import numpy as np
from src.score import geometric_mean_speed_ratio_correct_only

def analyze_correct_counts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    eval_path = os.path.join(args.run_dir, "eval_results.json")

    with open(eval_path, "r") as f:
        eval_results = json.load(f)

    # eval_results[level][problem_id][sample_id]["correctness"]
    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    # good_problems = []
    for level in eval_results:
        for problem_id in eval_results[level]:
            samples = eval_results[level][problem_id]
            total = len(samples)
            correct = sum(1 for sample_id in samples if samples[sample_id].get("correctness", False))
            correct_counts[correct] += 1
            # if correct >= 5:
                # good_problems.append(int(problem_id))

            # print(f"Level {level} Problem {problem_id}: {correct}/{total} correct")
    
    # Summary: for 0, 1, ..., 8, print how many problems have that many correct samples
    for correct in correct_counts:
        print(f"Correct {correct}/8: {correct_counts[correct]} problems")
    
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    correct_values = list(correct_counts.keys())
    problem_counts = [correct_counts[c] for c in correct_values]

    plt.figure(figsize=(8, 5))
    plt.bar(correct_values, problem_counts, color='skyblue')
    plt.xlabel('Number of Correct Samples (out of 8)')
    plt.ylabel('Number of Problems')
    plt.title('Distribution of Correct Samples')
    plt.xticks(correct_values)
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "correct_counts.png"))

    # print(f"Good problems: {good_problems}")


def analyze_rule_speedup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_run_dir", type=str, required=True)
    parser.add_argument("--target_run_dir", type=str, required=True)
    parser.add_argument("--hardware", default="A6000_babel")
    args = parser.parse_args()

    ref_eval_path = os.path.join(args.ref_run_dir, "eval_results.json")
    target_eval_path = os.path.join(args.target_run_dir, "eval_results.json")

    with open(ref_eval_path, "r") as f:
        ref_eval_results = json.load(f)

    with open(target_eval_path, "r") as f:
        target_eval_results = json.load(f)


    correct_count_ref = 0
    correct_count_target = 0
    correct_in_both = []
    target_runtimes = []
    ref_runtimes = []

    for level in ref_eval_results:
        if level not in target_eval_results:
            continue
        for problem_id in ref_eval_results[level]:
            if problem_id not in target_eval_results[level]:
                continue

            # Find a sample in each run that is correct
            ref_samples = ref_eval_results[level][problem_id]
            target_samples = target_eval_results[level][problem_id]

            # Find the correct sample in ref with the lowest runtime
            ref_correct_sample = None
            min_runtime = float('inf')
            for sample_id, sample in ref_samples.items():
                if sample.get("correctness", False):
                    runtime = sample.get("runtime", None)
                    if runtime is not None and runtime > 0 and runtime < min_runtime:
                        min_runtime = runtime
                        ref_correct_sample = sample
            if ref_correct_sample is not None:
                correct_count_ref += 1

            # Find the correct sample in target with the lowest runtime
            target_correct_sample = None
            min_runtime = float('inf')
            for sample_id, sample in target_samples.items():
                if sample.get("correctness", False):
                    runtime = sample.get("runtime", None)
                    if runtime is not None and runtime > 0 and runtime < min_runtime:
                        min_runtime = runtime
                        target_correct_sample = sample
            if target_correct_sample is not None:
                correct_count_target += 1

            if ref_correct_sample is not None and target_correct_sample is not None:
                # Both have at least one correct sample
                correct_in_both.append((level, problem_id))
                ref_runtime = ref_correct_sample.get("runtime", None)
                target_runtime = target_correct_sample.get("runtime", None)
                target_runtimes.append(target_runtime)
                ref_runtimes.append(ref_runtime)


    print(f"Number of problems correct in ref: {correct_count_ref}")
    print(f"Number of problems correct in target: {correct_count_target}")
    print(f"Number of problems correct in both: {len(correct_in_both)}")

    # Get baseline speed 
    baseline_file_path = f'results/timing/{args.hardware}/baseline_time_torch.json'
    assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

    with open(baseline_file_path, 'r') as f:
        baseline_results = json.load(f)

    baseline_results = baseline_results

    baseline_runtimes = []
    for (level, problem_id) in correct_in_both:
        for prob_name, prob_data in baseline_results[f"level{level}"].items():
            if prob_name.split("_")[0] == str(problem_id):
                baseline_runtimes.append(prob_data.get("mean", None))
                break

    # Median speedup of ref over baseline
    is_correct = np.array([1] * len(ref_runtimes))
    geo_mean = geometric_mean_speed_ratio_correct_only(is_correct, baseline_runtimes, ref_runtimes, len(ref_runtimes))
    print(f"Geometric mean of speedups (ref over baseline): {geo_mean:.3f}")

    # Mean speedup of target over baseline
    is_correct = np.array([1] * len(target_runtimes))
    geo_mean = geometric_mean_speed_ratio_correct_only(is_correct, baseline_runtimes, target_runtimes, len(target_runtimes))
    print(f"Geometric mean of speedups (target over baseline): {geo_mean:.3f}")

   

def main():
    # analyze_correct_counts()
    analyze_rule_speedup()


if __name__ == "__main__":
    main()
