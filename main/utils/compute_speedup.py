import os
import argparse
import json
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Compute speedup ratio for kernels in a run directory.")
    parser.add_argument("run_dir", type=str, help="Path to the run directory containing kernel files.")
    return parser.parse_args()

def extract_problem_sample(filename):
    # Example: level_1_problem_1_sample_0_kernel.py
    m = re.match(r"level_(\d+)_problem_(\d+)_sample_(\d+)_kernel\.py", filename)
    if m:
        level = int(m.group(1))
        problem = int(m.group(2))
        sample = int(m.group(3))
        return level, problem, sample
    return None

def main():
    args = parse_args()
    run_dir = args.run_dir

    kernel_files = [f for f in os.listdir(run_dir) if f.endswith("_kernel.py")]
    if not kernel_files:
        print("No kernel files found in the run directory.")
        return

    baseline_file = os.path.join("results", "timing", "A6000_babel", "baseline_time_torch.json")
    if not os.path.exists(baseline_file):
        print(f"Baseline file {baseline_file} does not exist.")
        return
    
    with open(baseline_file, "r") as f:
        baseline_data = json.load(f)
    

    eval_results_file = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(eval_results_file):
        print(f"Eval results file {eval_results_file} does not exist.")
        return
    
    with open(eval_results_file, "r") as f:
        eval_results = json.load(f)

    for kernel_file in kernel_files:
        try:
            info = extract_problem_sample(kernel_file)
            if not info:
                continue
            level, problem, sample = info

            level_key = f"level{level}"
            if level_key not in baseline_data:
                print(f"Level {level} not found in baseline data.")
                continue
            baseline_level_data = baseline_data[level_key]
            baseline_runtime = None
            for prob_name, prob_data in baseline_level_data.items():
                prob_id = int(prob_name.split("_")[0])
                if prob_id != problem:
                    continue
                baseline_runtime = prob_data["mean"]
                break
            if baseline_runtime is None:
                print(f"Baseline runtime not found for problem {problem}.")
                continue

            eval_result = eval_results[f"{level}"][f"{problem}"][f"{sample}"]

            if not eval_result["correctness"]:
                # print(f"Kernel {kernel_file} is incorrect, skipping speedup calculation.")
                continue

            speedup_ratio = baseline_runtime / eval_result["runtime"]
            if speedup_ratio > 0:
                print(f"{kernel_file}: Speedup ratio = {speedup_ratio:.4f}")
        except Exception as e:
            print(f"Error processing {kernel_file}: {e}")
            continue

if __name__ == "__main__":
    main()
