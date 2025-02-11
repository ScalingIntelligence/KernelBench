import json, os

"""
Analyze the greedy eval results for a run of a particular level
"""
from src.dataset import construct_kernelbench_dataset

run_name = "test_hf_level_1"  # Replace this with your run name
level = 1  # change if needed

dataset = construct_kernelbench_dataset(level)


# load json
eval_file_path = f"runs/{run_name}/eval_results.json"
assert os.path.exists(eval_file_path), f"Eval file does not exist at {eval_file_path}"


with open(eval_file_path, "r") as f:
    eval_results = json.load(f)


# Initialize counters
total_count = len(dataset)
total_eval = len(eval_results)
compiled_count = 0
correct_count = 0

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
