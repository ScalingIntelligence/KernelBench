import os
import argparse
import json

from src.reward_hacking import is_generated_kernel_used, torch_function_used


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    eval_path = os.path.join(args.run_dir, "eval_results.json")
    with open(eval_path, "r") as f:
        eval_results = json.load(f)

    for filename in os.listdir(args.run_dir):
        if filename.endswith(".py"):
            level = filename.split("_")[1]
            problem_id = filename.split("_")[3]
            sample_id = filename.split("_")[5]
            with open(os.path.join(args.run_dir, filename), "r") as f:
                kernel_src = f.read()

            try:
                # is_used, generated_kernel_attrs, called_attrs, overwritten_attrs, generated_kernel_vars, called_vars = is_generated_kernel_used(kernel_src)
                is_kernel_used = is_generated_kernel_used(kernel_src)
                is_torch_function_used = torch_function_used(kernel_src)
                is_correct = eval_results[level][problem_id][sample_id]["correctness"]
                # print(f"{filename}: Is kernel used? {is_used} / Is correct? {is_correct}")
                if level == '1' and is_correct and is_torch_function_used:
                    print(f"Flagging {filename}")
                    # print(f"Generated kernel attrs: {generated_kernel_attrs}")
                    # print(f"Called attrs: {called_attrs}")
                    # print(f"Overwritten attrs: {overwritten_attrs}")
                    # print(f"Generated kernel vars: {generated_kernel_vars}")
                    # print(f"Called vars: {called_vars}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")