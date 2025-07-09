import os
import argparse

from evaluation_utils import is_generated_kernel_used


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    for filename in os.listdir(args.run_dir):
        if filename.endswith(".py"):
            with open(os.path.join(args.run_dir, filename), "r") as f:
                kernel_src = f.read()
                print(f"Analyzing {filename}")
                is_used = is_generated_kernel_used(kernel_src)
                print(f"{filename}: Is kernel used? {is_used}")
