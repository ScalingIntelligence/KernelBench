import os
import json
from configs import parse_autorule_args

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_DIR, "KernelBench")

def read_best_k_kernels(level: int):
    with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "r") as f:
        best_k_kernels = json.load(f)
    return best_k_kernels

def main(config):
    best_k_kernels = read_best_k_kernels(config.level)
    total_combinations = 0
    for problem, kernels in best_k_kernels.items():
        n = len(kernels)
        if n >= 2:
            combinations = n * (n - 1) // 2
            print(f"Number of combinations in {problem}: {combinations}")
            total_combinations += combinations
    print(f"Total number of all possible combinations of 2 kernels per problem: {total_combinations}")


if __name__ == "__main__":
    config = parse_autorule_args()
    main(config)