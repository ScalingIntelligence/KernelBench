####
# Profiling Related Functions
# TODO: @kesavan @simon @arya
####

import torch
import nsight

# wrapper with tool to measure hardware metric


# nsight-python
# https://docs.nvidia.com/nsight-python/overview

def check_ncu_available() -> bool:
    """Check if ncu is in PATH. Returns True if found, False otherwise."""
    from shutil import which
    return which('ncu') is not None


# Note you need access to hardware counter
# you also need to have ncu installed and point to the right path
# sudo env "PATH=$PATH" $(which python) src/profile.py
@nsight.analyze.kernel
def benchmark_matmul(n):
    """
    The simplest possible benchmark.
    We create two matrices and multiply them.
    follow example from nsight-python docs
    """
    # Create two NxN matrices on GPU
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    # Mark the kernel we want to profile
    with nsight.annotate("matmul"):
        c = a @ b

    return c

if __name__ == "__main__":
    if not check_ncu_available():
        print("ncu not found in PATH. Please install ncu and point to path.")
        exit(1)
    # Run the benchmark
    result = benchmark_matmul(1024)

# pytorch profiler
# migrate from old repo during ICML / caesar repo