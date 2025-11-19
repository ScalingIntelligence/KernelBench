import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from timing import (
    time_execution_with_cuda_event,
    time_execution_with_time_dot_time,
    time_execution_with_do_bench,
)


"""
Test Timing

We want to systematically study different timing methodologies.

"""
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# use exampls in the few shot directory
EXAMPLES_PATH = os.path.join(REPO_PATH, "src", "prompts", "few_shot")

# Configure your test cases here
TEST_REF_FILE = "model_ex_tiled_matmul.py"
TEST_KERNEL_FILE = "model_new_ex_tiled_matmul.py"

assert os.path.exists(os.path.join(EXAMPLES_PATH, TEST_REF_FILE)), f"Reference file {TEST_REF_FILE} does not exist in {EXAMPLES_PATH}"
assert os.path.exists(os.path.join(EXAMPLES_PATH, TEST_KERNEL_FILE)), f"Kernel file {TEST_KERNEL_FILE} does not exist in {EXAMPLES_PATH}"


def _run_timing_smoke_test(timing_fn):
    """
    Scaffold function for timing smoke tests.
    
    Args:
        timing_fn: The timing function to test
        use_args: Whether the timing function expects args parameter (True for cuda_event/time_dot_time, False for do_bench)
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping timing tests")
    
    # Create test matrices
    size = 512
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    num_warmup = 5
    num_trials = 5
    
    # Define the kernel function to time
    def matmul_kernel(a, b):
        return torch.matmul(a, b)
    
    elapsed_times = timing_fn(
        matmul_kernel,
        args=[a, b],
        num_warmup=num_warmup,
        num_trials=num_trials,
        verbose=False,
    )
    
    # Validate results
    assert isinstance(elapsed_times, list), "Expected list of elapsed times"
    assert len(elapsed_times) == num_trials, f"Expected {num_trials} timing results, got {len(elapsed_times)}"
    assert all(isinstance(t, float) for t in elapsed_times), "All timing results should be floats"
    assert all(t > 0 for t in elapsed_times), "All timing results should be positive"


def test_time_execution_with_cuda_event_smoke():
    """
    Smoke test for time_execution_with_cuda_event using 512x512 matmul.
    Tests with 5 warmup and 5 trials, validates list of 5 positive floats is returned.
    """
    _run_timing_smoke_test(time_execution_with_cuda_event)


def test_time_execution_with_time_dot_time_smoke():
    """
    Smoke test for time_execution_with_time_dot_time using 512x512 matmul.
    Tests with 5 warmup and 5 trials, validates list of 5 positive floats is returned.
    """
    _run_timing_smoke_test(time_execution_with_time_dot_time)


def test_time_execution_with_do_bench_smoke():
    """
    Smoke test for time_execution_with_do_bench using 512x512 matmul.
    Tests with 5 warmup and 5 trials, validates list of 5 positive floats is returned.
    """
    _run_timing_smoke_test(time_execution_with_do_bench)


