import os

from src.dataset import KERNEL_BENCH_PATH
from src.eval import eval_kernel_against_ref
from src.utils import read_file

def run_test_kernel(problem_name, kernel_filename):
    """
    Run a test kernel agains the reference architecture and return the result.

    Inputs:
    - problem_name: name of the KernelBench problem to test on (e.g. "level1/1_Square_matrix_multiplication_.py")
    - kernel_filename: name of the adversarial kernel (e.g. "zero_out_kernel.py")

    Output:
    - result: KernelExecResult object
    """
    ref_src = read_file(os.path.join(KERNEL_BENCH_PATH, problem_name))
    kernel_src = read_file(os.path.join(os.path.dirname(__file__), "test_kernels", kernel_filename))
    return eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=kernel_src,
        num_correct_trials=1,
        num_perf_trials=1,
        measure_performance=False,
    )

def test_input_modification():
    """
    Test that even if the model tries to modify the inputs to pass the tests,
    we still flag it as incorrect.
    """
    result = run_test_kernel("level1/1_Square_matrix_multiplication_.py", "zero_out_kernel.py")
    assert result.correctness is False