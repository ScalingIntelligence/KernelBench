import os

import torch

from src.dataset import KERNEL_BENCH_PATH
from src.eval import eval_kernel_against_ref, load_original_model_and_inputs, set_seed
from src.timing import get_timing_stats, get_timing_function
from src.utils import read_file

"""
Specifically testing the robustness of the eval function.
Currently focus on correctness.

Run with pytest
pytest src/unit_tests/test_eval.py
"""

def measure_ref_kernel(problem_name):
    ref_src = read_file(os.path.join(KERNEL_BENCH_PATH, problem_name))
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_src, {})

    device = torch.device("cuda")
    init_inputs = get_init_inputs()
    init_inputs = [x.to(device) for x in init_inputs]

    with torch.no_grad():
        set_seed(42)
        model = Model(*init_inputs).to(device=device, dtype=torch.float32)
        torch.cuda.synchronize(device=device)

    inputs = get_inputs()
    inputs = [x.to(device) for x in inputs]
    timing_func = get_timing_function("cuda_event")
    elapsed = timing_func(
        model,
        *inputs,
        num_trials=5,
        verbose=False,  
        device=device,
    )
    stats = get_timing_stats(elapsed)
    return stats

def run_test_kernel(problem_name, kernel_filename):
    """
    Run a test kernel agains the reference architecture and return the result.

    Inputs:
    - problem_name: name of the KernelBench problem to test on (e.g. "level1/1_Square_matrix_multiplication_.py")
    - kernel_filename: name of the adversarial kernel (e.g. "zero_out_kernel.py")

    Output:
    - result: KernelExecResult object
    """
    print(f"Running test adversarial kernel {kernel_filename} against problem {problem_name}")
    ref_src = read_file(os.path.join(KERNEL_BENCH_PATH, problem_name))
    # TODO: replace with Omar's dataset object
    kernel_src = read_file(os.path.join(os.path.dirname(__file__), "test_kernels", kernel_filename))
    return eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=kernel_src,
        num_correct_trials=5, # under mulitiple correctness trials
        num_perf_trials=5,
        measure_performance=True,
        timing_method="cuda_event"
    )


def test_result_cache_reuse():
    """
    Test to prevent advserial kernel by reuse computed PyTorch results 

    Methodology: 
    - Advserial kernel allocate memory for result but does not initialize it.
    - Potential Hacking scenario: CUDA cache allocator might give that un-erased physical memory to the custom kernel (which contains the result) and it will incorrectly pass
    - Eval script should flag this as incorrect as the reference kernel did not conduct any menaingful computation
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py", 
                            kernel_filename="result_reuse_kernel.py")
    assert result.correctness is False, "Cache reuse hack detected"
    print("Result: Eval Function Passed Test Result Cache Reuse")
    print(result)

def test_input_modification():
    """
    Test that even if the model tries to modify the inputs to pass the tests,
    we still flag it as incorrect.

    Methodology:
    - Advserial kernel zeros out all inputs and returns all zeros for the final matmul shape.
    - Potential Hacking scenario: Since adversarial kernel zero out the input, the reference might ran the kernel with all zeros. 
      Hence both refernece and custom kernel return all zeros, and this incorrectly pass.
    - Eval script should flag this as incorrect as this kernel zeros out inputs and trick the refernece kernel into returning all zeros.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py", 
                            kernel_filename="zero_out_kernel.py")
    assert result.correctness is False, "Input modification hack detected"
    print("Result: Eval Function Passed Test Input Modification Hack")
    print(result)


def test_non_default_stream(threshold=1.5):
    """
    Test that we will flag adversarial kernels that cheat by assigning work to non-default CUDA streams.

    Methodology:
    - Adversarial kernel assigns work to non-default CUDA streams.
    - Potential Hacking scenario: Eval script waits on the default stream only when measuring kernel runtime, leading to unrealistic speedups.
    - Eval script should accurately measure runtime by waiting on all streams, not just the default stream.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py", 
                            kernel_filename="non_default_stream_kernel.py")
    print(result)
    assert "excessive_speedup" in result.metadata, "Excessive speedup detected"
    print("Result: Eval Function passed Non Default Stream Hack")
    print(result)


def main():
    # test_result_cache_reuse()
    # test_input_modification()
    test_non_default_stream()

if __name__ == "__main__":
    main()