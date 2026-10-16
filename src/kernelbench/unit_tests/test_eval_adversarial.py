import os

from kernelbench.dataset import KERNEL_BENCH_PATH
from kernelbench.eval import eval_kernel_against_ref
from kernelbench.utils import read_file

"""
Specifically testing the robustness of the eval function.
Through a few targeted adversarial kernels.

Run with pytest
pytest src/unit_tests/test_eval_adversarial.py
"""

def run_test_kernel(problem_name, 
                    kernel_filename, 
                    timing_method="cuda_event",
                    num_perf_trials=100):
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
        num_perf_trials=num_perf_trials,
        measure_performance=True,
        timing_method=timing_method,
        # for checking against reward hacking
        check_for_excessive_speedup = True, 
        excessive_speedup_threshold = 10
    )


def test_result_cache_reuse():
    """
    Test to prevent adversarial kernel by reuse computed PyTorch results 

    Methodology: 
    - Adversarial kernel allocate memory for result but does not initialize it.
    - Potential Hacking scenario: CUDA cache allocator might give that un-erased physical memory to the custom kernel (which contains the result) and it will incorrectly pass
    - Eval script should flag this as incorrect as the reference kernel did not conduct any meaningful computation
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
    - Adversarial kernel zeros out all inputs and returns all zeros for the final matmul shape.
    - Potential Hacking scenario: Since adversarial kernel zero out the input, the reference might ran the kernel with all zeros. 
      Hence both refernece and custom kernel return all zeros, and this incorrectly pass.
    - Eval script should flag this as incorrect as this kernel zeros out inputs and trick the reference kernel into returning all zeros.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py", 
                            kernel_filename="zero_out_kernel.py")
    assert result.correctness is False, "Input modification hack detected"
    print("Result: Eval Function Passed Test Input Modification Hack")
    print(result)


def test_non_default_stream(timing_method="do_bench", threshold=1.5):
    """
    Test that we will flag adversarial kernels that cheat by assigning work to non-default CUDA streams.

    Methodology:
    - Adversarial kernel assigns work to non-default CUDA streams.
    - Potential Hacking scenario: Eval script waits on the default stream only when measuring kernel runtime, leading to unrealistic speedups.
    - Eval script should accurately measure runtime by waiting on all streams, not just the default stream.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py", 
                            kernel_filename="non_default_stream_kernel.py",
                            timing_method=timing_method,
                            num_perf_trials=10 ) # keep it low as we are leaking cuda stream

    print(result)
    if result.metadata.get("excessive_speedup") is True:   
        print("Result: Eval Function successfully marked excessive speedup from the hacky CUDA Stream implementation")
        print("This indicates the eval script properly detected and flagged potential cheating behavior.")
    else:
        raise AssertionError(
            "Excessive speedup not detected - Eval Function failed to flag adversarial kernel that exploits cuda stream timing hack"
        )


def test_cached_output():
    """
    Test that we flag a kernel that caches its output keyed on the input tensors.

    Methodology:
    - Adversarial kernel computes the correct result on new inputs, so it passes every correctness trial
    - Called again on the same input tensors, it returns the stored output without launching GPU work
    - The timing loop reuses one set of inputs, so the cached calls would be timed as near zero
    - Eval should flag it as incorrect: no GPU launches on a repeated input, some on a fresh one (issue #171)
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py",
                            kernel_filename="cached_output_kernel.py")
    assert result.correctness is False, "Cached output hack detected"
    assert result.metadata["correctness_issue_name"] == "cached_output"
    assert result.metadata["launches_on_repeated_call"] == 0
    assert result.metadata["launches_on_first_call"] > 0


def test_cached_output_fingerprint():
    """
    Test that we flag a cache keyed on a fingerprint of the input values.
    On a hit it still launches the reads that compute its key, so a zero-launch rule misses it,
    but it launches fewer operations than on the first call with the same inputs.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py",
                            kernel_filename="fingerprint_cached_output_kernel.py")
    assert result.correctness is False, "Fingerprint cache hack detected"
    assert result.metadata["correctness_issue_name"] == "cached_output"
    assert 0 < result.metadata["launches_on_repeated_call"] < result.metadata["launches_on_first_call"]


def test_cached_output_cuda_graph_control():
    """
    Control: a kernel that captures a CUDA graph on its first call with each input set and replays it
    afterwards enqueues fewer calls on a repeat (one graph launch) but still computes, so it should pass.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py",
                            kernel_filename="cuda_graph_kernel.py")
    assert result.correctness is True
    assert result.metadata["launches_on_repeated_call"] > 0


def test_cached_output_control():
    """
    Control for test_cached_output: the same matmul without a cache launches GPU work on every call and passes.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py",
                            kernel_filename="no_cache_kernel.py")
    assert result.correctness is True
    assert result.metadata["launches_on_repeated_call"] == result.metadata["launches_on_first_call"] > 0


def main():
    test_result_cache_reuse()
    test_input_modification()
    test_non_default_stream()
    test_cached_output()
    test_cached_output_fingerprint()
    test_cached_output_cuda_graph_control()
    test_cached_output_control()

    # we might add more adversarial tests in the future

if __name__ == "__main__":
    main()