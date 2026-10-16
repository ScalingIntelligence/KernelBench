import os

import pytest
import torch

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
                    num_perf_trials=100,
                    fresh_perf_inputs=True):
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
        excessive_speedup_threshold = 10,
        fresh_perf_inputs=fresh_perf_inputs,
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


@pytest.mark.parametrize("timing_method", ["cuda_event", "do_bench_impl", "host_time"])
@pytest.mark.parametrize("kernel_filename", [
    "input_memoization_kernel.py",        # remembers the last input
    "multi_input_memoization_kernel.py",  # remembers the last 4 inputs
    "fingerprint_memoization_kernel.py",  # remembers outputs by the first few input values
])
def test_input_memoization(kernel_filename, timing_method):
    """
    Test that a kernel memoizing on its inputs is timed on real work.

    Methodology:
    - Adversarial kernels cache their output and return it whenever they see the same inputs again,
      by tensor identity (last input, last 4 inputs) or by a fingerprint of the first few values.
    - Potential Hacking scenario: the timing loop passes the same input tensors to every trial,
      or cycles through a few, so the kernel computes once and every later timed call is a cache hit.
      It is correct on every correctness trial, so only the timing is wrong.
    - Eval script should draw new inputs (new tensors, new values) for every timed call,
      so the kernel computes every call and its runtime is close to the reference's.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py",
                            kernel_filename=kernel_filename,
                            timing_method=timing_method)
    print(result)
    assert result.correctness is True, "Memoizing kernel is functionally correct and should pass correctness"
    assert result.metadata.get("excessive_speedup") is not True, (
        f"{kernel_filename} got an excessive speedup: timed calls are reusing input tensors"
    )
    # it is plain torch.matmul, so it should time about the same as the reference
    assert result.runtime >= 0.5 * result.ref_runtime


def test_input_memoization_without_fresh_inputs():
    """
    Control for test_input_memoization: without fresh inputs the hack works,
    which shows the test kernel really exploits the reused inputs.
    """
    result = run_test_kernel(problem_name="level1/1_Square_matrix_multiplication_.py",
                            kernel_filename="input_memoization_kernel.py",
                            fresh_perf_inputs=False)
    print(result)
    assert result.correctness is True
    assert result.metadata.get("excessive_speedup") is True


# ReLU problem shaped like level1/19 but scaled down to fit a 16 GB GPU
# (level1/19 itself uses 4096 x 393216 fp32, a 6.4 GB input set)
RELU_REF_SRC = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

batch_size = 4096
dim = 393216 // 8

def get_inputs():
    return [torch.rand(batch_size, dim)]

def get_init_inputs():
    return []
"""

# the memoizer from the review of #173: keeps the last 4 inputs
RELU_MEMOIZER_SRC = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = []  # [(x, out)], last 4 inputs

    def forward(self, x):
        for seen, out in self._cache:
            if seen is x:
                return out
        out = torch.relu(x)
        self._cache = (self._cache + [(x, out)])[-4:]
        return out
"""


def test_relu_multi_input_memoization():
    """
    The case from the review of #173: on level1/19 a pool capped by memory held only 2 input sets,
    so a ReLU memoizer keeping the last 4 inputs got 89x. With new inputs drawn per timed call there
    is no pool to cycle, so it should time like the reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    result = eval_kernel_against_ref(
        original_model_src=RELU_REF_SRC,
        custom_model_src=RELU_MEMOIZER_SRC,
        num_correct_trials=5,
        num_perf_trials=100,
        measure_performance=True,
        timing_method="cuda_event",
        check_for_excessive_speedup=True,
        excessive_speedup_threshold=10,
    )
    print(result)
    assert result.correctness is True
    assert result.metadata.get("excessive_speedup") is not True
    assert result.runtime >= 0.5 * result.ref_runtime


def test_fresh_inputs_memory_does_not_grow_with_trials():
    """
    Drawing new inputs per timed call should keep about one extra input set live,
    however many trials run, unlike a pool that holds one set per trial.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from kernelbench import timing
    from kernelbench.eval import make_fresh_inputs

    def get_inputs():
        return [torch.rand(4096, 65536)]  # 1 GiB fp32

    device = torch.device("cuda")
    set_bytes = 4096 * 65536 * 4
    inputs = [x.to(device) for x in get_inputs()]
    fresh_args = make_fresh_inputs(get_inputs, seed=42, device=device)

    peaks = {}
    for num_trials in (10, 50):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        timing.time_execution_with_cuda_event(
            torch.relu, inputs, num_warmup=3, num_trials=num_trials,
            verbose=False, device=device, fresh_args=fresh_args,
        )
        peaks[num_trials] = torch.cuda.max_memory_allocated() - baseline
    print({k: f"{v / 2**30:.2f} GiB" for k, v in peaks.items()})

    # one fresh set, this and the previous call's relu output, the L2 flush buffer (256 MB) and slack;
    # a pool with one set per trial would need num_trials sets
    assert peaks[50] < 4 * set_bytes, f"peak {peaks[50] / 2**30:.2f} GiB, expected about 3 GiB"
    assert peaks[50] <= peaks[10] * 1.1, "peak memory should not grow with the number of trials"


def main():
    test_result_cache_reuse()
    test_input_modification()
    test_non_default_stream()
    test_input_memoization("multi_input_memoization_kernel.py", "cuda_event")
    test_input_memoization_without_fresh_inputs()

    # we might add more adversarial tests in the future

if __name__ == "__main__":
    main()