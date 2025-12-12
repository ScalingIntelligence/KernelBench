import torch
import json
import numpy as np
import time
import warnings
from typing import Any
import os
from triton import runtime as triton_runtime
from triton import testing as triton_testing    

################################################################################
# Performance Eval
################################################################################

def clear_l2_cache(device: str = "cuda"):
    """
    Clear L2 Cache line by thrashing
    From GPU mode reference kernel repo:
    https://github.com/gpu-mode/reference-kernels/commit/7c15075a39286e88939d99d3f3a60be88b8e6223#diff-3a30a71cbf8db2badd224f4d92f9a2546925a5b522632a31d353526b7a5f3338R158-R163

    We can improve this 
    TODO; should prob check device_name
    """
    # don't reserve space for persisting lines
    # cp.cuda.runtime.cudaDeviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 0)
    
    # Thrash L2 cache by creating a larger dummy tensor, effectively flushing the cache
    # 32 * 1024 * 1024 * 8B = 256MB 
    # NOTE: we can make this more adaptive based on device
    # L2 cache sizes: A100=40MB, H100=50MB, H200=90MB, RTX4090=72MB, L40S=48MB, Blackwell≈192MB → overwrite >200MB to fully thrash L2
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    # write to tenosr with inplace fill
    dummy.fill_(42) 
    del dummy

def clear_l2_cache_triton(cache=None, device: str = "cuda"):
    """
    Thrash the cache by making a large dummy tensor, using triton runtime's functionality
    """
    with torch.cuda.device(device):
        cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()
        triton_runtime.driver.active.clear_cache(cache)


def get_timing_function(
    method: str = "cuda_event", # by default 
) -> callable:
    """
    Get the timing function based on different timing methods
    """
    print(
        f"[Profiling] Using timing method: {method}"
    )
    match method:
        case "cuda_event":
            return time_execution_with_cuda_event
        case "do_bench_interface":
            return time_execution_with_do_bench_interface
        case "do_bench_impl":
            return time_execution_with_do_bench_impl
        case "cpu_time":
            return time_execution_with_cpu_time 
        # we might add other methods in the future
        case _: 
            raise ValueError(f"Unsupported timing method: {method}")

"""
Kernel Timing Functions [Revamp WIP]
TODO: see our detailed study on how to time kernel execution and benchmarking guide
we implement a few ways to do timing studies 
These should be implemnted to be agnostic whether the modules are rather Model (reference kernel) or ModelNew (generated kernel)
"""


def time_execution_with_cuda_event(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event
    The first version of KenrelBench used this for evaluation.
    We care about cold cache performance here.

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    TODO: double check this with team 
    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    # note this only release PyTorch’s CUDA caching allocator, not necessarily clearing device's L2 cache
    torch.cuda.empty_cache()

    print(f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )

    elapsed_times: list[float] = [] # in ms

    # Timing trials
    for trial in range(num_trials + discard_first):
        torch.cuda.synchronize(device=device) # block on all streams

        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        clear_l2_cache() # measuring cold cache performance
        
        # note cuda events mark event on current stream
        start_event.record()
        _ = kernel_fn(*args)
        end_event.record() 

        # waits for all streams on that device
        # though it is important to note the events only record time between on current stream
        # TODO: find ways to check hacks by launching work on additional stream
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if trial >= discard_first:
            if verbose:
                logical_idx = trial - discard_first + 1
                print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def time_execution_with_do_bench_interface(
    kernel_fn: callable,
    args: list[Any],
    # this is different for triton do_bench
    num_warmup: int = 3, 
    num_trials: int = 10,
    discard_first: int = 1, # not used yet
    verbose: bool = True,
    device: torch.device | None = None) -> list[float]:
    """
    Just using triton's do_bench as it is 
    """

    do_bench_fn = lambda : kernel_fn(*args)
    return triton_testing.do_bench(fn=do_bench_fn,
            warmup=25,
            rep=100, 
            grad_to_none=None, 
            quantiles=None, 
            return_mode="all")


def time_execution_with_do_bench_impl(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1, # not used yet
    verbose: bool = True,
    device: torch.device | None = None) -> list[float]:
    """
    This is modifying the triton do_bench codebase
    See Triton's implementation for more details
    https://github.com/triton-lang/triton/blob/9073370d5979218d1afa44ec895bbd80e7419a8c/python/triton/testing.py#L127
    """

    device = torch.cuda.current_device() if device is not None else device
    if verbose: 
        print(f"Using do_bench to evaluate kernel on {device}")

    # speicfy device interface (supports both nvidia and amd)
    # under the hood, di is torch.cuda (amd uses a cuda compatible interface)
    di = triton_runtime.driver.active.get_device_interface()

    kernel_fn(*args)
    di.synchronize(device=device)

    # clear l2 cache
    cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()

    # do_bench Estimate the runtime of the function 
    # Here we are not using it not needed since now the warmup and repeat steps are set by the user)
    # start_event = di.Event(enable_timing=True)
    # end_event = di.Event(enable_timing=True)
    # start_event.record()
    # for _ in range(5):
    #     triton_runtime.driver.active.clear_cache(cache)
    #     kernel_fn(*args)
    # end_event.record()
    # di.synchronize()
    # estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    # Change
    # n_warmup = max(1, int(warmup / estimate_ms))
    # n_repeat = max(1, int(rep / estimate_ms))
    # n_warmup = warmup
    # n_repeat = rep
    # end of change
    start_event = [di.Event(enable_timing=True) for i in range(num_trials)]
    end_event = [di.Event(enable_timing=True) for i in range(num_trials)]
    # Warm-up
    for _ in range(num_warmup):
        kernel_fn(*args)
    # Benchmark
    for i in range(num_trials):
        # All KernelBench functions are forward passes, so we don't need to reset gradients
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        # if grad_to_none is not None:
        #     for x in grad_to_none:
        #         x.grad = None
        
        # we clear the L2 cache before each run
        triton_runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        kernel_fn(*args)
        end_event[i].record()
    # Record clocks
    di.synchronize(device=device)
    if verbose: print('Done with do_bench evaluation')
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return times


def time_execution_with_cpu_time(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using CPU side timing
    [WIP]
    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds

    Not recommended: 
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}")
    elapsed_times = []

    # clear PyTorch allocator cache
    torch.cuda.empty_cache()

    # Actual trials
    for trial in range(num_trials + discard_first):
        # block all streams on device
        torch.cuda.synchronize(device=device)

        # focus on cold_cache performance
        clear_l2_cache()

        # CPU-side wall clock time using perf_counter (high-resolution timer)
        start_time = time.perf_counter()
        kernel_fn(*args)
        torch.cuda.synchronize(device=device) # wait for all stream to finish
        # this blocks the CPU until all GPU work on device is done
        # this means all kernels on all streams
        end_time = time.perf_counter()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        if trial >= discard_first:
            if verbose:
                logical_idx = trial - discard_first + 1
                print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times

########################################################
# Timing stats
#########################################################
def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    # TODO: replace with the new Dataset object that Omar will merge in
    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats
