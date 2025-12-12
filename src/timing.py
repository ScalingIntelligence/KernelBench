import torch
import json
import numpy as np
import time
import warnings
from typing import Any
import os
from triton import runtime
from eval import clear_l2_cache

################################################################################
# Performance Eval
################################################################################

"""
Kernel Timing Functions [Revamp WIP]
TODO: see our detailed study on how to time kernel execution and benchmarking guide
we implement a few ways to do timing studies 
These should be implemnted to be agnostic whether the modules are rather Model (reference kernel) or ModelNew (generated kernel)
"""
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
        case "do_bench":
            return time_execution_with_do_bench
        case "time_time":
            return time_execution_with_time_dot_time # this is just for education purpose, don't use this
        case _: 
            raise ValueError(f"Unsupported timing method: {method}")

def time_execution_with_do_bench(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device | None = None) -> list[float]:
    """
    TODO: need check do_bench
    [WIP] need to check
    """

    device = torch.cuda.current_device() if device is not None else device

    if verbose: print("Using do_bench to evaluate kernel")

    # note: for both nvidia and amd, di is torch.cuda (amd uses a cuda compatible interface), so we could really just have torch.cuda
    di = runtime.driver.active.get_device_interface()

    kernel_fn(*args)
    di.synchronize(device=device)

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function (not needed since now the warmup and repeat steps are set by the user)

    # start_event = di.Event(enable_timing=True)
    # end_event = di.Event(enable_timing=True)
    # start_event.record()
    # for _ in range(5):
    #     runtime.driver.active.clear_cache(cache)
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

        # All our functions are forward passes, so we don't need to reset gradients
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        # if grad_to_none is not None:
        #     for x in grad_to_none:
        #         x.grad = None
        
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        kernel_fn(*args)
        end_event[i].record()
    # Record clocks
    di.synchronize(device=device)
    if verbose: print('Done with do_bench evaluation')
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return times


def time_execution_with_time_dot_time(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using time.time()
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

    # give warning that this is not the way to do it
    warnings.warn(
        "time_execution_with_time_dot_time is meant for educational purposes only, please other options like time_with_cuda_event or time_with_do_bench",
        UserWarning,
    )

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

    # Actual trials
    for trial in range(num_trials):
        start_time = time.time()
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)
        end_time = time.time()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times




def time_execution_with_cuda_event(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
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
    for trial in range(num_trials):
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
        if verbose:
            print(f"Timing Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
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
