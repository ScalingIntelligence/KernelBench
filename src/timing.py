import torch
import json
import numpy as np
import time
import warnings
from typing import Any
import os
from do_bench import do_bench

################################################################################
# Performance Eval
################################################################################

#############################################################
# Timing Functions 
# TODO: see our detailed study on how to time kernel execution
# we implement a few ways to do timing studies 
# agnositic whether the modules are rather Model or ModelNew
#############################################################

def get_timing_function(
    method: str = "cuda_event", # by default 
) -> callable:
    """
    Get the timing function based on different timing methods
    """
    assert method in ["cuda_event", "do_bench", "time_time"]
    print(
        f"[Profiling] Using timing method: {method}"
    )
    match method:
        case "cuda_event":
            return time_execution_with_cuda_event
        case "do_bench":
            return time_execution_with_do_bench
        case "time_time":
            return time_execution_with_tim_dot_time
        case _: 
            raise ValueError(f"Unknown timing method: {method}")    

def time_execution_with_do_bench(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using triton.do_bench
    """
    return do_bench(
        lambda: kernel_fn(*args),
        warmup=num_warmup,
        rep=num_trials,
        return_mode="all",
    )
    

def time_execution_with_time_dot_time(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using time.time()

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
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

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
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

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    TODO: make this super solid and check this
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
        torch.cuda.clear_cache()

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)
        torch.cuda.clear_cache()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
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
