import functools
import os
from typing import Callable

import torch

from .utils import bench_kineto as bench_kineto
from .utils import calc_diff as calc_diff
from .utils import count_bytes as count_bytes
from .utils import empty_suppress as empty_suppress
from .utils import suppress_stdout_stderr as suppress_stdout_stderr

__all__ = [
    "bench",
    "bench_kineto",
    "calc_diff",
    "count_bytes",
    "empty_suppress",
    "get_arch_major",
    "ignore_env",
    "suppress_stdout_stderr",
    "test_filter",
]


def _get_device_type() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("DeepGEMM benchmark helpers require a MUSA or CUDA device")


def _get_accelerator(device_type: str):
    return getattr(torch, device_type)


def bench(
    fn,
    num_warmups: int = 5,
    num_tests: int = 10,
    high_precision: bool = False,
):
    device_type = _get_device_type()
    accelerator = _get_accelerator(device_type)

    # Touch a large buffer to evict most L2 cache contents before warmup.
    accelerator.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)
    cache.zero_()

    for _ in range(num_warmups):
        fn()

    # Add a large kernel to reduce CPU launch overhead for short kernels.
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float, device=device_type)
        y = torch.randn((8192, 8192), dtype=torch.float, device=device_type)
        x @ y

    start_event = accelerator.Event(enable_timing=True)
    end_event = accelerator.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_tests):
        fn()
    end_event.record()
    accelerator.synchronize()

    return start_event.elapsed_time(end_event) / num_tests / 1e3


def get_arch_major() -> int:
    if hasattr(torch, "musa") and torch.musa.is_available():
        major, _ = torch.musa.get_device_capability()
        return major
    major, _ = torch.cuda.get_device_capability()
    return major


def test_filter(condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                func(*args, **kwargs)
            else:
                print(f"{func.__name__}:")
                print(f" > Filtered by {condition}")
                print()

        return wrapper

    return decorator


def ignore_env(name: str, condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                saved = os.environ.pop(name, None)
                func(*args, **kwargs)
                if saved is not None:
                    os.environ[name] = saved
            else:
                func(*args, **kwargs)

        return wrapper

    return decorator
