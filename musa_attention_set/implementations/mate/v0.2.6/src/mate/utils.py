"""Shared MATE Python helpers."""

import functools
from typing import Any
from collections import OrderedDict
from collections.abc import Callable

import torch


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 256). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache: "OrderedDict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], tuple[tuple[Any, ...], dict[str, Any], Any]]" = OrderedDict()
    cache_size = 256

    def get_id(x: Any):
        if (type(x) is int) or (type(x) is float) or (type(x) is str):
            return x
        else:
            return id(x)

    def make_identity_key(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...]]:
        args_key = tuple(get_id(a) for a in args)
        kwargs_key = tuple(sorted((k, get_id(v)) for k, v in kwargs.items()))
        return args_key, kwargs_key

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        key = make_identity_key(args, kwargs)
        if key in cache:
            cache.move_to_end(key, last=True)
            _, _, cached_result = cache[key]
            return cached_result

        result = fn(*args, **kwargs)
        cache[key] = (args, kwargs, result)
        cache.move_to_end(key, last=True)
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return result

    return wrapper


def _row_major_strides(shape):
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def cosize(shape, strides=None):
    """Return the storage span for a shape/stride layout, measured in elements."""

    shape = tuple(shape)
    if strides is None:
        strides = _row_major_strides(shape)
    else:
        strides = tuple(strides)
        assert len(shape) == len(strides), "shape and strides must have the same rank"

    size = 1
    for extent, stride in zip(shape, strides):
        size += (extent - 1) * stride
    return size


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


__all__ = ["ceil_div", "cosize", "round_up", "tensor_cache"]
