import math
import statistics
from triton import runtime


# pure Python implementation of np.quantile/torch.quantile
# to avoid unnecessary runtime dependency on numpy/torch

# This is a slightly modfied version of triton.testing.do_bench (triton v3.5.x) from 
# https://github.com/triton-lang/triton/blob/0add68262ab0a2e33b84524346cb27cbb2787356/python/triton/testing.py#L127
# with minor a minor modification to support having warmup and repeat time instead be specified in number of iterations
# instead of ms. All changes are explcitly marked

def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile. CHANGE: warmup and repeat time are specified in number of iterations rather than ms
    

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in number of iterations)
    :type warmup: int
    :param rep: Repetition time (in number of iterations)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    # Change
    # mean, max, min, quantiles, etc. make no sense with 0 reps
    if not (return_mode == "all" and quantiles is None) and rep < 1:
        error_msg = (
            f"You are running with {rep} reps. This is likely a mistake!!!\n"
            "We do let you do this, but ONLY when quantiles is None when return_mode is not 'all'\n"
            "to be consistent with the rest of KernelBench's timing functions"
        )
        raise ValueError(error_msg)
    # End of change
    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    # Change
    # n_warmup = max(1, int(warmup / estimate_ms))
    # n_repeat = max(1, int(rep / estimate_ms))
    n_warmup = warmup
    n_repeat = rep
    # end of change
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)