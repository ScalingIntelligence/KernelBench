import math

import numpy as np


def _log_speedups(
    is_correct: np.ndarray,
    baseline_speed: np.ndarray,
    actual_speed: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Log speedups for correct samples, and whether any speedup is infinite.

    Speedup is baseline runtime / actual runtime. A finite positive speedup
    requires both timings to be finite and strictly positive, so the untimed
    sentinel (-1), NaNs, infinities, and other non-positive timings are not
    ratios and are left out. A correct sample with actual runtime 0 and a
    finite positive baseline has an infinite speedup.

    The log is log(baseline) - log(actual). Forming that difference never
    overflows a ratio that is too large for float64.
    """
    correct = np.asarray(is_correct, dtype=bool)
    baseline = np.asarray(baseline_speed, dtype=np.float64)
    actual = np.asarray(actual_speed, dtype=np.float64)
    if correct.shape != baseline.shape or baseline.shape != actual.shape:
        raise ValueError(
            "is_correct, baseline_speed, and actual_speed must have the same shape"
        )

    finite_positive_baseline = np.isfinite(baseline) & (baseline > 0)
    saw_infinite = bool(
        np.any(correct & finite_positive_baseline & np.isfinite(actual) & (actual == 0))
    )
    usable = correct & finite_positive_baseline & np.isfinite(actual) & (actual > 0)
    if not np.any(usable):
        return np.empty(0, dtype=np.float64), saw_infinite
    return np.log(baseline[usable]) - np.log(actual[usable]), saw_infinite


def _geometric_mean_from_log_speedups(log_speedups: np.ndarray) -> float:
    """exp(mean(log speedups)). An empty set is 0.

    A mean log outside the float64 range returns inf or 0. That is the
    geometric mean itself failing to fit, not an intermediate product.
    """
    if log_speedups.size == 0:
        return 0.0
    mean_log = float(np.mean(log_speedups))
    if math.isnan(mean_log):
        return float("nan")
    if math.isinf(mean_log):
        return float("inf") if mean_log > 0 else 0.0
    try:
        return math.exp(mean_log)
    except OverflowError:
        return float("inf")


def geometric_mean_speed_ratio_correct_only(
    is_correct: np.ndarray,
    baseline_speed: np.ndarray,
    actual_speed: np.ndarray,
    n: int,
) -> float:
    """Geometric mean of speedup over correct samples with usable timings.

    `n` is part of the existing call signature and is not the denominator.
    The mean runs over correct samples only. Non-positive, NaN, and
    non-finite timings are omitted. A correct sample with actual runtime 0
    makes the mean inf. If every correct sample is omitted, the result is 0.
    """
    del n
    log_speedups, saw_infinite = _log_speedups(is_correct, baseline_speed, actual_speed)
    if saw_infinite:
        return float("inf")
    return _geometric_mean_from_log_speedups(log_speedups)


def geometric_mean_speed_ratio_correct_and_faster_only(
    is_correct: np.ndarray,
    baseline_speed: np.ndarray,
    actual_speed: np.ndarray,
    n: int,
) -> float:
    """Geometric mean of speedup over correct samples that are strictly faster.

    Faster means a finite positive speedup above 1, or an infinite speedup
    from an actual runtime of 0. The same non-positive / NaN / non-finite
    timings as `geometric_mean_speed_ratio_correct_only` are omitted. `n` is
    unused and kept for the existing call signature. If none remain, the
    result is 0.
    """
    del n
    log_speedups, saw_infinite = _log_speedups(is_correct, baseline_speed, actual_speed)
    if saw_infinite:
        return float("inf")
    faster = log_speedups[log_speedups > 0]
    return _geometric_mean_from_log_speedups(faster)


def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0
