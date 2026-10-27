import pytest
from kernelbench.score import *
import math
import warnings

import numpy as np

'''
Usage:
pytest test_score.py
'''

abs_tol = 0.0000001

def test_geometric_mean_speed_ratio_correct_only():

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    '''
    Geometric mean of the speed ratio for correct samples
    '''
    assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 1.185631101, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 0.801816719, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 0, abs_tol=abs_tol)


def test_geometric_mean_speed_ratio_correct_and_faster_only():

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    '''
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    '''

    assert math.isclose(geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n), 5, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n), 1.033333333, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    assert math.isclose(geometric_mean_speed_ratio_correct_and_faster_only(is_correct, baseline_speed, actual_speed, n), 0, abs_tol=abs_tol)

def test_fastp():

    '''
    Rate of samples within a threshold p (1.0)
    '''

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 1.0

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.2, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5
    p = 1.0

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.2, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 1.0

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0, abs_tol=abs_tol)

    '''
    Rate of samples within a threshold p (0.5)
    '''

    is_correct = [1,0,1,1,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 0.5

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.4, abs_tol=abs_tol)

    is_correct = [1,1,1,1,0]
    baseline_speed = [0.24, 0.31, 100.0, 0.0001, 0.3]
    actual_speed = [0.3, 0.3, 200.0, 0.0001, 0.3]
    n = 5
    p = 0.5

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0.6, abs_tol=abs_tol)

    is_correct = [0,0,0,0,0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5
    p = 0.5

    assert math.isclose(fastp(is_correct, baseline_speed, actual_speed, n, p), 0, abs_tol=abs_tol)


def _assert_no_runtime_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return fn()


def test_geometric_mean_matches_log_space_reference_on_existing_values():
    is_correct = [1, 0, 1, 1, 0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    speedups = [0.1 / 0.2, 0.2 / 0.3, 0.05 / 0.01]
    expected = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    assert math.isclose(
        geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, 5),
        expected,
        abs_tol=abs_tol,
    )


def test_geometric_mean_does_not_overflow_or_underflow_product():
    """100 ratios whose product does not fit in float64. The mean does."""
    n = 100
    is_correct = np.ones(n, dtype=bool)
    baseline = np.full(n, 20.0)

    actual_fast = np.full(n, 0.001)  # 20,000x; (2e4)^100 overflows float64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        naive_fast = float(np.prod(baseline / actual_fast) ** (1 / n))
    assert math.isinf(naive_fast)
    stable_fast = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(is_correct, baseline, actual_fast, n)
    )
    assert math.isclose(stable_fast, 20000.0, rel_tol=1e-9)

    actual_slow = np.full(n, 200_000.0)  # 1e-4x; (1e-4)^100 underflows to 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        naive_slow = float(np.prod(baseline / actual_slow) ** (1 / n))
    assert naive_slow == 0.0
    stable_slow = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(is_correct, baseline, actual_slow, n)
    )
    assert math.isclose(stable_slow, 1e-4, rel_tol=1e-9)

    # One ratio is 1e308 / 1e-20 = 1e328, which is not a finite float.
    # Paired with 1e-100 the geometric mean 1e114 still fits.
    mixed_baseline = np.array([1e308, 1e-100])
    mixed_actual = np.array([1e-20, 1.0])
    mixed = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            np.ones(2, dtype=bool), mixed_baseline, mixed_actual, 2
        )
    )
    assert math.isclose(mixed, 1e114, rel_tol=1e-6)


def test_geometric_mean_extreme_but_representable_ratios():
    n = 50
    is_correct = np.ones(n, dtype=bool)
    tiny = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            is_correct, np.full(n, 1e-200), np.ones(n), n
        )
    )
    huge = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            is_correct, np.full(n, 1e200), np.ones(n), n
        )
    )
    assert math.isclose(tiny, 1e-200, rel_tol=1e-9)
    assert math.isclose(huge, 1e200, rel_tol=1e-9)


def test_geometric_mean_result_outside_float64_is_infinite_or_zero():
    is_correct = np.ones(2, dtype=bool)
    # geomean 1e400 does not fit in float64
    overflow = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            is_correct, np.array([1e200, 1e200]), np.array([1e-200, 1e-200]), 2
        )
    )
    assert math.isinf(overflow)
    underflow = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            is_correct, np.array([1e-200, 1e-200]), np.array([1e200, 1e200]), 2
        )
    )
    assert underflow == 0.0


def test_geometric_mean_skips_non_positive_nan_and_non_finite_timings():
    # Usable speedups: 10/2 = 5, 9/3 = 3, 4/5 = 0.8.
    # Dropped: actual -1, actual NaN, baseline inf, and the incorrect sample.
    is_correct = [1, 1, 1, 1, 1, 1, 0]
    baseline = [10.0, 9.0, 4.0, 4.0, 4.0, math.inf, 1e300]
    actual = [2.0, 3.0, -1.0, 5.0, math.nan, 2.0, 1e-300]
    expected = (5.0 * 3.0 * 0.8) ** (1.0 / 3.0)
    result = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            is_correct, baseline, actual, len(is_correct)
        )
    )
    assert math.isclose(result, expected, rel_tol=1e-12)

    # Negative timings are omitted even when their ratio would be positive.
    # A NaN baseline is omitted. The incorrect sample is omitted.
    only_invalid = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only(
            [1, 1], [-10.0, math.nan], [-2.0, 1.0], 2
        )
    )
    assert only_invalid == 0.0

    # 0/0 is not an infinite speedup.
    zero_over_zero = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only([1], [0.0], [0.0], 1)
    )
    assert zero_over_zero == 0.0


def test_geometric_mean_zero_runtime_is_infinite_speedup():
    result = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_only([1, 1], [3.0, 8.0], [1.0, 0.0], 2)
    )
    assert math.isinf(result) and result > 0

    # An incorrect sample with runtime 0 does not make the mean infinite.
    finite = geometric_mean_speed_ratio_correct_only([1, 0], [4.0, 4.0], [2.0, 0.0], 2)
    assert math.isclose(finite, 2.0, abs_tol=abs_tol)

    faster = geometric_mean_speed_ratio_correct_and_faster_only(
        [1, 1], [1.0, 5.0], [0.0, 10.0], 2
    )
    assert math.isinf(faster) and faster > 0


def test_geometric_mean_correct_and_faster_filters_before_the_mean():
    # Usable speedups: 0.5 (not faster), 5, and an invalid -1. Mean of [5] is 5.
    is_correct = [1, 1, 1]
    baseline = [1.0, 5.0, 3.0]
    actual = [2.0, 1.0, -1.0]
    result = geometric_mean_speed_ratio_correct_and_faster_only(
        is_correct, baseline, actual, 3
    )
    assert math.isclose(result, 5.0, abs_tol=abs_tol)

    n = 80
    stable = _assert_no_runtime_warning(
        lambda: geometric_mean_speed_ratio_correct_and_faster_only(
            np.ones(n, dtype=bool), np.full(n, 1e6), np.ones(n), n
        )
    )
    assert math.isclose(stable, 1e6, rel_tol=1e-9)


def test_geometric_mean_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        geometric_mean_speed_ratio_correct_only([1, 0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 3)

