import pytest
from kernelbench.score import *
import math

"""
Usage:
pytest test_score.py
"""

abs_tol = 0.0000001


def test_geometric_mean_speed_ratio():
    """Test geometric mean calculations with representative test cases"""

    # Test case with mixed correct/incorrect results
    is_correct = [1, 0, 1, 1, 0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    # Test correct-only metric
    assert math.isclose(
        geometric_mean_speed_ratio_correct_only(
            is_correct, baseline_speed, actual_speed, n
        ),
        1.185631101,
        abs_tol=abs_tol,
    )

    # Test correct-and-faster-only metric
    assert math.isclose(
        geometric_mean_speed_ratio_correct_and_faster_only(
            is_correct, baseline_speed, actual_speed, n
        ),
        5,
        abs_tol=abs_tol,
    )

    # Test edge case: no correct samples
    is_correct_none = [0, 0, 0, 0, 0]
    assert (
        geometric_mean_speed_ratio_correct_only(
            is_correct_none, baseline_speed, actual_speed, n
        )
        == 0
    )

    assert (
        geometric_mean_speed_ratio_correct_and_faster_only(
            is_correct_none, baseline_speed, actual_speed, n
        )
        == 0
    )


def test_fastp():
    """Test fastp metric with different thresholds"""

    is_correct = [1, 0, 1, 1, 0]
    baseline_speed = [0.1, 0.15, 0.2, 0.05, 0.3]
    actual_speed = [0.2, 0.15, 0.3, 0.01, 0.2]
    n = 5

    # Test with threshold p=1.0 (strict speedup)
    assert math.isclose(
        fastp(is_correct, baseline_speed, actual_speed, n, 1.0), 0.2, abs_tol=abs_tol
    )

    # Test with threshold p=0.5 (allowing more tolerance)
    assert math.isclose(
        fastp(is_correct, baseline_speed, actual_speed, n, 0.5), 0.4, abs_tol=abs_tol
    )

    # Edge case: no correct samples
    is_correct_none = [0, 0, 0, 0, 0]
    assert fastp(is_correct_none, baseline_speed, actual_speed, n, 1.0) == 0
