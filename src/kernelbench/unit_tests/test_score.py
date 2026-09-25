import pytest
from kernelbench.score import *
import math
import warnings

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


# Correct samples that were never timed keep the default runtime of -1.0
# (measure_performance=False, or an error while measuring performance).
# They have no speedup, so the geometric mean leaves them out and warns.

def test_geometric_mean_speed_ratio_correct_only_one_untimed_correct_sample():

    is_correct = [1,1,1,0]
    baseline_speed = [0.2, 0.3, 0.5, 0.1]
    actual_speed = [0.1, 0.3, -1.0, -1.0]
    n = 4

    with pytest.warns(UserWarning, match="1 correct sample"):
        assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 1.414213562, abs_tol=abs_tol)


def test_geometric_mean_speed_ratio_correct_only_two_untimed_correct_samples():

    is_correct = [1,1,1]
    baseline_speed = [2.0, 3.0, 5.0]
    actual_speed = [1.0, -1.0, -1.0]
    n = 3

    with pytest.warns(UserWarning, match="2 correct sample"):
        assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 2.0, abs_tol=abs_tol)


def test_geometric_mean_speed_ratio_correct_only_no_timed_correct_sample():

    is_correct = [1,1,0]
    baseline_speed = [0.2, 0.3, 0.1]
    actual_speed = [-1.0, -1.0, -1.0]
    n = 3

    with pytest.warns(UserWarning, match="2 correct sample"):
        assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 0, abs_tol=abs_tol)


def test_geometric_mean_speed_ratio_correct_only_untimed_incorrect_samples():

    '''
    Incorrect samples with runtime -1.0 are ignored as before, with no warning
    '''

    is_correct = [1,0,1]
    baseline_speed = [0.2, 0.3, 0.5]
    actual_speed = [0.1, -1.0, 0.5]
    n = 3

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 1.414213562, abs_tol=abs_tol)


def test_geometric_mean_speed_ratio_correct_only_slower_and_incorrect_timed_samples():

    '''
    A correct sample slower than baseline still counts, and an incorrect sample
    with a positive runtime still does not
    '''

    is_correct = [1,0,1,1]
    baseline_speed = [0.2, 0.3, 0.25, 0.4]
    actual_speed = [0.1, 0.3, 0.5, 0.1]
    n = 4

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isclose(geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n), 1.587401052, abs_tol=abs_tol)


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

