import torch

from kernelbench.eval import build_perf_input_pool
from kernelbench.timing import get_trial_args

"""
Tests for rotating timed trials over distinct input sets (issue #171).
These run on CPU, no GPU needed.

Run with pytest
pytest src/kernelbench/unit_tests/test_perf_input_rotation.py
"""

N = 64
SET_BYTES = 2 * N * N * 4  # two fp32 NxN tensors


def get_inputs():
    return [torch.rand(N, N), torch.rand(N, N)]


def test_get_trial_args_without_pool_returns_args():
    args = [torch.zeros(1)]
    for trial in range(5):
        assert get_trial_args(args, None, trial) is args
        assert get_trial_args(args, [], trial) is args


def test_get_trial_args_rotates_through_pool():
    pool = [[torch.zeros(1)], [torch.zeros(1)], [torch.zeros(1)]]
    picked = [get_trial_args(None, pool, trial) for trial in range(7)]
    assert picked == [pool[0], pool[1], pool[2], pool[0], pool[1], pool[2], pool[0]]
    # consecutive calls never share input objects
    for prev, cur in zip(picked, picked[1:]):
        assert prev[0] is not cur[0]


def test_pool_sets_are_distinct():
    pool = build_perf_input_pool(get_inputs, 5, 1 << 30, seed=42, device="cpu")
    assert len(pool) == 5
    ptrs = {x.data_ptr() for s in pool for x in s}
    assert len(ptrs) == 10, "every tensor in the pool should have its own storage"
    for a, b in zip(pool, pool[1:]):
        assert not torch.equal(a[0], b[0]), "sets should have different values"


def test_pool_is_deterministic():
    pool_a = build_perf_input_pool(get_inputs, 3, 1 << 30, seed=42, device="cpu")
    pool_b = build_perf_input_pool(get_inputs, 3, 1 << 30, seed=42, device="cpu")
    for sa, sb in zip(pool_a, pool_b):
        for xa, xb in zip(sa, sb):
            assert torch.equal(xa, xb)


def test_pool_respects_memory_cap():
    pool = build_perf_input_pool(get_inputs, 100, 4 * SET_BYTES, seed=42, device="cpu")
    assert len(pool) == 4


def test_pool_has_at_least_two_sets():
    # even when a single set is over the cap, consecutive trials must not share inputs
    pool = build_perf_input_pool(get_inputs, 100, SET_BYTES // 2, seed=42, device="cpu")
    assert len(pool) == 2


def test_pool_applies_precision_and_keeps_non_tensors():
    def get_mixed_inputs():
        return [torch.rand(N, N), 3, "label"]

    pool = build_perf_input_pool(get_mixed_inputs, 2, 1 << 30, seed=42, device="cpu", precision=torch.float16)
    for s in pool:
        assert s[0].dtype == torch.float16
        assert s[1:] == [3, "label"]
