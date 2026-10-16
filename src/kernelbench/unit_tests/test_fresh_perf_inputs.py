import torch

from kernelbench import timing
from kernelbench.eval import make_fresh_inputs

"""
Tests for drawing new inputs for every timed call (issue #171).
These run on CPU, no GPU needed.

Run with pytest
pytest src/kernelbench/unit_tests/test_fresh_perf_inputs.py
"""


def get_inputs():
    return [torch.rand(64, 64), torch.rand(64, 64)]


def test_each_call_draws_new_values():
    fresh_args = make_fresh_inputs(get_inputs, seed=42, device="cpu")
    sets = [fresh_args(call) for call in range(5)]
    for a, b in zip(sets, sets[1:]):
        assert not torch.equal(a[0], b[0]), "consecutive calls should get different values"
        assert a[0] is not b[0]


def test_calls_are_reproducible():
    fresh_a = make_fresh_inputs(get_inputs, seed=42, device="cpu")
    fresh_b = make_fresh_inputs(get_inputs, seed=42, device="cpu")
    for call in range(3):
        for xa, xb in zip(fresh_a(call), fresh_b(call)):
            assert torch.equal(xa, xb)


def test_calls_differ_from_the_warmup_inputs():
    # the eval warms up on the inputs drawn with seed_num itself
    torch.manual_seed(42)
    warmup = get_inputs()
    fresh_args = make_fresh_inputs(get_inputs, seed=42, device="cpu")
    for call in range(5):
        assert not torch.equal(fresh_args(call)[0], warmup[0])


def test_precision_and_non_tensors():
    def get_mixed_inputs():
        return [torch.rand(8, 8), 3, "label"]

    out = make_fresh_inputs(get_mixed_inputs, seed=42, device="cpu", precision=torch.float16)(0)
    assert out[0].dtype == torch.float16
    assert out[1:] == [3, "label"]


def test_timing_loop_gives_each_timed_call_new_inputs(monkeypatch):
    # run the real host_time loop on CPU with the CUDA calls stubbed out
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "cpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(timing, "clear_l2_cache", lambda *a, **k: None)

    seen = []
    def kernel(a, b):
        seen.append(a)
        return a @ b

    args = get_inputs()
    num_warmup, num_trials, discard_first = 3, 10, 1
    timing.time_execution_with_host_time(
        kernel, args, num_warmup=num_warmup, num_trials=num_trials, discard_first=discard_first,
        verbose=False, device="cpu", fresh_args=make_fresh_inputs(get_inputs, seed=42, device="cpu"),
    )

    warmup_calls, timed_calls = seen[:num_warmup], seen[num_warmup:]
    assert all(a is args[0] for a in warmup_calls), "warmup should use the original inputs"
    assert len(timed_calls) == num_trials + discard_first
    for i, a in enumerate(timed_calls):
        assert a is not args[0]
        for b in timed_calls[i + 1:]:
            assert not torch.equal(a, b), "no two timed calls should share input values"


def test_timing_loop_without_fresh_args_reuses_inputs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "cpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(timing, "clear_l2_cache", lambda *a, **k: None)

    seen = []
    args = get_inputs()
    timing.time_execution_with_host_time(
        lambda a, b: seen.append(a), args, num_warmup=2, num_trials=5, verbose=False, device="cpu",
    )
    assert all(a is args[0] for a in seen)


def test_falls_back_to_cpu_when_get_inputs_cannot_run_on_device():
    def get_cpu_only_inputs():
        if torch.get_default_device().type != "cpu":
            raise RuntimeError("this get_inputs only works on the CPU")
        return [torch.rand(8, 8)]

    # "meta" stands in for a GPU here: drawing there fails, so it should fall back to the CPU
    out = make_fresh_inputs(get_cpu_only_inputs, seed=42, device="meta")(0)
    assert out[0].shape == (8, 8)
