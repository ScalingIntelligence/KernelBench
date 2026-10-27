"""Regression tests for trivial outputs that pass torch.allclose.

CPU only. Nothing here allocates the shipped level-1 tensors (those inputs
are several GB). Structural cases call the real problem modules on smaller
inputs of the same op. The eval-loop test stubs torch.cuda.synchronize
because that call requires a CUDA build; the comparison itself does not.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kernelbench.correctness import output_mismatch_reason

REPO_ROOT = Path(__file__).resolve().parents[3]
LEVEL1 = REPO_ROOT / "KernelBench" / "level1"

_MAGNITUDE = "Trivial output rejected: reference magnitude not matched"
_SPREAD = "Trivial output rejected: reference spread collapsed"
_MISMATCH = "Output mismatch"


def _load_problem(filename):
    path = LEVEL1 / filename
    spec = importlib.util.spec_from_file_location(filename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_scale_outputs_keep_allclose_behavior():
    torch.manual_seed(0)
    reference = torch.randn(128, 64)
    within_atol = reference + 1e-5
    assert output_mismatch_reason(reference, within_atol, 1e-4) is None
    assert output_mismatch_reason(reference, reference + 1e-3, 1e-4) == _MISMATCH
    assert output_mismatch_reason(reference, torch.zeros_like(reference), 1e-4) == _MISMATCH


def test_exact_zero_reference_still_matches_zero_and_atol_noise():
    reference = torch.zeros(32, 32)
    assert output_mismatch_reason(reference, reference.clone(), 1e-4) is None
    assert output_mismatch_reason(reference, torch.full_like(reference, 1e-6), 1e-4) is None
    assert output_mismatch_reason(reference, torch.ones_like(reference), 1e-4) == _MISMATCH


def test_near_zero_entries_inside_a_large_tensor_are_not_relative_failures():
    """A fixed per-element denominator floor rejects this. The tensor norm must not."""
    reference = torch.zeros(10000)
    reference[0] = 1.0
    candidate = reference.clone()
    candidate[1] = 1e-8
    assert torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4)
    assert output_mismatch_reason(reference, candidate, 1e-4) is None


def test_empty_and_nan_outputs():
    empty = torch.empty(0, 4)
    assert output_mismatch_reason(empty, empty.clone(), 1e-4) is None
    reference = torch.tensor([1.0, 2.0])
    candidate = torch.tensor([1.0, math.nan])
    assert output_mismatch_reason(reference, candidate, 1e-4) == _MISMATCH
    assert output_mismatch_reason(reference, reference[:1], 1e-4) == _MISMATCH


def test_softmax_bound_for_the_shipped_shape_is_inside_both_atols():
    """Any torch.rand row of length 393216 has softmax entries < e / 393216.

    torch.rand is in [0, 1), so this does not depend on the seed.
    """
    assert math.e / 393216 < 1e-4
    assert math.e / 393216 < 1e-2


def test_real_softmax_module_rejects_zeros_and_a_rescaled_output():
    module = _load_problem("23_Softmax.py")
    model = module.Model(*module.get_init_inputs()).eval()
    torch.manual_seed(0)
    # Row length is still enough that max(softmax) < e/50000 < 1e-4.
    logits = torch.rand(2, 50000)
    with torch.no_grad():
        reference = model(logits)
    assert float(reference.abs().max()) <= math.e / 50000
    zeros = torch.zeros_like(reference)
    for tolerance in (1e-4, 1e-2):
        assert torch.allclose(reference, zeros, atol=tolerance, rtol=tolerance)
        assert output_mismatch_reason(reference, zeros, tolerance) == _MAGNITUDE
        assert output_mismatch_reason(reference, reference.clone(), tolerance) is None

    # Uniformly 50x too large, but every element is still inside bf16 atol.
    rescaled = reference * 50
    assert torch.allclose(reference, rescaled, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, rescaled, 1e-2) == _MAGNITUDE
    assert output_mismatch_reason(reference, reference * 1.01, 1e-2) is None


def test_real_frobenius_and_l2_modules_reject_zeros_inside_bf16_atol():
    torch.manual_seed(0)
    frobenius = _load_problem("37_FrobeniusNorm_.py")
    model = frobenius.Model(*frobenius.get_init_inputs()).eval()
    with torch.no_grad():
        reference = model(torch.rand(4, 4, 64, 64))
    assert float(reference.abs().max()) < 1e-2
    zeros = torch.zeros_like(reference)
    assert torch.allclose(reference, zeros, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, zeros, 1e-2) == _MAGNITUDE
    assert output_mismatch_reason(reference, reference.clone(), 1e-2) is None

    l2 = _load_problem("39_L2Norm_.py")
    model = l2.Model(*l2.get_init_inputs()).eval()
    with torch.no_grad():
        reference = model(torch.rand(8, 100000))
    assert 1e-4 < float(reference.abs().max()) < 1e-2
    zeros = torch.zeros_like(reference)
    assert torch.allclose(reference, zeros, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(reference, zeros, atol=1e-4, rtol=1e-4)
    assert output_mismatch_reason(reference, zeros, 1e-2) == _MAGNITUDE
    assert output_mismatch_reason(reference, zeros, 1e-4) == _MISMATCH
    assert output_mismatch_reason(reference, reference.clone(), 1e-2) is None


def test_real_min_reduction_rejects_zeros_only_at_bf16_atol():
    module = _load_problem("53_Min_reduction_over_a_dimension.py")
    model = module.Model(*module.get_init_inputs()).eval()
    torch.manual_seed(0)
    # Min over the shipped reduction length (4096); the extra trailing dim
    # does not change the per-row minimum.
    with torch.no_grad():
        reference = model(torch.rand(128, 4096))
    assert 1e-4 < float(reference.abs().max()) < 1e-2
    zeros = torch.zeros_like(reference)
    assert torch.allclose(reference, zeros, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, zeros, 1e-2) == _MAGNITUDE
    assert output_mismatch_reason(reference, zeros, 1e-4) == _MISMATCH
    assert output_mismatch_reason(reference, reference.clone(), 1e-2) is None


def test_bf16_softmax_zeros_fail_and_matching_kernel_passes():
    torch.manual_seed(0)
    logits = torch.rand(4, 20000, dtype=torch.bfloat16)
    reference = torch.softmax(logits, dim=1)
    zeros = torch.zeros_like(reference)
    assert reference.dtype == torch.bfloat16
    assert float(reference.float().abs().max()) < 1e-2
    assert torch.allclose(reference, zeros, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, zeros, 1e-2) == _MAGNITUDE
    assert output_mismatch_reason(reference, reference.clone(), 1e-2) is None


def test_one_far_element_fails_relative_max_when_allclose_passes():
    reference = torch.full((1000,), 1e-5)
    candidate = reference.clone()
    candidate[0] = 1e-3
    assert torch.allclose(reference, candidate, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, candidate, 1e-2) == _MAGNITUDE


def test_large_mean_small_spread_constant_is_rejected_when_allclose_passes():
    """level2/56 shape of failure: mean ~1e4, range a few dozen.

    At fp32 tolerance the constant is already outside allclose. At bf16
    tolerance allclose and both magnitude ratios accept it; the spread does not.
    """
    reference = torch.linspace(16364.2 - 19.3, 16364.2 + 19.3, 128)
    constant = torch.full_like(reference, float(reference.mean()))
    diff = constant - reference
    assert float(diff.norm() / reference.norm()) < 2e-2
    assert float(diff.abs().max() / reference.abs().max()) < 5e-2
    assert output_mismatch_reason(reference, constant, 1e-4) == _MISMATCH
    assert torch.allclose(reference, constant, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, constant, 1e-2) == _SPREAD
    assert output_mismatch_reason(reference, reference.clone(), 1e-2) is None
    # Noise well under the spread still matches.
    assert output_mismatch_reason(reference, reference + 1e-3, 1e-2) is None


def test_bf16_quantization_of_a_small_range_is_not_a_spread_failure():
    # Adjacent bf16 bins around 16384. The gap is inside the bf16 allclose
    # band and inside a few ulps, so a constant is still a match.
    reference = torch.linspace(16344.9, 16383.5, 128).to(torch.bfloat16)
    assert reference.unique().numel() > 1
    constant = torch.full_like(reference, float(reference.float().mean()))
    assert torch.allclose(reference, constant, atol=1e-2, rtol=1e-2)
    assert output_mismatch_reason(reference, constant, 1e-2) is None

    collapsed = torch.linspace(16355.0, 16380.0, 64).to(torch.bfloat16)
    assert collapsed.unique().numel() == 1
    matching = torch.full_like(collapsed, float(collapsed.reshape(-1)[0]))
    assert output_mismatch_reason(collapsed, matching, 1e-2) is None


class _Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class _Zeros(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def _cpu_inputs():
    return [torch.rand(2, 50000)]


def test_run_and_check_correctness_rejects_zero_kernel_without_cuda(monkeypatch):
    """The eval loop calls torch.cuda.synchronize even for a CPU device.

    This build has no CUDA, so synchronize is stubbed. The models run on CPU.
    """
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *args, **kwargs: None)

    from kernelbench.eval import run_and_check_correctness

    zeros = run_and_check_correctness(
        _Softmax(),
        _Zeros(),
        _cpu_inputs,
        {},
        num_correct_trials=2,
        verbose=False,
        device=torch.device("cpu"),
        precision=torch.float32,
    )
    assert zeros.correctness is False
    assert zeros.metadata["correctness_issue"] == _MAGNITUDE
    assert zeros.metadata["correctness_trials"] == "(0 / 2)"

    match = run_and_check_correctness(
        _Softmax(),
        _Softmax(),
        _cpu_inputs,
        {},
        num_correct_trials=2,
        verbose=False,
        device=torch.device("cpu"),
        precision=torch.float32,
    )
    assert match.correctness is True
    assert "correctness_issue" not in match.metadata
