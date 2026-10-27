"""Output comparison for kernel correctness.

`torch.allclose(reference, candidate, atol=tolerance, rtol=tolerance)` scales
the relative term by the candidate. An all-zero candidate therefore passes
whenever every reference element is within `atol`, even if the reference is a
real non-zero tensor (a long softmax, a norm of a huge tensor, a min over
many uniforms). A constant equal to the mean can pass the same way when the
reference spread is smaller than `atol + rtol * |mean|`.
"""

import math
from typing import Optional

import torch

# Tensor-level limits used only when allclose's absolute term hides the
# reference. These match the scale-invariant checks proposed for the
# zero-output failures: Frobenius catches a distributed scale error, max
# catches one far element. Denominators are the reference's own peak and
# norm, not a per-element floor.
_RELATIVE_MAX_ERROR_LIMIT = 5e-2
_RELATIVE_FROBENIUS_ERROR_LIMIT = 2e-2

# A candidate whose RMS error exceeds half the reference std is closer to a
# constant than to the reference. A std below one tenth of the reference has
# collapsed. Both apply only when that std is real signal, not a few ulps.
_SPREAD_ERROR_RMS_LIMIT = 0.5
_SPREAD_COLLAPSE_LIMIT = 0.1
_SPREAD_SIGNAL_ULPS = 8.0

_MAGNITUDE_REJECTION = "Trivial output rejected: reference magnitude not matched"
_SPREAD_REJECTION = "Trivial output rejected: reference spread collapsed"
_OUTPUT_MISMATCH = "Output mismatch"


def output_mismatch_reason(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerance: float,
) -> Optional[str]:
    """Why `candidate` is not correct, or None when it matches `reference`.

    The primary test is allclose with `atol == rtol == tolerance`. Two extra
    rejections apply only where that test cannot see a trivial stand-in:

    * If the reference lies inside `atol` of zero, require the error to be
      small relative to the reference peak and Frobenius norm.
    * If a constant equal to the mean would itself pass allclose, and the
      reference std is more than a few ulps of the reference dtype, reject a
      candidate that drops that spread.

    Matching kernels, exact-zero references, and outputs whose spread already
    exceeds the allclose band are decided by allclose alone.
    """
    if reference.shape != candidate.shape:
        return _OUTPUT_MISMATCH
    if not torch.allclose(reference, candidate, atol=tolerance, rtol=tolerance):
        return _OUTPUT_MISMATCH
    if reference.numel() == 0 or not reference.dtype.is_floating_point:
        return None
    if _reference_within_zero_atol(reference, tolerance) and _relative_magnitude_rejected(
        reference, candidate
    ):
        return _MAGNITUDE_REJECTION
    if _spread_collapsed(reference, candidate, tolerance):
        return _SPREAD_REJECTION
    return None


def _reference_within_zero_atol(reference: torch.Tensor, tolerance: float) -> bool:
    """True when zeros pass allclose against `reference`.

    allclose(reference, zeros) holds iff every |reference| <= atol, because
    the rtol term multiplies the candidate.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        return False
    peak = reference.detach().abs().amax()
    if not torch.isfinite(peak):
        return False
    return float(peak.item()) <= tolerance


def _relative_magnitude_rejected(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    ref = reference.detach()
    got = candidate.detach()
    ref_peak_value = float(ref.abs().amax().item())
    if ref_peak_value == 0.0 or not math.isfinite(ref_peak_value):
        return False
    # Stay in the reference dtype. A float32 copy of a full level-1 activation
    # is several GB, and both limits below are far above a dtype epsilon.
    diff = got - ref
    rel_max = float(diff.abs().amax().item()) / ref_peak_value
    if not math.isfinite(rel_max) or rel_max > _RELATIVE_MAX_ERROR_LIMIT:
        return True
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    if ref_norm == 0.0 or not math.isfinite(ref_norm):
        return False
    rel_fro = float(torch.linalg.vector_norm(diff).item()) / ref_norm
    return (not math.isfinite(rel_fro)) or rel_fro > _RELATIVE_FROBENIUS_ERROR_LIMIT


def _spread_collapsed(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerance: float,
) -> bool:
    if reference.numel() < 2 or not _mean_constant_passes_allclose(reference, tolerance):
        return False
    ref = reference.detach()
    got = candidate.detach()
    ref_std = float(ref.std(correction=0).item())
    if not math.isfinite(ref_std) or ref_std <= _spread_signal_floor(reference):
        return False
    error_rms = float((got - ref).square().mean().sqrt().item())
    if not math.isfinite(error_rms) or error_rms > _SPREAD_ERROR_RMS_LIMIT * ref_std:
        return True
    got_std = float(got.std(correction=0).item())
    return (not math.isfinite(got_std)) or got_std < _SPREAD_COLLAPSE_LIMIT * ref_std


def _mean_constant_passes_allclose(reference: torch.Tensor, tolerance: float) -> bool:
    ref = reference.detach()
    # Accumulate in float32 so a bf16 reduction does not materialize a
    # full-size upcast of the activation.
    mean = ref.mean(dtype=torch.float32).to(dtype=ref.dtype)
    return bool(torch.allclose(ref, mean.expand_as(ref), atol=tolerance, rtol=tolerance))


def _spread_signal_floor(reference: torch.Tensor) -> float:
    peak = reference.detach().abs().amax()
    if not torch.isfinite(peak):
        return float("inf")
    eps = float(torch.finfo(reference.dtype).eps)
    return _SPREAD_SIGNAL_ULPS * eps * float(peak.item())
