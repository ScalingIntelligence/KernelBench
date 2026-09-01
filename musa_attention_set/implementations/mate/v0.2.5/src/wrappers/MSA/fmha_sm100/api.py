from __future__ import annotations

from mate import msa_interface as _msa

__all__ = [
    "_fmha_sm100",
    "_fmha_sm100_plan",
    "fmha_sm100",
    "fmha_sm100_plan",
    "sparse_topk_select",
]


def fmha_sm100_plan(*args, **kwargs):
    """MSA-compatible plan wrapper backed by ``mate.msa_interface``."""

    return _msa._msa_plan_from_lengths(*args, **kwargs)


def fmha_sm100(*args, **kwargs):
    """MSA-compatible forward wrapper backed by ``mate.msa_interface``."""

    return _msa.msa(*args, **kwargs)


def sparse_topk_select(*args, **kwargs):
    """Forward sparse top-k selection calls to MATE when implemented."""

    return _msa.sparse_topk_select(*args, **kwargs)


_fmha_sm100_plan = fmha_sm100_plan
_fmha_sm100 = fmha_sm100
