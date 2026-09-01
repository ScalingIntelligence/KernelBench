"""HyperConnection operators.

The main MHC pre path is organized as:

1. prenorm GEMM + row square-sum;
2. fused MHC pre stage, which applies RMS scaling, split/sinkhorn, and
   produces the layer input.

The returned tensor order follows the SGLang MHC pre convention:
``(post_mix, comb_mix, layer_input)``.
"""

from typing import Optional, Tuple

import torch

from mate.api_logging import mate_api
from mate.hyperconnection_kernels.tilelang import mhc_pre as mhc_pre_tilelang

MHC_PRENORM_BACKEND_DEEPGEMM = "deepgemm"
MHC_BIG_FUSE_BACKEND_TILELANG = "tilelang"

__all__ = [
    "MHC_BIG_FUSE_BACKEND_TILELANG",
    "MHC_PRENORM_BACKEND_DEEPGEMM",
    "mhc_pre",
    "mhc_pre_big_fuse",
    "mhc_prenorm_gemm_sqrsum",
]


def _check_same_device(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"Expected {name} to be on device {reference.device}, got {tensor.device}."
            )


def _validate_mhc_inputs(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
) -> Tuple[Tuple[int, ...], int, int, int]:
    if residual.dim() < 2:
        raise ValueError(
            f"residual must have shape [..., mhc_mult, hidden_size], got {residual.shape}."
        )
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be bfloat16, got {residual.dtype}.")
    if hc_fn.dtype != torch.float32:
        raise ValueError(f"hc_fn must be float32, got {hc_fn.dtype}.")
    if mhc_scale.dtype != torch.float32 or mhc_base.dtype != torch.float32:
        raise ValueError("mhc_scale and mhc_base must be float32.")

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    if hc_fn.shape != (mhc_mult3, mhc_hidden_size):
        raise ValueError(
            f"hc_fn must have shape {(mhc_mult3, mhc_hidden_size)}, "
            f"got {tuple(hc_fn.shape)}."
        )
    if mhc_scale.shape != (3,):
        raise ValueError(
            f"mhc_scale must have shape [3], got {tuple(mhc_scale.shape)}."
        )
    if mhc_base.shape != (mhc_mult3,):
        raise ValueError(
            f"mhc_base must have shape [{mhc_mult3}], got {tuple(mhc_base.shape)}."
        )
    _check_same_device(residual, hc_fn=hc_fn, mhc_scale=mhc_scale, mhc_base=mhc_base)
    outer_shape = tuple(residual.shape[:-2])
    num_tokens = 1
    for dim in outer_shape:
        num_tokens *= dim
    return outer_shape, num_tokens, mhc_mult, hidden_size


def _normalize_prenorm_partials(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gemm_out_mul.dim() == 2:
        gemm_out_mul = gemm_out_mul.unsqueeze(0)
    if gemm_out_sqrsum.dim() == 1:
        gemm_out_sqrsum = gemm_out_sqrsum.unsqueeze(0)
    return gemm_out_mul, gemm_out_sqrsum


def _default_split_k(num_tokens: int) -> int:
    return 32 if num_tokens <= 64 else 16


@mate_api
def mhc_prenorm_gemm_sqrsum(
    residual_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    *,
    backend: str = MHC_PRENORM_BACKEND_DEEPGEMM,
    split_k: Optional[int] = None,
    return_partials: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Run MHC prenorm GEMM and row square-sum.

    Parameters
    ----------
    residual_flat:
        Tensor with shape ``(M, mhc_mult, hidden_size)`` or ``(M, K)`` and dtype
        ``torch.bfloat16``.
    hc_fn:
        Weight tensor with shape ``(N, K)`` and dtype ``torch.float32``.
    backend:
        Prenorm backend selector.
    split_k:
        Prenorm backend split-K factor. If omitted, uses ``32`` for decode-like
        ``M <= 64`` and ``16`` otherwise.
    return_partials:
        When true, return tensors with leading split dimension ``[S, M, ...]``.
    """
    backend = backend.strip().lower()
    if residual_flat.dtype != torch.bfloat16:
        raise ValueError(f"residual_flat must be bfloat16, got {residual_flat.dtype}.")
    if hc_fn.dtype != torch.float32:
        raise ValueError(f"hc_fn must be float32, got {hc_fn.dtype}.")

    x_flat = residual_flat.reshape(residual_flat.shape[0], -1)
    num_tokens, hc_hidden_size = x_flat.shape
    mhc_mult3 = hc_fn.shape[0]
    if hc_fn.shape != (mhc_mult3, hc_hidden_size):
        raise ValueError(
            f"hc_fn must have shape [N, {hc_hidden_size}], got {tuple(hc_fn.shape)}."
        )
    _check_same_device(x_flat, hc_fn=hc_fn)

    if backend != MHC_PRENORM_BACKEND_DEEPGEMM:
        raise ValueError(
            f"Unsupported MHC prenorm backend={backend!r}; expected 'deepgemm'."
        )

    import mate.deep_gemm as deep_gemm

    num_splits = _default_split_k(num_tokens) if split_k is None else int(split_k)
    if num_splits <= 1:
        d_out = torch.empty(
            (num_tokens, mhc_mult3), dtype=torch.float32, device=x_flat.device
        )
        s_out = torch.empty((num_tokens,), dtype=torch.float32, device=x_flat.device)
        deep_gemm.tf32_hc_prenorm_gemm(
            x_flat,
            hc_fn.contiguous(),
            d_out,
            s_out,
            num_splits=None if num_splits <= 0 else num_splits,
        )
        if return_partials:
            return d_out.unsqueeze(0), s_out.unsqueeze(0)
        return d_out, s_out

    d_part = torch.empty(
        (num_splits, num_tokens, mhc_mult3),
        dtype=torch.float32,
        device=x_flat.device,
    )
    s_part = torch.empty(
        (num_splits, num_tokens), dtype=torch.float32, device=x_flat.device
    )
    deep_gemm.tf32_hc_prenorm_gemm(
        x_flat,
        hc_fn.contiguous(),
        d_part,
        s_part,
        num_splits=num_splits,
    )
    if return_partials:
        return d_part, s_part
    return d_part.sum(dim=0), s_part.sum(dim=0)


@mate_api
def mhc_pre_big_fuse(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual_flat: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    backend: str = MHC_BIG_FUSE_BACKEND_TILELANG,
    threads: int = 128,
    hidden_block: int = 512,
    pass_config: str = "safe",
    compile_profile: Optional[str] = None,
    post_mix: Optional[torch.Tensor] = None,
    comb_mix: Optional[torch.Tensor] = None,
    layer_input: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Run the fused MHC pre stage.

    This is the second stage of the DeepSeek V4 MHC pre path. It consumes the
    prenorm GEMM output ``mixes`` and row square-sum produced by
    ``mhc_prenorm_gemm_sqrsum`` or ``mate.deep_gemm.tf32_hc_prenorm_gemm``, applies
    RMS scaling, computes the MHC pre/post/comb branches, runs Sinkhorn
    normalization for the comb branch, and writes the fused layer input.

    ``gemm_out_mul`` and ``gemm_out_sqrsum`` may be either final outputs with
    shapes ``[num_tokens, mhc_mult * (2 + mhc_mult)]`` / ``[num_tokens]`` or
    split-K partials with shapes
    ``[num_splits, num_tokens, mhc_mult * (2 + mhc_mult)]`` /
    ``[num_splits, num_tokens]``. Passing split-K partials avoids a separate
    reduction before the fused stage.

    Parameters
    ----------
    gemm_out_mul:
        Float32 prenorm GEMM output or split-K partials.
    gemm_out_sqrsum:
        Float32 row square-sum output or split-K partials.
    mhc_scale:
        Float32 scale tensor with shape ``[3]`` for pre/post/comb branches.
    mhc_base:
        Float32 bias tensor with shape ``[mhc_mult * (2 + mhc_mult)]``.
    residual_flat:
        Contiguous bfloat16 tensor with shape
        ``[num_tokens, mhc_mult, hidden_size]``.
    rms_eps:
        Epsilon added before RMS inverse square root.
    mhc_pre_eps:
        Epsilon added to the sigmoid pre branch.
    mhc_sinkhorn_eps:
        Epsilon used by Sinkhorn normalization.
    mhc_post_mult_value:
        Multiplicative scale applied to the post branch.
    sinkhorn_repeat:
        Number of Sinkhorn normalization iterations.
    backend:
        Fused-stage backend selector.
    threads:
        Kernel thread count.
    hidden_block:
        Hidden dimension tile size used by the layer-input reduction.
    pass_config:
        Backend pass configuration name.
    compile_profile:
        Optional MUSA compiler flag profile.
    post_mix:
        Optional contiguous float32 output buffer with shape
        ``[num_tokens, mhc_mult]``.
    comb_mix:
        Optional contiguous float32 output buffer with shape
        ``[num_tokens, mhc_mult * mhc_mult]``.
    layer_input:
        Optional contiguous bfloat16 output buffer with shape
        ``[num_tokens, hidden_size]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(post_mix, comb_mix, layer_input)``.

    ``residual_flat`` must have contiguous shape
    ``[num_tokens, mhc_mult, hidden_size]``. Prenorm inputs and optional output
    buffers must also be contiguous; this avoids hidden copies and keeps the
    indexing path specialized for the hot serving case. The returned
    tensors are ``(post_mix, comb_mix, layer_input)`` with shapes
    ``[num_tokens, mhc_mult]``, ``[num_tokens, mhc_mult * mhc_mult]``, and
    ``[num_tokens, hidden_size]``. Optional output buffers may be provided to
    avoid allocations in hot serving paths or graph-capture paths.
    """
    backend = backend.strip().lower()
    if residual_flat.dim() != 3:
        raise ValueError(
            "residual_flat must have shape [num_tokens, mhc_mult, hidden_size]."
        )
    if sinkhorn_repeat <= 0:
        raise ValueError(f"sinkhorn_repeat must be > 0, got {sinkhorn_repeat}.")

    if backend != MHC_BIG_FUSE_BACKEND_TILELANG:
        raise ValueError(
            f"Unsupported MHC big-fuse backend={backend!r}; expected 'tilelang'."
        )

    mhc_mult = residual_flat.shape[1]
    hidden_size = residual_flat.shape[2]
    num_tokens = residual_flat.shape[0]
    if post_mix is None:
        post_mix = torch.empty(
            (num_tokens, mhc_mult), dtype=torch.float32, device=residual_flat.device
        )
    if comb_mix is None:
        comb_mix = torch.empty(
            (num_tokens, mhc_mult * mhc_mult),
            dtype=torch.float32,
            device=residual_flat.device,
        )
    if layer_input is None:
        layer_input = torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=residual_flat.device
        )
    mhc_pre_tilelang.run_mhc_pre_big_fuse(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
        compile_profile=compile_profile,
    )
    return post_mix, comb_mix, layer_input


@mate_api
def mhc_pre(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    split_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Run MHC pre end to end.

    This convenience wrapper runs the prenorm stage followed by the fused MHC
    pre stage. It intentionally exposes only model-level semantics; backend
    tuning knobs remain on ``mhc_prenorm_gemm_sqrsum`` and
    ``mhc_pre_big_fuse`` for callers that need explicit control.

    Parameters
    ----------
    residual:
        Bfloat16 input tensor with shape ``[..., mhc_mult, hidden_size]``.
    hc_fn:
        Float32 prenorm GEMM weight with shape
        ``[mhc_mult * (2 + mhc_mult), mhc_mult * hidden_size]``.
    mhc_scale:
        Float32 scale tensor with shape ``[3]`` for pre/post/comb branches.
    mhc_base:
        Float32 bias tensor with shape ``[mhc_mult * (2 + mhc_mult)]``.
    rms_eps:
        Epsilon added before RMS inverse square root.
    mhc_pre_eps:
        Epsilon added to the sigmoid pre branch.
    mhc_sinkhorn_eps:
        Epsilon used by Sinkhorn normalization.
    mhc_post_mult_value:
        Multiplicative scale applied to the post branch.
    sinkhorn_repeat:
        Number of Sinkhorn normalization iterations.
    split_k:
        Optional prenorm split-K factor. If omitted, uses the decode/prefill
        default selected by ``mhc_prenorm_gemm_sqrsum``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(post_mix, comb_mix, layer_input)`` with shapes
        ``[..., mhc_mult, 1]``, ``[..., mhc_mult, mhc_mult]``, and
        ``[..., hidden_size]``.
    """
    outer_shape, num_tokens, mhc_mult, hidden_size = _validate_mhc_inputs(
        residual, hc_fn, mhc_scale, mhc_base
    )
    residual_flat = residual.reshape(num_tokens, mhc_mult, hidden_size)
    gemm_out_mul, gemm_out_sqrsum = mhc_prenorm_gemm_sqrsum(
        residual_flat,
        hc_fn,
        split_k=split_k,
        return_partials=True,
    )
    post_mix, comb_mix, layer_input = mhc_pre_big_fuse(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    return (
        post_mix.reshape(*outer_shape, mhc_mult, 1),
        comb_mix.reshape(*outer_shape, mhc_mult, mhc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )
