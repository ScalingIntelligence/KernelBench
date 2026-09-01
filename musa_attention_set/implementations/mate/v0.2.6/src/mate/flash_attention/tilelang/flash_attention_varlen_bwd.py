# ruff: noqa
# type: ignore
import torch

from ...execution_context import raise_complete_if_dry_run
from ._flash_attention_bwd_common import (
    _check_attention_strides,
    _contiguous_cosize_bytes,
    _jit_for_index_type_promotion,
    _needs_index_type_promotion,
    _tensor_cosize_bytes,
)
from ._flash_attention_bwd_post import (
    ceil_div,
    compute_delta_ws,
    pack_dq_from_accum_ws,
    reduce_kv_grads_ws,
    singleton_k_dv_ws,
    to_tilelang_dtype,
)
from ._flash_attention_bwd_split import flashattn_bwd_ws_split
from ._flash_attention_bwd_split_dkdv import flashattn_bwd_ws_split_dkdv
from ._flash_attention_bwd_split_dq import flashattn_bwd_ws_split_dq
from ._flash_attention_bwd_unsplit import flashattn_bwd_ws_unsplit
from ._flash_attention_bwd_unsplit_dkdv import flashattn_bwd_ws_unsplit_dkdv
from ._flash_attention_bwd_unsplit_dq import flashattn_bwd_ws_unsplit_dq


_THREADS = 640
_SPLIT_BLOCK_M = 64
_SPLIT_BLOCK_N = 64
_UNSPLIT_BLOCK_M = 64
_UNSPLIT_BLOCK_N = 128
_SPLIT_DKDV_BLOCK_M = 64
_SPLIT_DKDV_BLOCK_N = 128
_SPLIT_DQ_BLOCK_M = 128
_SPLIT_DQ_BLOCK_N = 64
_UNSPLIT_DKDV_BLOCK_M = 64
_UNSPLIT_DKDV_BLOCK_N = 128
_UNSPLIT_DQ_BLOCK_M = 128
_UNSPLIT_DQ_BLOCK_N = 64
_LOG2E = 1.44269504


def _select_bwd_plan(dim, deterministic, heads_q_eq_heads_kv=True):
    if dim == 256:
        return "split_separate" if deterministic else "split"
    if dim == 128:
        if deterministic or not heads_q_eq_heads_kv:
            return "unsplit_separate"
        return "unsplit"
    raise NotImplementedError(
        f"TileLang FlashAttention backward currently supports dim 128 or 256, got {dim}"
    )


def _select_bwd_feature_plan(
    plan,
    use_static_dims,
    is_causal,
    has_window_left,
    has_window_right,
    has_softcap,
):
    if plan == "unsplit" and (
        not use_static_dims or has_softcap or has_window_left or has_window_right
    ):
        return "split"
    return plan


def _use_static_dims_for_dkdv(plan, use_static_dims, is_varlen, heads_q_eq_heads_kv):
    return use_static_dims and not (
        plan == "unsplit_separate" and not is_varlen and not heads_q_eq_heads_kv
    )


def _should_recompute_delta(plan, max_seqlen_k, block_N):
    return plan == "split" and max_seqlen_k <= block_N


def _select_kernel_dim(dim):
    if dim < 8:
        raise NotImplementedError(
            f"TileLang FlashAttention backward requires head dim >= 8, got {dim}"
        )
    if dim <= 128:
        return 128
    if dim <= 256:
        return 256
    raise NotImplementedError(
        f"TileLang FlashAttention backward currently supports head dim <= 256, got {dim}"
    )


def _pad_last_dim(tensor, dim):
    pad = dim - tensor.shape[-1]
    if pad == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad))


def _crop_last_dim(tensor, dim):
    if tensor.shape[-1] == dim:
        return tensor
    return tensor[..., :dim].contiguous()


def _unified_blocks(plan):
    if plan == "split":
        return _SPLIT_BLOCK_M, _SPLIT_BLOCK_N
    if plan == "unsplit":
        return _UNSPLIT_BLOCK_M, _UNSPLIT_BLOCK_N
    raise ValueError(f"{plan} is not a unified backward plan")


def _separate_blocks(plan):
    if plan == "split_separate":
        return (
            _SPLIT_DKDV_BLOCK_M,
            _SPLIT_DKDV_BLOCK_N,
            _SPLIT_DQ_BLOCK_M,
            _SPLIT_DQ_BLOCK_N,
        )
    if plan == "unsplit_separate":
        return (
            _UNSPLIT_DKDV_BLOCK_M,
            _UNSPLIT_DKDV_BLOCK_N,
            _UNSPLIT_DQ_BLOCK_M,
            _UNSPLIT_DQ_BLOCK_N,
        )
    raise ValueError(f"{plan} is not a split backward plan")


def _make_unified_bwd_kernel(
    plan,
    *,
    dim,
    is_causal,
    has_window_left,
    has_window_right,
    has_softcap,
    is_varlen,
    is_bhsd,
    heads_q_eq_heads_kv,
    has_seqused_q,
    has_seqused_k,
    dtype,
    use_static_dims=False,
    recompute_delta=False,
    enable_index_type_promotion=False,
):
    block_M, block_N = _unified_blocks(plan)
    jit = flashattn_bwd_ws_split if plan == "split" else flashattn_bwd_ws_unsplit
    jit = _jit_for_index_type_promotion(jit, enable_index_type_promotion)
    jit_kwargs = dict(
        dim=dim,
        is_causal=is_causal,
        has_window_left=has_window_left,
        has_window_right=has_window_right,
        has_softcap=has_softcap,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        has_seqused_q=has_seqused_q,
        has_seqused_k=has_seqused_k,
        block_M=block_M,
        block_N=block_N,
        dtype=dtype,
        use_static_dims=use_static_dims,
    )
    if plan == "split":
        jit_kwargs["recompute_delta"] = recompute_delta
    return jit(**jit_kwargs)


def _make_separate_bwd_kernel(
    plan,
    *,
    dim,
    is_causal,
    has_window_left,
    has_window_right,
    has_softcap,
    is_varlen,
    is_bhsd,
    heads_q_eq_heads_kv,
    has_seqused_q,
    has_seqused_k,
    dtype,
    use_static_dims=False,
    enable_index_type_promotion=False,
):
    dkdv_block_M, dkdv_block_N, dq_block_M, dq_block_N = _separate_blocks(plan)
    if plan == "split_separate":
        dkdv_jit = flashattn_bwd_ws_split_dkdv
        dq_jit = flashattn_bwd_ws_split_dq
    else:
        dkdv_jit = flashattn_bwd_ws_unsplit_dkdv
        dq_jit = flashattn_bwd_ws_unsplit_dq

    dkdv_jit = _jit_for_index_type_promotion(dkdv_jit, enable_index_type_promotion)
    dq_jit = _jit_for_index_type_promotion(dq_jit, enable_index_type_promotion)
    dkdv_use_static_dims = _use_static_dims_for_dkdv(
        plan, use_static_dims, is_varlen, heads_q_eq_heads_kv
    )
    dkdv_kernel = dkdv_jit(
        dim=dim,
        is_causal=is_causal,
        has_window_left=has_window_left,
        has_window_right=has_window_right,
        has_softcap=has_softcap,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        has_seqused_q=has_seqused_q,
        has_seqused_k=has_seqused_k,
        block_M=dkdv_block_M,
        block_N=dkdv_block_N,
        dtype=dtype,
        use_static_dims=dkdv_use_static_dims,
    )
    dq_kernel = dq_jit(
        dim=dim,
        is_causal=is_causal,
        has_window_left=has_window_left,
        has_window_right=has_window_right,
        has_softcap=has_softcap,
        is_varlen=is_varlen,
        is_bhsd=is_bhsd,
        heads_q_eq_heads_kv=heads_q_eq_heads_kv,
        has_seqused_q=has_seqused_q,
        has_seqused_k=has_seqused_k,
        block_M=dq_block_M,
        block_N=dq_block_N,
        dtype=dtype,
        use_static_dims=use_static_dims,
    )

    def run(
        Q,
        K,
        V,
        dQ,
        dK,
        dV,
        dO,
        cu_seq_q,
        cu_seq_kv,
        seqused_q,
        seqused_k,
        Lse,
        Delta,
        max_seq_q,
        max_seq_kv,
        window_size_left,
        window_size_right,
        softcap,
        smscale,
        rln2_scale,
    ):
        dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            cu_seq_q,
            cu_seq_kv,
            seqused_q,
            seqused_k,
            Lse,
            Delta,
            max_seq_kv,
            window_size_left,
            window_size_right,
            softcap,
            smscale,
            rln2_scale,
        )
        dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            cu_seq_q,
            cu_seq_kv,
            seqused_q,
            seqused_k,
            Lse,
            Delta,
            max_seq_q,
            window_size_left,
            window_size_right,
            softcap,
            smscale,
            rln2_scale,
        )

    return run


def flashattn_bwd_ws(
    dim,
    is_causal,
    is_varlen,
    is_local=False,
    is_bhsd=False,
    heads_q_eq_heads_kv=False,
    has_seqused_q=False,
    has_seqused_k=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    smscale=None,
    dtype="bfloat16",
    deterministic=False,
):
    has_window_left = is_local and window_size_left >= 0
    has_window_right = is_local and window_size_right >= 0 and not is_causal
    softcap = float(softcap)
    has_softcap = softcap > 0.0
    smscale, rln2_scale = _resolve_backward_scales(dim, smscale)
    plan = _select_bwd_plan(dim, deterministic, heads_q_eq_heads_kv)
    if plan.endswith("_separate"):
        kernel = _make_separate_bwd_kernel(
            plan,
            dim=dim,
            is_causal=is_causal,
            has_window_left=has_window_left,
            has_window_right=has_window_right,
            has_softcap=has_softcap,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            has_seqused_q=has_seqused_q,
            has_seqused_k=has_seqused_k,
            dtype=dtype,
        )
    else:
        kernel = _make_unified_bwd_kernel(
            plan,
            dim=dim,
            is_causal=is_causal,
            has_window_left=has_window_left,
            has_window_right=has_window_right,
            has_softcap=has_softcap,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            has_seqused_q=has_seqused_q,
            has_seqused_k=has_seqused_k,
            dtype=dtype,
        )

    def run(*args):
        return kernel(
            *args,
            window_size_left,
            window_size_right,
            softcap,
            smscale,
            rln2_scale,
        )

    return run


flashattn_bwd_ws_unsplit_d128 = flashattn_bwd_ws_unsplit


def _normalize_window_for_backward(is_causal, window_size, max_seqlen_q, max_seqlen_k):
    if window_size is None:
        window_size = (-1, -1)
    window_size_left, window_size_right = window_size
    window_size_left = -1 if window_size_left is None else int(window_size_left)
    window_size_right = -1 if window_size_right is None else int(window_size_right)
    if max_seqlen_k is not None and window_size_left >= max_seqlen_k - 1:
        window_size_left = -1
    if max_seqlen_q is not None and window_size_right >= max_seqlen_q - 1:
        window_size_right = -1
    if is_causal:
        window_size_right = 0
    kernel_is_causal = window_size_right == 0
    has_window_left = window_size_left >= 0
    has_window_right = window_size_right >= 0 and not kernel_is_causal
    return (
        kernel_is_causal,
        has_window_left,
        has_window_right,
        window_size_left,
        window_size_right,
    )


def _resolve_backward_scales(actual_qk_dim, smscale):
    resolved_smscale = actual_qk_dim**-0.5 if smscale is None else float(smscale)
    return resolved_smscale, resolved_smscale * _LOG2E


def _check_seqused(name, tensor, batch):
    if tensor is None:
        return
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    if tensor.numel() != batch:
        raise ValueError(
            f"{name} must have shape ({batch},), got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def flashattn_varlen_bwd_interface(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    is_causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    smscale=None,
    dtype=None,
    is_bhsd=False,
    deterministic=False,
):
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen and (cu_seqlens_q is None or cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")

    if is_varlen:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("varlen inputs must have shape [total_seq, heads, dim]")
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("max_seqlen_q and max_seqlen_k are required for varlen")
        q_flat, k_flat, v_flat = q, k, v
        out_flat, dout_flat = out, dout
        total_seq_q, heads_q, dim_qk = q_flat.shape
        total_seq_kv, heads_kv, dim_k = k_flat.shape
        dim_v = v_flat.shape[-1]
        if dim_k != dim_qk:
            raise ValueError("q and k must have the same head dimension")
        out_expected_shape = (total_seq_q, heads_q, dim_v)
        lse_expected_shape = (heads_q, total_seq_q)
    else:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "non-varlen inputs must have shape [batch, seqlen, heads, dim]"
            )
        if softmax_lse.ndim != 3:
            raise ValueError(
                "non-varlen softmax_lse must have shape [batch, heads, seqlen]"
            )
        batch = q.shape[0]
        if softmax_lse.shape[0] != batch:
            raise ValueError("q and softmax_lse must have the same batch size")
        _, seqlen_q, heads_q, dim_qk = q.shape
        batch_k, seqlen_k, heads_kv, dim_k = k.shape
        dim_v = v.shape[-1]
        expected_v_shape = (batch, seqlen_k, heads_kv, dim_v)
        if batch_k != batch:
            raise ValueError("q and k must have the same batch size for non-varlen")
        if dim_k != dim_qk:
            raise ValueError("q and k must have the same head dimension")
        if v.shape != expected_v_shape:
            raise ValueError(
                f"v must match k layout for non-varlen backward, expected {expected_v_shape}, "
                f"got {tuple(v.shape)}"
            )
        total_seq_q = batch * seqlen_q
        total_seq_kv = batch * seqlen_k
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k
        q_flat, k_flat, v_flat = q, k, v
        out_flat, dout_flat = out, dout
        out_expected_shape = (batch, seqlen_q, heads_q, dim_v)
        lse_expected_shape = (batch, heads_q, seqlen_q)

    if is_varlen and v_flat.shape != (total_seq_kv, heads_kv, dim_v):
        raise ValueError(
            "v must match k sequence/head layout for the TileLang varlen backward interface"
        )
    if out.shape != out_expected_shape or dout.shape != out_expected_shape:
        raise ValueError("out and dout must have the same shape as q")
    if is_varlen and (cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D tensors")
    if softmax_lse.shape != lse_expected_shape:
        raise ValueError(
            f"softmax_lse must be shaped {lse_expected_shape}, got {tuple(softmax_lse.shape)}"
        )
    if not softmax_lse.is_contiguous():
        raise ValueError("softmax_lse must be contiguous")

    actual_qk_dim = dim_qk
    actual_v_dim = dim_v
    if actual_qk_dim < 8 or actual_v_dim < 8:
        raise NotImplementedError(
            "TileLang FlashAttention backward requires qk/v head dims >= 8, "
            f"got qk={actual_qk_dim}, v={actual_v_dim}"
        )

    # Empty query or KV batches have identically zero gradients. TileLang TMA
    # descriptors cannot represent zero-sized global tensors, so do not launch
    # any backward kernels for these inputs.
    if total_seq_q == 0 or total_seq_kv == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    runtime_qk_dim = ceil_div(actual_qk_dim, 8) * 8
    runtime_v_dim = ceil_div(actual_v_dim, 8) * 8
    kernel_dim = max(
        _select_kernel_dim(runtime_qk_dim), _select_kernel_dim(runtime_v_dim)
    )
    work_qk_dim = runtime_qk_dim
    work_v_dim = runtime_v_dim
    static_dims_eligible = work_qk_dim == kernel_dim and work_v_dim == kernel_dim
    needs_qk_pad = runtime_qk_dim != actual_qk_dim
    needs_v_pad = runtime_v_dim != actual_v_dim

    if needs_qk_pad:
        _check_attention_strides("q", q_flat, multiple=1)
        _check_attention_strides("k", k_flat, multiple=1)
        # TileLang kernels require head-dim strides aligned to 8 elements.
        # Logical tails up to the compile dim are handled by descriptor OOB
        # loads and guarded gradient writeback inside the kernels.
        q_flat = _pad_last_dim(q_flat, work_qk_dim)
        k_flat = _pad_last_dim(k_flat, work_qk_dim)
    if needs_v_pad:
        for name, tensor in (("v", v_flat), ("out", out_flat), ("dout", dout_flat)):
            _check_attention_strides(name, tensor, multiple=1)
        v_flat = _pad_last_dim(v_flat, work_v_dim)
        out_flat = _pad_last_dim(out_flat, work_v_dim)
        dout_flat = _pad_last_dim(dout_flat, work_v_dim)

    for name, tensor in (
        ("q", q_flat),
        ("k", k_flat),
        ("v", v_flat),
        ("out", out_flat),
        ("dout", dout_flat),
    ):
        _check_attention_strides(name, tensor)

    kernel_dtype = dtype if dtype is not None else to_tilelang_dtype(q.dtype)
    heads_q_eq_heads_kv = heads_q == heads_kv
    if heads_q % heads_kv != 0:
        raise ValueError(
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv})"
        )

    smscale, rln2_scale = _resolve_backward_scales(actual_qk_dim, smscale)

    plan = _select_bwd_plan(kernel_dim, deterministic, heads_q_eq_heads_kv)

    batch = cu_seqlens_q.numel() - 1 if is_varlen else q.shape[0]
    _check_seqused("seqused_q", seqused_q, batch)
    _check_seqused("seqused_k", seqused_k, batch)
    (
        kernel_is_causal,
        has_window_left,
        has_window_right,
        window_size_left,
        window_size_right,
    ) = _normalize_window_for_backward(
        is_causal, window_size, max_seqlen_q, max_seqlen_k
    )
    is_local = has_window_left or has_window_right
    # Exact static dimensions can hang in dense local kernels. Keep the
    # proven runtime-dimension path until the compiler/backend issue is fixed.
    use_static_dims = static_dims_eligible and (is_varlen or not is_local)
    has_seqused_q = seqused_q is not None
    has_seqused_k = seqused_k is not None
    softcap = float(softcap)
    has_softcap = softcap > 0.0
    plan = _select_bwd_feature_plan(
        plan,
        use_static_dims,
        kernel_is_causal,
        has_window_left,
        has_window_right,
        has_softcap,
    )
    use_separate = plan.endswith("_separate")
    if use_separate:
        _, _, block_M, block_N = _separate_blocks(plan)
    else:
        block_M, block_N = _unified_blocks(plan)
    recompute_delta = _should_recompute_delta(plan, max_seqlen_k, block_N)
    max_seq_q_padded = ceil_div(max_seqlen_q, block_M) * block_M
    dq_work_shape = (
        (total_seq_q, heads_q, work_qk_dim)
        if is_varlen
        else (q.shape[0], q.shape[1], q.shape[2], work_qk_dim)
    )
    dk_work_shape = (
        (total_seq_kv, heads_kv, work_qk_dim)
        if is_varlen
        else (q.shape[0], seqlen_k, heads_kv, work_qk_dim)
    )
    dv_work_shape = (
        (total_seq_kv, heads_kv, work_v_dim)
        if is_varlen
        else (q.shape[0], seqlen_k, heads_kv, work_v_dim)
    )

    kernel_arg_byte_spans = [
        _tensor_cosize_bytes(q_flat),
        _tensor_cosize_bytes(k_flat),
        _tensor_cosize_bytes(v_flat),
        _tensor_cosize_bytes(out_flat),
        _tensor_cosize_bytes(dout_flat),
        _tensor_cosize_bytes(softmax_lse),
        _contiguous_cosize_bytes((total_seq_q, heads_q), torch.float32),
    ]
    if use_separate:
        kernel_arg_byte_spans.append(_contiguous_cosize_bytes(dq_work_shape, q.dtype))
    else:
        kernel_arg_byte_spans.append(
            _contiguous_cosize_bytes(
                (batch, heads_q, max_seq_q_padded, kernel_dim), torch.float32
            )
        )
    if not heads_q_eq_heads_kv:
        kv_accum_shape = (
            (total_seq_kv, heads_q, kernel_dim)
            if is_varlen
            else (batch, seqlen_k, heads_q, kernel_dim)
        )
        kernel_arg_byte_spans.extend(
            (
                _contiguous_cosize_bytes(kv_accum_shape, torch.float32),
                _contiguous_cosize_bytes(dk_work_shape, torch.float32),
                _contiguous_cosize_bytes(dv_work_shape, torch.float32),
            )
        )
    enable_index_type_promotion = _needs_index_type_promotion(*kernel_arg_byte_spans)

    delta = torch.empty((total_seq_q, heads_q), device=q.device, dtype=torch.float32)
    compute_delta_kernel = None
    if not recompute_delta:
        compute_delta = _jit_for_index_type_promotion(
            compute_delta_ws, enable_index_type_promotion
        )
        compute_delta_kernel = compute_delta(
            kernel_dim,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            has_seqused_q=has_seqused_q,
            dtype=kernel_dtype,
            block_M=block_M,
            threads=_THREADS,
            use_strided_tensors=True,
        )

    dq_work = torch.empty(dq_work_shape, dtype=q.dtype, device=q.device)
    dQ_accum = None
    if not use_separate:
        dQ_accum = torch.empty(
            (batch, heads_q, max_seq_q_padded, kernel_dim),
            dtype=torch.float32,
            device=q.device,
        )
    if heads_q_eq_heads_kv:
        if work_qk_dim == actual_qk_dim:
            dK_accum = torch.empty_like(k)
        else:
            dK_accum = torch.empty(dk_work_shape, dtype=k.dtype, device=k.device)
        if work_v_dim == actual_v_dim:
            dV_accum = torch.empty_like(v)
        else:
            dV_accum = torch.empty(dv_work_shape, dtype=v.dtype, device=v.device)
    else:
        kv_accum_shape = (
            (total_seq_kv, heads_q, kernel_dim)
            if is_varlen
            else (batch, seqlen_k, heads_q, kernel_dim)
        )
        dK_accum = torch.empty(kv_accum_shape, dtype=torch.float32, device=q.device)
        dV_accum = torch.empty(kv_accum_shape, dtype=torch.float32, device=q.device)

    cu_seqlens_q_arg = cu_seqlens_q if is_varlen else None
    cu_seqlens_k_arg = cu_seqlens_k if is_varlen else None

    pack_dq_kernel = None
    if use_separate:
        kernel = _make_separate_bwd_kernel(
            plan,
            dim=kernel_dim,
            is_causal=kernel_is_causal,
            has_window_left=has_window_left,
            has_window_right=has_window_right,
            has_softcap=has_softcap,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            has_seqused_q=has_seqused_q,
            has_seqused_k=has_seqused_k,
            dtype=kernel_dtype,
            use_static_dims=use_static_dims,
            enable_index_type_promotion=enable_index_type_promotion,
        )
    else:
        kernel = _make_unified_bwd_kernel(
            plan,
            dim=kernel_dim,
            is_causal=kernel_is_causal,
            has_window_left=has_window_left,
            has_window_right=has_window_right,
            has_softcap=has_softcap,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            heads_q_eq_heads_kv=heads_q_eq_heads_kv,
            has_seqused_q=has_seqused_q,
            has_seqused_k=has_seqused_k,
            dtype=kernel_dtype,
            use_static_dims=use_static_dims,
            recompute_delta=recompute_delta,
            enable_index_type_promotion=enable_index_type_promotion,
        )
        pack_dq = _jit_for_index_type_promotion(
            pack_dq_from_accum_ws, enable_index_type_promotion
        )
        pack_dq_kernel = pack_dq(
            kernel_dim,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            has_seqused_q=has_seqused_q,
            dtype=kernel_dtype,
            use_strided_tensors=True,
        )

    reduce_kv_grads_kernel = None
    singleton_k_dv_kernel = None
    dk_work = None
    dv_work = None
    if not heads_q_eq_heads_kv:
        dk_work = torch.empty(dk_work_shape, dtype=torch.float32, device=q.device)
        dv_work = torch.empty(dv_work_shape, dtype=torch.float32, device=q.device)
        reduce_kv_grads = _jit_for_index_type_promotion(
            reduce_kv_grads_ws, enable_index_type_promotion
        )
        reduce_kv_grads_kernel = reduce_kv_grads(
            heads_q // heads_kv,
            kernel_dim,
            is_varlen=is_varlen,
            is_bhsd=is_bhsd,
            use_strided_tensors=True,
            use_static_dims=use_static_dims,
        )
    elif max_seqlen_k <= 1:
        singleton_k_dv = _jit_for_index_type_promotion(
            singleton_k_dv_ws, enable_index_type_promotion
        )
        singleton_k_dv_kernel = singleton_k_dv(
            kernel_dim,
            is_varlen=is_varlen,
            is_causal=kernel_is_causal,
            has_window_left=has_window_left,
            has_window_right=has_window_right,
            has_seqused_q=has_seqused_q,
            has_seqused_k=has_seqused_k,
            dtype=kernel_dtype,
        )

    # All kernels needed by this backward path have been compiled at this point.
    raise_complete_if_dry_run()

    if compute_delta_kernel is not None:
        compute_delta_kernel(
            out_flat,
            dout_flat,
            cu_seqlens_q_arg,
            seqused_q,
            delta,
            max_seqlen_q,
        )

    if has_seqused_q:
        dq_work.zero_()
    if dQ_accum is not None:
        dQ_accum.zero_()
    dK_accum.zero_()
    dV_accum.zero_()

    if use_separate:
        kernel(
            q_flat,
            k_flat,
            v_flat,
            dq_work,
            dK_accum,
            dV_accum,
            dout_flat,
            cu_seqlens_q_arg,
            cu_seqlens_k_arg,
            seqused_q,
            seqused_k,
            softmax_lse,
            delta,
            max_seqlen_q,
            max_seqlen_k,
            window_size_left,
            window_size_right,
            softcap,
            smscale,
            rln2_scale,
        )
    else:
        kernel(
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            dQ_accum,
            dK_accum,
            dV_accum,
            dout_flat,
            cu_seqlens_q_arg,
            cu_seqlens_k_arg,
            seqused_q,
            seqused_k,
            softmax_lse,
            delta,
            max_seqlen_k,
            window_size_left,
            window_size_right,
            softcap,
            smscale,
            rln2_scale,
        )
        pack_dq_kernel(dQ_accum, cu_seqlens_q_arg, seqused_q, dq_work)

    if singleton_k_dv_kernel is not None:
        singleton_k_dv_kernel(
            dout_flat,
            dV_accum,
            cu_seqlens_q_arg,
            cu_seqlens_k_arg,
            seqused_q,
            seqused_k,
            window_size_left,
            window_size_right,
        )

    dq = _crop_last_dim(dq_work, actual_qk_dim)
    if heads_q_eq_heads_kv:
        dk = _crop_last_dim(dK_accum, actual_qk_dim)
        dv = _crop_last_dim(dV_accum, actual_v_dim)
    else:
        reduce_kv_grads_kernel(dK_accum, dV_accum, dk_work, dv_work)
        dk = _crop_last_dim(dk_work, actual_qk_dim)
        dv = _crop_last_dim(dv_work, actual_v_dim)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

    # With at most one valid key, singleton softmax is constant and its score
    # derivative is exactly zero. The general kernel computes dP and delta via
    # different reductions, which can otherwise leave a small rounding residue.
    if max_seqlen_k <= 1:
        dq.zero_()
        dk.zero_()
    return dq, dk, dv


__all__ = [
    "ceil_div",
    "compute_delta_ws",
    "flashattn_bwd_ws",
    "flashattn_bwd_ws_split",
    "flashattn_bwd_ws_split_dkdv",
    "flashattn_bwd_ws_split_dq",
    "flashattn_bwd_ws_unsplit",
    "flashattn_bwd_ws_unsplit_d128",
    "flashattn_bwd_ws_unsplit_dkdv",
    "flashattn_bwd_ws_unsplit_dq",
    "flashattn_varlen_bwd_interface",
    "pack_dq_from_accum_ws",
    "reduce_kv_grads_ws",
    "singleton_k_dv_ws",
    "to_tilelang_dtype",
]
