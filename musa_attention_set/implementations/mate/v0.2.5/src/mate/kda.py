"""Public chunk and decode APIs for KDA."""

from __future__ import annotations

import torch

from mate.api_logging import mate_api
from mate.jit.kda_ops import (
    get_kda_fused_ops_function_name,
    get_kda_fused_ops_module,
    make_kda_fused_ops_config,
)
from mate.kda_kernels.tilelang import kda_decode as kda_decode_tilelang

_SUPPORTED_QKVA_DTYPES = (torch.float16, torch.bfloat16)


def _as_4d_varlen_input(
    x: torch.Tensor,
    *,
    name: str,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if cu_seqlens is None:
        if x.ndim != 4:
            raise ValueError(f"{name} must be a 4D tensor [B, T, H, 128].")
        return x, False
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}.shape[0] must be 1 when cu_seqlens is provided; "
                "flatten variable-length input as [S, H, 128]."
            )
        return x, False
    raise ValueError(f"{name} must be [S, H, 128] or [1, S, H, 128] for varlen.")


def _check_state_dtype(
    x: torch.Tensor,
    *,
    name: str,
    value_dtype: torch.dtype,
) -> None:
    if x.dtype not in (value_dtype, torch.float32):
        raise TypeError(f"{name} must have dtype {value_dtype} or torch.float32.")


@mate_api
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    use_qk_l2norm_in_kernel: bool = True,
    output: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the fused chunk KDA kernel.

    Args:
        q: Query tensor with shape ``[B, T, Hqk, 128]`` for dense mode or
            ``[S, Hqk, 128]`` / ``[1, S, Hqk, 128]`` for varlen mode.
        k: Key tensor with the same shape and dtype as ``q``.
        v: Value tensor with shape ``[B, T, Hv, 128]`` or varlen equivalent.
        g: Gate input tensor with the same shape as ``v``.
        beta: Beta logits tensor with shape ``[B, T, Hv]`` or varlen equivalent.
        scale: Optional QK scaling factor. Defaults to ``128**-0.5``.
        initial_state: Optional recurrent state tensor.
        output_final_state: Whether to return the final recurrent state.
        cu_seqlens: Optional cumulative sequence lengths for varlen mode.
        A_log: Optional per-head gate parameter tensor.
        dt_bias: Optional per-head, per-channel gate bias tensor.
        lower_bound: Gate lower bound used when gate parameters ``A_log`` and
            ``dt_bias`` are enabled. Defaults to ``-5.0``.
        use_qk_l2norm_in_kernel: Whether to normalize Q/K in the kernel.
        output: Optional preallocated output tensor.
        final_state: Optional preallocated final-state tensor.
    """

    q, squeeze_varlen = _as_4d_varlen_input(q, name="q", cu_seqlens=cu_seqlens)
    k, _ = _as_4d_varlen_input(k, name="k", cu_seqlens=cu_seqlens)
    v, _ = _as_4d_varlen_input(v, name="v", cu_seqlens=cu_seqlens)
    g, _ = _as_4d_varlen_input(g, name="g", cu_seqlens=cu_seqlens)
    if beta.ndim == 2 and cu_seqlens is not None:
        beta = beta.unsqueeze(0)

    if (
        q.shape[-1] != 128
        or k.shape[-1] != 128
        or v.shape[-1] != 128
        or g.shape[-1] != 128
    ):
        raise ValueError("chunk_kda currently requires D=128.")
    if k.shape != q.shape:
        raise ValueError("k must have the same shape as q.")
    if q.shape[:2] != v.shape[:2] or q.shape[:2] != g.shape[:2]:
        raise ValueError("q, v and g must have matching [B, T] dimensions.")
    if g.shape != v.shape:
        raise ValueError("g must have the same shape as v.")
    if v.shape[2] % q.shape[2] != 0:
        raise ValueError("GVA requires v/g heads to be divisible by q/k heads.")
    if beta.shape != v.shape[:3]:
        raise ValueError("beta must have shape [B, T, Hv].")
    if q.dtype not in _SUPPORTED_QKVA_DTYPES:
        raise TypeError("chunk_kda supports torch.float16 and torch.bfloat16 inputs.")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or g.dtype != q.dtype
        or beta.dtype != q.dtype
    ):
        raise TypeError("k, v, g and beta must have the same dtype as q.")
    if (A_log is None) != (dt_bias is None):
        raise ValueError("A_log and dt_bias must be provided together.")
    if initial_state is not None:
        _check_state_dtype(initial_state, name="initial_state", value_dtype=q.dtype)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if cu_seqlens is not None:
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64.")
        cu_seqlens = cu_seqlens.contiguous()

    if output is None:
        output = torch.empty_like(v)
    elif squeeze_varlen and output.ndim == 3:
        output = output.unsqueeze(0)
    if output.shape != v.shape:
        raise ValueError("output must have the same shape as v.")
    if output.dtype != q.dtype:
        raise TypeError("output must have the same dtype as q.")

    if output_final_state and final_state is None:
        nseq = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else q.shape[0]
        state_dtype = initial_state.dtype if initial_state is not None else q.dtype
        final_state = torch.empty(
            (nseq, v.shape[2], 128, 128),
            device=q.device,
            dtype=state_dtype,
        )
    elif not output_final_state:
        final_state = None
    elif final_state is not None:
        _check_state_dtype(final_state, name="final_state", value_dtype=q.dtype)

    if (
        initial_state is not None
        and final_state is not None
        and initial_state.dtype != final_state.dtype
    ):
        raise TypeError("initial_state and final_state must have the same dtype.")

    state_fp32 = (
        initial_state is not None and initial_state.dtype == torch.float32
    ) or (final_state is not None and final_state.dtype == torch.float32)
    state_dtype = (
        initial_state.dtype
        if initial_state is not None
        else final_state.dtype
        if final_state is not None
        else q.dtype
    )
    kda_config = make_kda_fused_ops_config(
        q.dtype,
        state_dtype=state_dtype,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        state_fp32=state_fp32,
        has_gate_params=A_log is not None,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(use_qk_l2norm_in_kernel),
    )
    kda_func_name = get_kda_fused_ops_function_name(kda_config)

    get_kda_fused_ops_module(kda_config).get_function(kda_func_name)(
        q,
        k,
        v,
        g,
        beta,
        output,
        initial_state,
        final_state,
        cu_seqlens,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        bool(use_qk_l2norm_in_kernel),
    )

    if squeeze_varlen:
        output = output.squeeze(0)
    if output_final_state:
        assert final_state is not None
        return output, final_state
    return output


_SUPPORTED_GATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_QKV_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_STATE_DTYPES = (torch.float32, torch.bfloat16)
_SUPPORTED_OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_DT_BIAS_DTYPES = (torch.float32, torch.bfloat16)


def _check_same_device(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"Expected {name} to be on device {reference.device}, got {tensor.device}."
            )


def _validate_common_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A_log: torch.Tensor | None,
    a: torch.Tensor,
    dt_bias: torch.Tensor | None,
    b: torch.Tensor,
    *,
    output: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int]:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "q, k, and v must each have shape [B, T, H, D] or [1, total_tokens, H, D]."
        )
    if a.dim() != 4:
        raise ValueError("a must have shape [B, T, HV, K].")
    if A_log is not None and A_log.dim() != 1:
        raise ValueError("A_log must have shape [HV].")
    if dt_bias is not None and dt_bias.dim() != 2:
        raise ValueError("dt_bias must have shape [HV, K] when provided.")

    B, T, H, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, HV, V = v.shape

    if (Bk, Tk, Hk, Kk) != (B, T, H, K):
        raise ValueError(
            f"k must match q shape [B, T, H, K], got q={tuple(q.shape)}, k={tuple(k.shape)}."
        )
    if (Bv, Tv) != (B, T):
        raise ValueError(
            f"v must match q in batch/time dims, got q={tuple(q.shape)}, v={tuple(v.shape)}."
        )
    if HV % H != 0:
        raise ValueError(f"Expected HV to be divisible by H, got HV={HV}, H={H}.")
    if a.shape != (B, T, HV, K):
        raise ValueError(f"Expected a shape {(B, T, HV, K)}, got a={tuple(a.shape)}.")
    if A_log is not None and A_log.numel() != HV:
        raise ValueError(f"A_log must have {HV} elements, got {A_log.numel()}.")
    if dt_bias is not None and dt_bias.shape != (HV, K):
        raise ValueError(
            f"dt_bias must have shape {(HV, K)} when provided, got {tuple(dt_bias.shape)}."
        )

    if q.dtype not in _SUPPORTED_QKV_DTYPES:
        raise NotImplementedError(
            f"q/k/v dtype must be float16 or bfloat16, got {q.dtype}."
        )
    if A_log is not None and A_log.dtype != torch.float32:
        raise ValueError(f"A_log must be float32, got A_log={A_log.dtype}.")
    if dt_bias is not None and dt_bias.dtype not in _SUPPORTED_DT_BIAS_DTYPES:
        raise ValueError(
            f"dt_bias must be float32 or bfloat16 when provided, got dt_bias={dt_bias.dtype}."
        )

    same_device_tensors = {
        "k": k,
        "v": v,
        "a": a,
        "b": b,
    }
    if A_log is not None:
        same_device_tensors["A_log"] = A_log
    if dt_bias is not None:
        same_device_tensors["dt_bias"] = dt_bias
    _check_same_device(q, **same_device_tensors)

    if output is not None:
        if output.shape != (B, T, HV, V):
            raise ValueError(
                f"Expected output shape {(B, T, HV, V)}, got {tuple(output.shape)}."
            )
        if output.dtype not in _SUPPORTED_OUTPUT_DTYPES:
            raise NotImplementedError(
                f"Unsupported output dtype {output.dtype}. "
                f"Supported dtypes: {_SUPPORTED_OUTPUT_DTYPES}."
            )
        if output.device != q.device:
            raise ValueError(
                f"Expected output to be on device {q.device}, got {output.device}."
            )

    return B, T, H, K, HV, V


def _run_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor | None,
    a: torch.Tensor,
    dt_bias: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    b: torch.Tensor,
    *,
    state_indices: torch.Tensor | None,
    scale: float,
    output: torch.Tensor | None,
    use_qk_l2norm: bool | None,
    lower_bound: float | None,
    use_gate_in_kernel: bool | None,
    use_lower_bound: bool | None,
    apply_beta_sigmoid: bool | None,
    allow_neg_eigval: bool | None,
    use_initial_state: bool,
    state_v_first: bool,
    is_varlen: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size = q.shape[0]
    num_v_heads = v.shape[2]
    head_dim_v = v.shape[3]

    if output is None:
        output = torch.empty(
            (batch_size, q.shape[1], num_v_heads, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
    if use_initial_state or inplace_final_state:
        if state is None:
            raise ValueError(
                "state must be provided when use_initial_state or "
                "inplace_final_state is enabled."
            )
        state_is_contiguous = state.is_contiguous()
        state_kernel = state if state_is_contiguous else state.contiguous()
    else:
        state_is_contiguous = True
        pool_size = 1 if state_indices is None else int(state_indices.shape[0])
        if state_v_first:
            state_shape = (pool_size, num_v_heads, head_dim_v, q.shape[-1])
        else:
            state_shape = (pool_size, num_v_heads, q.shape[-1], head_dim_v)
        state_kernel = torch.empty(state_shape, dtype=state_dtype, device=q.device)
    output_result, final_state_result = (
        kda_decode_tilelang.run_gated_delta_rule_decode_vk_fp32(
            q=q,
            k=k,
            v=v,
            state=state_kernel,
            state_indices=state_indices,
            A_log=A_log,
            g=a,
            dt_bias=dt_bias,
            b=b,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            output=output,
            scale=float(scale),
            lower_bound=0.0 if lower_bound is None else float(lower_bound),
            use_qk_l2norm=True if use_qk_l2norm is None else bool(use_qk_l2norm),
            is_varlen=bool(is_varlen),
            inplace_final_state=bool(inplace_final_state),
            is_beta_headwise=b.dim() == 4,
            is_continuous_batching=state_indices is not None,
            is_spec_decoding=num_accepted_tokens is not None,
            store_final_state=bool(store_final_state),
            has_dt_bias=dt_bias is not None,
            use_gate_in_kernel=True
            if use_gate_in_kernel is None
            else bool(use_gate_in_kernel),
            use_lower_bound=False if use_lower_bound is None else bool(use_lower_bound),
            apply_beta_sigmoid=False
            if apply_beta_sigmoid is None
            else bool(apply_beta_sigmoid),
            allow_neg_eigval=False
            if allow_neg_eigval is None
            else bool(allow_neg_eigval),
            state_v_first=bool(state_v_first),
            use_initial_state=bool(use_initial_state),
        )
    )

    if not state_is_contiguous:
        state.copy_(state_kernel)
        if inplace_final_state and final_state_result is not None:
            final_state_result = state

    return output_result, final_state_result


@mate_api
def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor | None,
    a: torch.Tensor,
    dt_bias: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    b: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    use_qk_l2norm: bool | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool | None = None,
    use_lower_bound: bool | None = None,
    apply_beta_sigmoid: bool | None = None,
    allow_neg_eigval: bool | None = None,
    use_initial_state: bool | None = None,
    state_v_first: bool | None = None,
    is_varlen: bool | None = None,
    store_final_state: bool = False,
    inplace_final_state: bool = False,
    state_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Unified Gated Delta Rule Decode API.

    Args:
        q (torch.Tensor):
            Query of shape ``[B, T, H, K] or [1, total_tokens, H, K]``.
        k (torch.Tensor):
            Key of shape ``[B, T, H, K] or [1, total_tokens, H, K]``. Must match ``q`` in shape and dtype.
        v (torch.Tensor):
            Value of shape ``[B, T, HV, V] or [1, total_tokens, HV, K]``. Must match ``q`` in batch/time dims.
        state (Optional[torch.Tensor]):
            State buffer in VK layout. Shape is ``[B_or_pool, HV, V, K]`` when
            `state_v_first` is `True`, otherwise
            ``[B_or_pool, HV, K, V]``. Supported dtypes are ``torch.float32``
            and ``torch.bfloat16``. It may be `None` when initial-state reads and
            in-place writes are disabled. The returned final state uses the same
            dtype. BF16 state is widened for arithmetic and rounded back to BF16
            after each decoded token, matching repeated single-token decode.
        A_log (Optional[torch.Tensor]):
            Optional log decay of shape ``[HV]``. When `None`, the backend uses zeros.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, T, HV, K]``.
        dt_bias (Optional[torch.Tensor]):
            Optional decay bias of shape ``[HV, K]``. When ``None``, the backend runs with
            `has_dt_bias=False` and supplies a dummy tensor internally. Supported dtypes are
            ``torch.float32`` and ``torch.bfloat16``.
        num_accepted_tokens (Optional[torch.Tensor]):
            Optional speculative-decoding accepted-token counts of shape ``[B]``.
        cu_seqlens (Optional[torch.Tensor]):
            Optional cumulative sequence lengths for varlen mode.
        b (torch.Tensor):
            Update gate of shape ``[B, T, HV]`` or ``[B, T, HV, V]``.
        state_indices (Optional[torch.Tensor]):
            Optional ``[B]`` or ``[B, T]`` int32/int64 mapping batch entries
            and decode steps to a state pool. Every non-negative index must be
            smaller than the state pool size.
        scale (Optional[float]):
            Query scale. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Optional pre-allocated output tensor of shape ``[B, T, HV, V]``.
        use_qk_l2norm (Optional[bool]):
            Whether to L2-normalize q and k in-kernel. ``None`` uses the current default.
        lower_bound (Optional[float]):
            Optional backend lower-bound value. ``None`` uses the current default.
        use_gate_in_kernel (Optional[bool]):
            Optional explicit control of in-kernel gate application.
        use_lower_bound (Optional[bool]):
            Optional explicit control of backend lower-bound logic.
        apply_beta_sigmoid (Optional[bool]):
            Optional explicit control of in-kernel sigmoid application on ``b``.
        allow_neg_eigval (Optional[bool]):
            Optional explicit control of backend negative-eigenvalue handling.
        use_initial_state (Optional[bool]):
            Optional explicit control of whether the backend reads the provided state.
        state_v_first (Optional[bool]):
            State matrix layout selector. ``None`` defaults to `True`.
        is_varlen (Optional[bool]):
            Whether to run the varlen path. ``None`` defaults to whether
            `cu_seqlens` is provided.
        store_final_state (bool):
            Whether to request final-state output from the backend.
        inplace_final_state (bool):
            Whether final-state writes should reuse `state`.
        state_dtype (Optional[torch.dtype]):
            State storage dtype. When omitted, it is inferred from `state`, or
            defaults to `torch.float32` when `state` is `None`. Supported dtypes
            are `torch.float32` and `torch.bfloat16`.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - output: backend output tensor
            - final_state: semantic final-state result, or `None` when not requested
    """
    resolved_state_v_first = True if state_v_first is None else bool(state_v_first)
    resolved_is_varlen = (
        (cu_seqlens is not None) if is_varlen is None else bool(is_varlen)
    )
    resolved_use_initial_state = (
        True if use_initial_state is None else bool(use_initial_state)
    )
    if state_dtype is None:
        resolved_state_dtype = torch.float32 if state is None else state.dtype
    else:
        resolved_state_dtype = state_dtype
    if resolved_state_dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError(
            "state_dtype must be torch.float32 or torch.bfloat16, "
            f"got {resolved_state_dtype}."
        )

    _, _, _, K, HV, V = _validate_common_decode_inputs(
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        output=output,
        num_accepted_tokens=num_accepted_tokens,
    )

    if resolved_is_varlen:
        if cu_seqlens is None or cu_seqlens.dim() != 1:
            raise ValueError("cu_seqlens must be a 1D tensor when is_varlen=True.")
        logical_batch_size = cu_seqlens.numel() - 1
    else:
        logical_batch_size = q.shape[0]

    if num_accepted_tokens is not None and (
        num_accepted_tokens.dim() != 1
        or num_accepted_tokens.shape[0] != logical_batch_size
    ):
        raise ValueError(
            "num_accepted_tokens must have shape "
            f"[B={logical_batch_size}], got {tuple(num_accepted_tokens.shape)}."
        )

    if state_indices is not None:
        if state_indices.dim() not in (1, 2):
            raise ValueError(
                "state_indices must have shape [B] or [B, T], "
                f"got {tuple(state_indices.shape)}."
            )
        if state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "state_indices must have dtype torch.int32 or torch.int64, "
                f"got {state_indices.dtype}."
            )
        if state_indices.shape[0] != logical_batch_size:
            raise ValueError(
                f"state_indices first dimension must be {logical_batch_size}, "
                f"got {state_indices.shape[0]}."
            )
        if state_indices.dim() == 1 and (
            num_accepted_tokens is not None
            or (inplace_final_state and (resolved_is_varlen or q.shape[1] > 1))
        ):
            raise ValueError(
                "1D state_indices only support non-speculative single-token "
                "in-place decode; use [B, T] state_indices for speculative, "
                "multi-token, or varlen in-place decode."
            )
        if state_indices.dim() == 2:
            if state_indices.shape[1] < 1:
                raise ValueError("state_indices must contain at least one column.")
            if (
                not resolved_is_varlen
                and (inplace_final_state or num_accepted_tokens is not None)
                and state_indices.shape[1] < q.shape[1]
            ):
                raise ValueError(
                    f"state_indices needs at least {q.shape[1]} columns for "
                    "fixed multi-token in-place or speculative decode, "
                    f"got {state_indices.shape[1]}."
                )

    if resolved_use_initial_state and state is None:
        raise ValueError("state must be provided when use_initial_state=True.")
    if inplace_final_state and state is None:
        raise ValueError("state must be provided when inplace_final_state=True.")
    if state is not None:
        if state.dtype not in _SUPPORTED_STATE_DTYPES:
            raise ValueError(
                "state must have dtype torch.float32 or torch.bfloat16, "
                f"got {state.dtype}."
            )
        if state.device != q.device:
            raise ValueError(
                f"Expected state to be on device {q.device}, got {state.device}."
            )
        if state.dtype != resolved_state_dtype:
            raise ValueError(
                f"state has dtype {state.dtype}, but state_dtype="
                f"{resolved_state_dtype}."
            )
        expected_tail = (HV, V, K) if resolved_state_v_first else (HV, K, V)
        if state.dim() != 4 or tuple(state.shape[1:]) != expected_tail:
            raise ValueError(
                "state must have shape [pool, HV, V, K] for V-first or "
                "[pool, HV, K, V] for K-first; "
                f"expected trailing dimensions {expected_tail}, got {tuple(state.shape)}."
            )
        if state_indices is None and state.shape[0] < logical_batch_size:
            raise ValueError(
                f"state pool must contain at least {logical_batch_size} slots, "
                f"got {state.shape[0]}."
            )
        if state_indices is not None and state.shape[0] == 0:
            raise ValueError("state pool must contain at least one slot.")

    return _run_kda_decode(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        b=b,
        state_indices=state_indices,
        scale=K**-0.5 if scale is None else float(scale),
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_lower_bound=use_lower_bound,
        apply_beta_sigmoid=apply_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        use_initial_state=resolved_use_initial_state,
        state_v_first=resolved_state_v_first,
        is_varlen=resolved_is_varlen,
        store_final_state=store_final_state,
        inplace_final_state=inplace_final_state,
        state_dtype=resolved_state_dtype,
    )
