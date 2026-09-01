"""Unified Gated Delta Rule decode API for MATE.

This module follows the FlashInfer unified decode API design and keeps the API
layer responsible for parameter semantics, dispatch, and compatibility checks.
Concrete execution is delegated to backend-specific implementations.
"""

from typing import Optional, Tuple

import torch

from mate.api_logging import mate_api
from mate.gdn_kernels.tilelang import gdn_decode as gdn_decode_tilelang
from mate.gdn_kernels.tilelang import gdn_mtp as gdn_mtp_tilelang

_SUPPORTED_QKV_DTYPES = (torch.float16, torch.bfloat16)
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
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    *,
    output: Optional[torch.Tensor],
) -> Tuple[int, int, int, int, int, int]:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must each have shape [B, T, H, D].")
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("a and b must each have shape [B, T, HV].")
    if A_log.dim() != 1 or dt_bias.dim() != 1:
        raise ValueError("A_log and dt_bias must each have shape [HV].")

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
    if a.shape != (B, T, HV) or b.shape != (B, T, HV):
        raise ValueError(
            f"Expected a/b shape {(B, T, HV)}, got a={tuple(a.shape)}, b={tuple(b.shape)}."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"A_log and dt_bias must each have {HV} elements, got "
            f"{A_log.numel()} and {dt_bias.numel()}."
        )

    if q.dtype not in _SUPPORTED_QKV_DTYPES:
        raise NotImplementedError(
            f"q/k/v dtype must be float16 or bfloat16, got {q.dtype}."
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"Expected q/k/v to share the same dtype, got "
            f"q={q.dtype}, k={k.dtype}, v={v.dtype}."
        )
    if a.dtype != q.dtype or b.dtype != q.dtype:
        raise ValueError(
            f"Expected a/b to have the same dtype as q, got "
            f"q={q.dtype}, a={a.dtype}, b={b.dtype}."
        )
    if A_log.dtype != torch.float32:
        raise ValueError(f"A_log must be float32, got A_log={A_log.dtype}.")
    if dt_bias.dtype not in _SUPPORTED_DT_BIAS_DTYPES:
        raise ValueError(
            f"dt_bias must be float32 or bfloat16, got dt_bias={dt_bias.dtype}."
        )

    _check_same_device(
        q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
    )

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


def _prepare_state_indices_kernel_arg(
    state_indices: Optional[torch.Tensor],
    *,
    state: torch.Tensor,
    batch_size: int,
    q: torch.Tensor,
    num_v_heads: int,
    head_dim_v: int,
    head_dim_k: int,
) -> Optional[torch.Tensor]:
    if state_indices is None:
        if state.shape[0] != batch_size:
            raise ValueError(
                "state_indices=None uses identity mapping, so state must have "
                f"shape [B={batch_size}, HV={num_v_heads}, V={head_dim_v}, K={head_dim_k}], "
                f"got {tuple(state.shape)}"
            )
        return None

    if state_indices.dim() != 1 or state_indices.shape[0] != batch_size:
        raise ValueError(
            f"state_indices must have shape [B={batch_size}], got {tuple(state_indices.shape)}"
        )
    if state_indices.device != q.device:
        raise ValueError(
            f"Expected state_indices to be on device {q.device}, got {state_indices.device}."
        )
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"state_indices must be int32 or int64, got {state_indices.dtype}."
        )
    return (
        state_indices
        if state_indices.dtype == torch.int32
        else state_indices.to(dtype=torch.int32)
    )


def _gated_delta_rule_decode_pretranspose_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    head_dim_k: int,
    num_v_heads: int,
    head_dim_v: int,
    state_indices: Optional[torch.Tensor],
    scale: Optional[float],
    output: Optional[torch.Tensor],
    use_qk_l2norm: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seq_len != 1:
        raise ValueError(f"VK fp32 decode only supports T=1, got T={seq_len}")
    if state.dtype != torch.float32:
        raise ValueError(f"VK fp32 decode requires float32 state, got {state.dtype}")
    if state.dim() != 4 or state.shape[1:] != (
        num_v_heads,
        head_dim_v,
        head_dim_k,
    ):
        raise ValueError(
            f"Expected state [pool_size, HV={num_v_heads}, V={head_dim_v}, "
            f"K={head_dim_k}] for VK, got {tuple(state.shape)}"
        )
    if state.device != q.device:
        raise ValueError(
            f"Expected state to be on device {q.device}, got {state.device}."
        )
    state_indices_kernel = _prepare_state_indices_kernel_arg(
        state_indices,
        state=state,
        batch_size=batch_size,
        q=q,
        num_v_heads=num_v_heads,
        head_dim_v=head_dim_v,
        head_dim_k=head_dim_k,
    )

    scale_value = head_dim_k**-0.5 if scale is None else float(scale)

    if output is None:
        output = torch.empty(
            (batch_size, num_v_heads, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )

    state_is_contiguous = state.is_contiguous()
    state_kernel = state if state_is_contiguous else state.contiguous()

    gdn_decode_tilelang.run_gated_delta_rule_decode_vk_fp32(
        q=q,
        k=k,
        v=v,
        state=state_kernel,
        state_indices=state_indices_kernel,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        output=output,
        scale=scale_value,
        use_qk_l2norm=use_qk_l2norm,
    )

    if not state_is_contiguous:
        state.copy_(state_kernel)

    return output, state


def _gated_delta_rule_mtp_vk_fp32_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    head_dim_k: int,
    num_v_heads: int,
    head_dim_v: int,
    state_indices: Optional[torch.Tensor],
    scale: Optional[float],
    output: Optional[torch.Tensor],
    intermediate_states_buffer: Optional[torch.Tensor],
    disable_state_update: bool,
    use_qk_l2norm: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if state.device != q.device:
        raise ValueError(
            f"Expected state to be on device {q.device}, got {state.device}."
        )
    if state.dim() != 4 or state.shape[1:] != (
        num_v_heads,
        head_dim_v,
        head_dim_k,
    ):
        raise ValueError(
            f"Expected state [pool_size, HV={num_v_heads}, V={head_dim_v}, "
            f"K={head_dim_k}] for VK MTP, got {tuple(state.shape)}"
        )

    state_indices_kernel = _prepare_state_indices_kernel_arg(
        state_indices,
        state=state,
        batch_size=batch_size,
        q=q,
        num_v_heads=num_v_heads,
        head_dim_v=head_dim_v,
        head_dim_k=head_dim_k,
    )

    if output is None:
        output = torch.empty(
            (batch_size, seq_len, num_v_heads, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )

    if intermediate_states_buffer is not None:
        if intermediate_states_buffer.dim() != 5 or intermediate_states_buffer.shape[
            2:
        ] != (
            num_v_heads,
            head_dim_v,
            head_dim_k,
        ):
            raise ValueError(
                "Expected intermediate_states_buffer shape "
                f"[buffer_size, cache_steps, HV={num_v_heads}, V={head_dim_v}, K={head_dim_k}], "
                f"got {tuple(intermediate_states_buffer.shape)}."
            )
        if intermediate_states_buffer.shape[0] < batch_size:
            raise ValueError(
                "intermediate_states_buffer dim 0 must be at least "
                f"batch_size={batch_size}, got {intermediate_states_buffer.shape[0]}."
            )
        if intermediate_states_buffer.shape[1] < seq_len:
            raise ValueError(
                "intermediate_states_buffer dim 1 must be at least "
                f"seq_len={seq_len}, got {intermediate_states_buffer.shape[1]}."
            )
        if intermediate_states_buffer.dtype != state.dtype:
            raise ValueError(
                "intermediate_states_buffer dtype must match state dtype "
                f"{state.dtype}, got {intermediate_states_buffer.dtype}."
            )
        if intermediate_states_buffer.device != q.device:
            raise ValueError(
                "intermediate_states_buffer must be on the same device as q."
            )
        if not intermediate_states_buffer.is_contiguous():
            raise ValueError("intermediate_states_buffer must be contiguous.")

    state_is_contiguous = state.is_contiguous()
    state_kernel = state if state_is_contiguous else state.contiguous()

    scale_value = head_dim_k**-0.5 if scale is None else float(scale)
    gdn_mtp_tilelang.run_gated_delta_rule_mtp_vk_fp32(
        q=q,
        k=k,
        v=v,
        state=state_kernel,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_indices=state_indices_kernel,
        output=output,
        intermediate_states_buffer=intermediate_states_buffer,
        scale=scale_value,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
    )

    # Match FlashInfer wrapper behavior: update a contiguous working buffer in
    # the kernel, then let PyTorch copy it back into the original strided state.
    if not disable_state_update and not state_is_contiguous:
        state.copy_(state_kernel)

    return output, state


@mate_api
def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    state_layout: str = "VK",
    state_indices: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Unified Gated Delta Rule Decode API.

    Single entry point for decode (``T=1``) and MTP/speculative decode (``T>1``).
    The API layer owns parameter semantics and backend dispatch. The current MATE
    implementation supports the VK-layout + TileLang backends for single-token
    decode with FP32 state and MTP/speculative decode with FP32 or BF16 state.

    Args:
        q (torch.Tensor):
            Query of shape ``[B, T, H, K]``. Must be float16/bfloat16.
        k (torch.Tensor):
            Key of shape ``[B, T, H, K]``. Must match ``q`` in shape and dtype.
        v (torch.Tensor):
            Value of shape ``[B, T, HV, V]``. Must match ``q`` in batch/time dims.
        state (torch.Tensor):
            State ``[B_or_pool, HV, V, K]`` if ``state_layout="VK"``, else
            ``[B_or_pool, HV, K, V]`` if ``state_layout="KV"``.
        A_log (torch.Tensor):
            Log decay of shape ``[HV]``. Must be float32.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, T, HV]``.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``. Must be float32 or bfloat16.
        b (torch.Tensor):
            Update gate of shape ``[B, T, HV]``.
        state_layout (str):
            ``"VK"`` (K-last) or ``"KV"`` (K-major). Default ``"VK"``.
        state_indices (Optional[torch.Tensor]):
            Optional ``[B]`` int32/int64 mapping batch entries to a state pool.
            ``None`` means identity mapping. Negative values are padding rows:
            output is zeroed and state is not read or updated for that batch.
        scale (Optional[float]):
            Query scale. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Optional pre-allocated output tensor of shape ``[B, T, HV, V]``.
        intermediate_states_buffer (Optional[torch.Tensor]):
            Optional rollback buffer for MTP/speculative decode. Its dtype must
            match ``state.dtype``.
        disable_state_update (bool):
            If True, state is treated as read-only on the MTP path.
        use_qk_l2norm (bool):
            Whether to L2-normalize q and k in-kernel. Default True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: ``[B, T, HV, V]``
            - state: Updated state tensor (same storage as input on supported paths)

    Dispatch:
        - ``state_layout="VK"``, ``state.dtype=float32``, ``T=1`` -> TileLang pretranspose fp32 decode
        - ``state_layout="VK"``, ``state.dtype=float32/bfloat16``, ``T>1`` -> TileLang MTP
        - Other combinations currently raise with a clear error

    Note:
        This module follows the unified API responsibilities from FlashInfer:
        parameter semantics and dispatch stay here, while concrete execution
        lives in backend-specific implementation functions.
    """
    B, T, H, K, HV, V = _validate_common_decode_inputs(
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        output=output,
    )

    if state_layout not in ("VK", "KV"):
        raise ValueError(f"state_layout must be 'VK' or 'KV', got {state_layout!r}")

    use_pool = state_indices is not None

    if state_layout == "KV":
        if use_pool:
            raise NotImplementedError(
                "state_indices (pool) is not supported for state_layout='KV' yet"
            )
        if T != 1:
            raise ValueError(f"state_layout='KV' only supports T=1, got T={T}")
        if state.dtype != torch.float32:
            raise ValueError(
                f"state_layout='KV' requires float32 state, got {state.dtype}"
            )
        if state.shape != (B, HV, K, V):
            raise ValueError(
                f"Expected state shape [B={B}, HV={HV}, K={K}, V={V}] for KV layout, got {state.shape}"
            )
        raise NotImplementedError(
            "state_layout='KV' backend is not implemented in MATE yet"
        )

    # state_layout == "VK"
    if state.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"VK layout supports bfloat16 or float32 state, got {state.dtype}"
        )

    if T == 1:
        if state.dtype == torch.bfloat16:
            raise NotImplementedError(
                "VK bfloat16 T=1 decode is not implemented in MATE yet; "
                "the bfloat16 state backend currently supports the MTP path (T>1)."
            )
        if intermediate_states_buffer is not None:
            raise NotImplementedError(
                "VK fp32 T=1 does not support intermediate_states_buffer"
            )
        if disable_state_update:
            raise NotImplementedError(
                "VK fp32 T=1 does not support disable_state_update"
            )
        return _gated_delta_rule_decode_pretranspose_impl(
            q=q,
            k=k,
            v=v,
            state=state,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            batch_size=B,
            seq_len=T,
            head_dim_k=K,
            num_v_heads=HV,
            head_dim_v=V,
            state_indices=state_indices,
            scale=scale,
            output=output,
            use_qk_l2norm=use_qk_l2norm,
        )

    if K != 128 or V != 128:
        raise NotImplementedError(
            f"VK MTP currently supports K=V=128 only, got K={K}, V={V}"
        )

    return _gated_delta_rule_mtp_vk_fp32_impl(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        batch_size=B,
        seq_len=T,
        head_dim_k=K,
        num_v_heads=HV,
        head_dim_v=V,
        state_indices=state_indices,
        scale=scale,
        output=output,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
    )
