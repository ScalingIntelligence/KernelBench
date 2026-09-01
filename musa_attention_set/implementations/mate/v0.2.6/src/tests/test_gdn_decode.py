from __future__ import annotations

import os
import random

import pytest
import torch
import torch.nn.functional as F

import mate
from mate.testing import supported_musa_compute_capability


# Keep the reference path numerically aligned with gdn_unified_tilelang testing.
pytestmark = pytest.mark.usefixtures("disable_mudnn_tf32")


@torch.inference_mode
def decode_delta_rule(
    q: torch.Tensor,  # [B, num_q_heads, K]
    k: torch.Tensor,  # [B, num_k_heads, K]
    v: torch.Tensor,  # [B, num_v_heads, V]
    state: torch.Tensor,  # [B, num_heads, K, V]
    A_log: torch.Tensor,  # [num_heads] - log decay parameter
    a: torch.Tensor,  # [B, num_heads] - input-dependent decay
    dt_bias: torch.Tensor,  # [num_heads] - decay bias
    b: torch.Tensor,  # [B, num_heads] - update gate input
    scale_factor: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_l2_norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for single-step decode with GDN formula.

    Strictly follows the Triton kernel logic from fused_sigmoid_gating_recurrent.py:
        1. Compute g = -exp(A_log) * softplus(a + dt_bias)
        2. Compute beta = sigmoid(b)
        3. Apply L2 norm to q and k (if enabled)
        4. h *= exp(g)                    # Apply decay to state
        5. v_new = v - k^T @ h            # Delta rule (h is [K,V], k is [K])
        6. v_new *= beta                  # Apply update gate
        7. h += k @ v_new^T               # Update state (outer product)
        8. o = q^T @ h                    # Compute output (h is [K,V], q is [K])

    Args:
        q: Query [B, num_q_heads, K]
        k: Key [B, num_k_heads, K]
        v: Value [B, num_v_heads, V]
        state: Input state [B, num_heads, K, V], where num_heads = num_v_heads
        A_log: Log decay parameter [num_heads]
        a: Input-dependent decay [B, num_heads]
        dt_bias: Decay bias [num_heads]
        b: Update gate input [B, num_heads]
        scale_factor: Scale factor for q
        softplus_beta: Beta parameter for softplus activation
        softplus_threshold: Threshold for softplus numerical stability
        use_l2_norm: Whether to apply L2 normalization to q and k
        state_dtype: Storage dtype for the hidden state (read in fp32, stored in this dtype)

    Returns:
        output: [B, num_heads, V]
        new_state: [B, num_heads, K, V]
    """
    B = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    K = q.size(2)
    V = v.size(2)

    # State and output are always based on num_v_heads (matches kernel's HV dimension)
    num_heads = num_v_heads

    device = q.device
    dtype = torch.float32

    # Convert to float32 for computation
    A_log = A_log.to(dtype).to(device)
    a = a.to(dtype).to(device)
    dt_bias = dt_bias.to(dtype).to(device)
    b = b.to(dtype).to(device)

    # ============================================
    # Compute gating values (following Triton kernel exactly)
    # ============================================

    # Step 1: Compute g = -exp(A_log) * softplus(a + dt_bias)
    # Triton kernel lines 100-109
    x = a + dt_bias  # [B, num_heads]
    beta_x = softplus_beta * x

    # Apply softplus with numerical stability
    # softplus(x) = (1/beta) * log(1 + exp(beta*x)) if beta*x <= threshold, else x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )

    # Compute g (log-space decay gate)
    # Triton kernel line 109: b_g = -tl.exp(b_A_log) * softplus_x
    g = -torch.exp(A_log) * softplus_x  # [B, num_heads]

    # Step 2: Compute beta = sigmoid(b)
    # Triton kernel line 112: b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    beta = 1.0 / (1.0 + torch.exp(-b))  # [B, num_heads]

    # Expand heads if needed (for GQA/GVA)
    # The reference works at v_heads level
    # For GQA (num_q_heads > num_v_heads): k and q need to be averaged/pooled per v_head
    # For GVA (num_v_heads > num_q_heads): q and k need to be repeated
    if num_k_heads < num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    if num_q_heads < num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    elif num_q_heads > num_v_heads:
        # GQA: multiple q_heads per v_head, reshape and average
        # [B, num_q_heads, K] -> [B, num_v_heads, num_q_heads//num_v_heads, K]
        q = q.reshape(B, num_v_heads, num_q_heads // num_v_heads, K).mean(dim=2)
        if num_k_heads == num_q_heads:
            k = k.reshape(B, num_v_heads, num_k_heads // num_v_heads, K).mean(dim=2)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    state = state.to(dtype)

    # Apply L2 normalization if requested
    if use_l2_norm:
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

    # Apply scale to q
    q = q * scale_factor

    # ============================================
    # Process each batch and head
    # ============================================
    new_state = torch.zeros(B, num_heads, K, V, device=device, dtype=state_dtype)
    output = torch.zeros(B, num_heads, V, device=device, dtype=dtype)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            # Get current vectors
            q_h = q[b_idx, h_idx]  # [K]
            k_h = k[b_idx, h_idx]  # [K]
            v_h = v[b_idx, h_idx]  # [V]
            h_state = (
                state[b_idx, h_idx].clone().to(torch.float32)
            )  # [K, V] read as fp32

            # Get gating values for this batch and head
            g_val = g[b_idx, h_idx]  # scalar
            beta_val = beta[b_idx, h_idx]  # scalar

            # ============================================
            # Recurrent update (following Triton kernel lines 121-134)
            # ============================================

            # Step 1: Apply gating to hidden state: h *= exp(g)
            # Triton kernel line 122: b_h *= tl.exp(b_g)
            h_state = h_state * torch.exp(g_val)

            # Step 2: Delta rule: v -= sum(h * k, dim=0)
            # Triton kernel line 125: b_v -= tl.sum(b_h * b_k[:, None], 0)
            # Triton: b_h is [BK, BV], b_k is [BK]
            # b_k[:, None] makes it [BK, 1]
            # b_h * b_k[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: k^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            # Match the kernel's explicit FP32 reduction for k * state rather
            # than dispatching a backend GEMV with an implementation-defined
            # reduction order.
            v_new = v_h - torch.sum(h_state * k_h[:, None], dim=0)

            # Step 3: Apply beta gating: v *= beta
            # Triton kernel line 128: b_v *= b_beta
            v_new = v_new * beta_val

            # Step 4: Update hidden state: h += k[:, None] * v[None, :]
            # Triton kernel line 131: b_h += b_k[:, None] * b_v[None, :]
            # Triton: [BK, BV] += [BK, 1] * [1, BV]
            # This is outer product: k @ v^T
            # [K, V] += [K, 1] @ [1, V]
            # Keep the reference as element-wise FP32 arithmetic.  Expressing
            # this outer product with ``@`` dispatches to MUBLAS, whose TF32
            # override changes the reference values.
            h_state = h_state + k_h[:, None] * v_new[None, :]

            # Step 5: Compute output: o = sum(h * q, dim=0)
            # Triton kernel line 134: b_o = tl.sum(b_h * b_q[:, None], 0)
            # Triton: b_h is [BK, BV], b_q is [BK]
            # b_q[:, None] makes it [BK, 1]
            # b_h * b_q[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: q^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            # Keep the reference reduction explicit instead of dispatching a
            # MUSA GEMV.  The kernel accumulates q * state in a fixed FP32
            # warp-reduction order; using ``q_h @ h_state`` can select a
            # backend reduction with a different order near low-precision
            # output rounding boundaries.
            output[b_idx, h_idx] = torch.sum(h_state * q_h[:, None], dim=0)

            # Store updated state (cast back to state_dtype)
            new_state[b_idx, h_idx] = h_state.to(state_dtype)

    return output, new_state


def _get_default_seed() -> int:
    return int(os.environ.get("SEED", "0"))


def _get_runtime_device(*, allow_skip: bool) -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    if allow_skip:
        pytest.skip("MUSA is not available")
    raise RuntimeError("MUSA is not available")


def _manual_seed_device(seed: int | None, device_type: str) -> None:
    if seed is None:
        return
    if device_type == "musa":
        torch.musa.manual_seed(seed)


def _synchronize_device(device_type: str) -> None:
    if device_type == "musa":
        torch.musa.synchronize()


def _make_sglang_split_qkv_views(
    *,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dim = num_q_heads * head_size
    k_dim = num_k_heads * head_size
    v_dim = num_v_heads * head_size
    mixed_qkv = torch.empty(
        batch_size, q_dim + k_dim + v_dim, dtype=dtype, device=device
    )

    q_dense = torch.randn(
        batch_size, 1, num_q_heads, head_size, dtype=dtype, device=device
    )
    k_dense = torch.randn(
        batch_size, 1, num_k_heads, head_size, dtype=dtype, device=device
    )
    k_dense = torch.nn.functional.normalize(k_dense, p=2.0, dim=-1)
    v_dense = torch.randn(
        batch_size, 1, num_v_heads, head_size, dtype=dtype, device=device
    )

    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    q_flat.copy_(q_dense.view(batch_size, q_dim))
    k_flat.copy_(k_dense.view(batch_size, k_dim))
    v_flat.copy_(v_dense.view(batch_size, v_dim))

    q = q_flat.view(batch_size, 1, num_q_heads, head_size)
    k = k_flat.view(batch_size, 1, num_k_heads, head_size)
    v = v_flat.view(batch_size, 1, num_v_heads, head_size)

    fused_stride = q_dim + k_dim + v_dim
    assert q.stride()[1:] == (q_dim, head_size, 1)
    assert k.stride()[1:] == (k_dim, head_size, 1)
    assert v.stride()[1:] == (v_dim, head_size, 1)
    if batch_size > 1:
        assert q.stride()[0] == fused_stride
        assert k.stride()[0] == fused_stride
        assert v.stride()[0] == fused_stride
        assert not q.is_contiguous()
        assert not k.is_contiguous()
        assert not v.is_contiguous()
    return q, k, v


def _make_split_gate_views(
    *,
    batch_size: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixed_ab = torch.empty(batch_size, 1, num_heads * 2, dtype=dtype, device=device)
    a_view, b_view = torch.split(mixed_ab, [num_heads, num_heads], dim=-1)

    a_dense = torch.randn(batch_size, 1, num_heads, dtype=dtype, device=device) * 0.1
    b_dense = torch.randn(batch_size, 1, num_heads, dtype=dtype, device=device)
    a_view.copy_(a_dense)
    b_view.copy_(b_dense)

    assert a_view.stride() == (num_heads * 2, num_heads * 2, 1)
    assert b_view.stride() == (num_heads * 2, num_heads * 2, 1)
    if batch_size > 1:
        assert not a_view.is_contiguous()
        assert not b_view.is_contiguous()
    return a_view, b_view


def _make_strided_decode_output(
    *,
    batch_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    workspace = torch.empty(
        batch_size,
        1,
        num_heads,
        head_size * 2,
        dtype=dtype,
        device=device,
    )
    output = workspace[..., :head_size]
    assert output.shape == (batch_size, 1, num_heads, head_size)
    assert not output.is_contiguous()
    return output


def _assert_decode_matches_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_state_ref: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b_tensor: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
    output: torch.Tensor | None = None,
    output_rtol: float | None = None,
) -> None:
    state = input_state_ref.transpose(-2, -1).contiguous()
    state_ptr = state.data_ptr()
    out, returned_state = mate.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        state_layout="VK",
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
    )
    _synchronize_device(q.device.type)

    assert state.data_ptr() == state_ptr
    assert returned_state.data_ptr() == state_ptr
    if output is not None:
        assert out.data_ptr() == output.data_ptr()

    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state_ref,
        A_log=A_log.float(),
        a=a.squeeze(1).float(),
        dt_bias=dt_bias.float(),
        b=b_tensor.squeeze(1).float(),
        scale_factor=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=use_qk_l2norm,
    )
    ref_o = ref_o.to(q.dtype)
    ref_state = ref_state.transpose(-2, -1).contiguous()

    default_atol = 1e-3 if q.dtype == torch.float16 else 5e-3
    default_rtol = 1e-3 if q.dtype == torch.float16 else 5e-3
    torch.testing.assert_close(
        out.squeeze(1),
        ref_o,
        atol=default_atol,
        rtol=default_rtol if output_rtol is None else output_rtol,
    )
    torch.testing.assert_close(
        returned_state,
        ref_state,
        atol=default_atol,
        rtol=default_rtol,
    )


def _make_decode_pool_inputs(
    *,
    batch_size: int,
    pool_size: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
    head_size = 128
    dtype_torch = torch.bfloat16
    device = _get_runtime_device(allow_skip=True)
    _manual_seed_device(seed, device.type)

    q = torch.randn(
        batch_size, 1, num_q_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_k_heads, head_size, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_v_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    state_pool = torch.randn(
        pool_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )

    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, num_v_heads, dtype=dtype_torch, device=device) * 0.1
    b_tensor = torch.randn(
        batch_size,
        1,
        num_v_heads,
        dtype=dtype_torch,
        device=device,
    )

    return q, k, v, state_pool, A_log, a, dt_bias, b_tensor


def _assert_decode_pool_matches_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, _, _ = q.shape
    _, _, HV, V = v.shape
    assert T == 1
    scale = 1.0

    state_under_test = state_pool.clone()
    state_ptr = state_under_test.data_ptr()
    out, returned_state = mate.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=state_under_test,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        state_layout="VK",
        state_indices=state_indices,
        scale=scale,
        use_qk_l2norm=True,
    )
    _synchronize_device(q.device.type)

    assert returned_state.data_ptr() == state_ptr

    ref_out = torch.zeros(B, T, HV, V, dtype=q.dtype, device=q.device)
    ref_state = state_pool.clone()
    valid_mask = state_indices >= 0
    if valid_mask.any():
        valid_batch = torch.where(valid_mask)[0]
        valid_slots = state_indices[valid_batch].long()
        gathered_state_kv = state_pool[valid_slots].transpose(-2, -1).contiguous()

        ref_o, ref_state_kv = decode_delta_rule(
            q[valid_batch].squeeze(1).float(),
            k[valid_batch].squeeze(1).float(),
            v[valid_batch].squeeze(1).float(),
            gathered_state_kv,
            A_log=A_log.float(),
            a=a[valid_batch].squeeze(1).float(),
            dt_bias=dt_bias.float(),
            b=b_tensor[valid_batch].squeeze(1).float(),
            scale_factor=scale,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            use_l2_norm=True,
        )
        ref_out[valid_batch, 0] = ref_o.to(q.dtype)
        ref_state[valid_slots] = ref_state_kv.transpose(-2, -1).contiguous()

    atol = 5e-3
    rtol = 5e-3
    out_4d = out.unsqueeze(1) if out.dim() == 3 else out
    torch.testing.assert_close(out_4d, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(state_under_test, ref_state, atol=atol, rtol=rtol)

    invalid_mask = state_indices < 0
    if invalid_mask.any():
        assert torch.all(out_4d[invalid_mask] == 0)

    valid_slots = state_indices[state_indices >= 0].long().unique()
    untouched = torch.ones(
        state_pool.shape[0],
        dtype=torch.bool,
        device=state_pool.device,
    )
    if valid_slots.numel() > 0:
        untouched[valid_slots] = False
    assert torch.equal(state_under_test[untouched], state_pool[untouched])
    return out_4d, state_under_test


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_decode_kernel_pool_state_indices_pretranspose(
    batch_size: int,
    seed: int = _get_default_seed(),
) -> None:
    pool_multiplier = 3
    pool_size = batch_size * pool_multiplier
    q, k, v, state_pool, A_log, a, dt_bias, b_tensor = _make_decode_pool_inputs(
        batch_size=batch_size,
        pool_size=pool_size,
        seed=seed,
    )
    state_indices = (
        torch.arange(batch_size, dtype=torch.int32, device=q.device) * pool_multiplier
    )

    _assert_decode_pool_matches_reference(
        q=q,
        k=k,
        v=v,
        state_pool=state_pool,
        state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_tensor=b_tensor,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch_size", [4, 8, 32])
def test_decode_kernel_pool_negative_state_indices_pretranspose(
    batch_size: int,
    seed: int = _get_default_seed(),
) -> None:
    pool_size = batch_size * 2
    q, k, v, state_pool, A_log, a, dt_bias, b_tensor = _make_decode_pool_inputs(
        batch_size=batch_size,
        pool_size=pool_size,
        seed=seed,
    )
    state_indices = torch.randperm(pool_size, device=q.device)[:batch_size].to(
        torch.int32
    )
    state_indices[1::2] = -1

    out, _ = _assert_decode_pool_matches_reference(
        q=q,
        k=k,
        v=v,
        state_pool=state_pool,
        state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_tensor=b_tensor,
    )
    assert torch.all(out[state_indices < 0] == 0)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
def test_decode_kernel_pool_all_padding_state_indices_pretranspose(
    batch_size: int,
    seed: int = _get_default_seed(),
) -> None:
    pool_size = batch_size * 2
    q, k, v, state_pool, A_log, a, dt_bias, b_tensor = _make_decode_pool_inputs(
        batch_size=batch_size,
        pool_size=pool_size,
        seed=seed,
    )
    state_indices = torch.full(
        (batch_size,),
        -1,
        dtype=torch.int32,
        device=q.device,
    )

    out, state_under_test = _assert_decode_pool_matches_reference(
        q=q,
        k=k,
        v=v,
        state_pool=state_pool,
        state_indices=state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_tensor=b_tensor,
    )
    assert torch.all(out == 0)
    assert torch.equal(state_under_test, state_pool)


def _test_decode_kernel_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
    compare_values: bool = False,
    allow_skip: bool = False,
):
    del alpha

    random.seed(seed)
    torch.random.manual_seed(seed)

    num_sab_heads = num_v_heads
    dtype_torch = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = _get_runtime_device(allow_skip=allow_skip)
    _manual_seed_device(seed, device.type)

    q = torch.randn(
        batch_size, 1, num_q_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_k_heads, head_size, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_v_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    input_state_ref = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=kv_dtype,
        device=device,
    )
    input_state_kernel = input_state_ref.transpose(-2, -1).contiguous()

    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = (
        torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
        * 0.1
    )

    if beta:
        b_tensor = torch.randn(
            batch_size,
            1,
            num_sab_heads,
            dtype=dtype_torch,
            device=device,
        )
    else:
        b_tensor = (
            torch.ones(
                batch_size,
                1,
                num_sab_heads,
                dtype=dtype_torch,
                device=device,
            )
            * 10.0
        )

    our_state = input_state_kernel.clone()
    our_state_ptr = our_state.data_ptr()
    our_o, returned_state = mate.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=our_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        state_layout="VK",
        scale=scale,
        use_qk_l2norm=True,
    )
    _synchronize_device(device.type)
    assert our_state.data_ptr() == our_state_ptr
    assert returned_state.data_ptr() == our_state_ptr
    our_state = returned_state
    our_o = our_o.squeeze(1)

    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state_ref,
        A_log=A_log.float(),
        a=a.squeeze(1).float(),
        dt_bias=dt_bias.float(),
        b=b_tensor.squeeze(1).float(),
        scale_factor=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,
    )

    ref_o = ref_o.to(dtype_torch)
    ref_state = ref_state.transpose(-2, -1).contiguous().to(kv_dtype)

    assert our_o.shape == ref_o.shape
    assert our_state.shape == ref_state.shape

    if compare_values:
        atol_o = 5e-3
        rtol_o = 5e-3
        atol_kv = 5e-3
        rtol_kv = 5e-3

        torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
        torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)

    print(f"decode kernel test passed (batch={batch_size}, dtype={dtype})")


def _test_decode_kernel_bitwise_stable_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    beta: bool,
    seed: int | None = None,
    repeats: int = 50,
    allow_skip: bool = False,
):
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_sab_heads = num_v_heads
    dtype_torch = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = _get_runtime_device(allow_skip=allow_skip)
    _manual_seed_device(seed, device.type)

    q = torch.randn(
        batch_size, 1, num_q_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_k_heads, head_size, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_v_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    input_state_ref = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=kv_dtype,
        device=device,
    )
    input_state_kernel = input_state_ref.transpose(-2, -1).contiguous()

    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = (
        torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
        * 0.1
    )

    if beta:
        b_tensor = torch.randn(
            batch_size,
            1,
            num_sab_heads,
            dtype=dtype_torch,
            device=device,
        )
    else:
        b_tensor = (
            torch.ones(
                batch_size,
                1,
                num_sab_heads,
                dtype=dtype_torch,
                device=device,
            )
            * 10.0
        )

    baseline_o = None
    baseline_state = None

    for repeat_idx in range(repeats):
        state = input_state_kernel.clone()
        state_ptr = state.data_ptr()
        out, returned_state = mate.gated_delta_rule_decode(
            q=q,
            k=k,
            v=v,
            state=state,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b_tensor,
            state_layout="VK",
            scale=scale,
            use_qk_l2norm=True,
        )
        _synchronize_device(device.type)
        assert state.data_ptr() == state_ptr
        assert returned_state.data_ptr() == state_ptr

        out = out.squeeze(1)

        if repeat_idx == 0:
            baseline_o = out.clone()
            baseline_state = returned_state.clone()
            continue

        if not torch.equal(out, baseline_o):
            diff = (out.float() - baseline_o.float()).abs().max().item()
            raise AssertionError(
                f"Output is not bitwise stable at repeat={repeat_idx}, max_abs_diff={diff}"
            )
        if not torch.equal(returned_state, baseline_state):
            diff = (returned_state - baseline_state).abs().max().item()
            raise AssertionError(
                f"State is not bitwise stable at repeat={repeat_idx}, max_abs_diff={diff}"
            )

    print(
        f"decode kernel bitwise stability test passed "
        f"(batch={batch_size}, dtype={dtype}, repeats={repeats})"
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_basic_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int = _get_default_seed(),
) -> None:
    _test_decode_kernel_pretranspose(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        scale=scale,
        alpha=alpha,
        beta=beta,
        seed=seed,
        compare_values=True,
        allow_skip=True,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("dt_bias_dtype", ["float32", "bfloat16"])
def test_decode_kernel_sglang_split_qkv_pretranspose(
    dtype: str,
    dt_bias_dtype: str,
    seed: int = _get_default_seed(),
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)

    batch_size = 32
    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
    head_size = 128
    scale = 1.0

    dtype_torch = getattr(torch, dtype)
    device = _get_runtime_device(allow_skip=True)
    _manual_seed_device(seed, device.type)

    q, k, v = _make_sglang_split_qkv_views(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        dtype=dtype_torch,
        device=device,
    )

    input_state_ref = torch.randn(
        batch_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )

    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = (
        torch.randn(num_v_heads, dtype=getattr(torch, dt_bias_dtype), device=device)
        * 0.1
    )
    a = torch.randn(batch_size, 1, num_v_heads, dtype=dtype_torch, device=device) * 0.1
    b_tensor = torch.randn(
        batch_size,
        1,
        num_v_heads,
        dtype=dtype_torch,
        device=device,
    )

    _assert_decode_matches_reference(
        q=q,
        k=k,
        v=v,
        input_state_ref=input_state_ref,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_tensor=b_tensor,
        scale=scale,
        output=None,
        use_qk_l2norm=True,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("use_qk_l2norm", [True, False])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_decode_kernel_strided_inputs_and_output_pretranspose(
    dtype: str,
    batch_size: int,
    use_qk_l2norm: bool,
    seed: int = _get_default_seed(),
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_q_heads = 8
    num_k_heads = 8
    num_v_heads = 32
    head_size = 128
    scale = 1.0

    dtype_torch = getattr(torch, dtype)
    device = _get_runtime_device(allow_skip=True)
    _manual_seed_device(seed, device.type)

    q, k, v = _make_sglang_split_qkv_views(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        dtype=dtype_torch,
        device=device,
    )
    a, b_tensor = _make_split_gate_views(
        batch_size=batch_size,
        num_heads=num_v_heads,
        dtype=dtype_torch,
        device=device,
    )
    output = _make_strided_decode_output(
        batch_size=batch_size,
        num_heads=num_v_heads,
        head_size=head_size,
        dtype=dtype_torch,
        device=device,
    )

    input_state_ref = torch.randn(
        batch_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1

    _assert_decode_matches_reference(
        q=q,
        k=k,
        v=v,
        input_state_ref=input_state_ref,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b_tensor=b_tensor,
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        # Allow one bf16-step rounding-boundary difference for strided output.
        output_rtol=8e-3 if dtype_torch == torch.bfloat16 else None,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_bitwise_stable_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    beta: bool,
    seed: int = _get_default_seed(),
) -> None:
    _test_decode_kernel_bitwise_stable_pretranspose(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        scale=scale,
        beta=beta,
        seed=seed,
        allow_skip=True,
    )


def main() -> None:
    _test_decode_kernel_bitwise_stable_pretranspose(
        dtype="bfloat16",
        batch_size=8,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        beta=True,
        seed=_get_default_seed(),
    )
    _test_decode_kernel_pretranspose(
        dtype="bfloat16",
        batch_size=8,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        alpha=True,
        beta=True,
        seed=_get_default_seed(),
        compare_values=True,
    )


if __name__ == "__main__":
    main()
