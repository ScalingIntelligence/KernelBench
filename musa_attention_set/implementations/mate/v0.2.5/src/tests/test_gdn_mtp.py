from __future__ import annotations

import math
import os
import random
from typing import Optional

import pytest
import torch

import mate
from mate.testing import supported_musa_compute_capability


pytestmark = pytest.mark.usefixtures("disable_mudnn_tf32")


@torch.inference_mode
def verify_delta_rule(
    q: torch.Tensor,  # [B, T, num_q_heads, K]
    k: torch.Tensor,  # [B, T, num_k_heads, K]
    v: torch.Tensor,  # [B, T, num_v_heads, V]
    state: torch.Tensor,  # [B, num_heads, K, V]
    A_log: torch.Tensor,  # [num_heads] - log decay parameter
    a: torch.Tensor,  # [B, T, num_heads] - input-dependent decay
    dt_bias: torch.Tensor,  # [num_heads] - decay bias
    b: torch.Tensor,  # [B, T, num_heads] - update gate input
    scale_factor: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_l2_norm: bool = True,
    cache_intermediate_states: bool = False,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Reference implementation for multi-token (verify mode) delta rule.

    Processes T tokens sequentially, updating the state after each token.
    Optionally caches intermediate states for rollback in speculative decoding.

    Args:
        q: Query tensor [B, T, num_q_heads, K]
        k: Key tensor [B, T, num_k_heads, K]
        v: Value tensor [B, T, num_v_heads, V]
        state: Initial state tensor [B, num_heads, K, V]
        A_log: Log decay parameter [num_heads]
        a: Input-dependent decay [B, T, num_heads]
        dt_bias: Decay bias [num_heads]
        b: Update gate input [B, T, num_heads]
        scale_factor: Scaling factor for queries
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus approximation
        use_l2_norm: Whether to apply L2 normalization
        cache_intermediate_states: Whether to cache state at each time step
        state_dtype: Storage dtype for the hidden state (read in fp32, stored in this dtype)

    Returns:
        output: Output tensor [B, T, num_heads, V]
        new_state: Final state tensor [B, num_heads, K, V]
        intermediate_states: Cached intermediate states [B, T, num_heads, K, V] or None
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = state.shape[1]

    # Handle GQA/GVA: expand or average heads
    if num_q_heads != num_heads:
        # Expand q heads to match num_heads (num_v_heads)
        assert num_heads % num_q_heads == 0
        repeat_factor = num_heads // num_q_heads
        q = q.repeat_interleave(repeat_factor, dim=2)  # [B, T, num_heads, K]

    if num_k_heads != num_heads:
        # Expand k heads to match num_heads (num_v_heads)
        assert num_heads % num_k_heads == 0
        repeat_factor = num_heads // num_k_heads
        k = k.repeat_interleave(repeat_factor, dim=2)  # [B, T, num_heads, K]

    # Convert to float32 for computation
    q = q.float()
    k = k.float()
    v = v.float()
    state = state.float()
    A_log = A_log.float()
    a = a.float()
    dt_bias = dt_bias.float()
    b = b.float()

    # Pre-compute gating values for all time steps
    # Shape: [B, T, num_heads]
    x = a + dt_bias.unsqueeze(0).unsqueeze(0)  # [B, T, num_heads]
    beta_x = softplus_beta * x

    # Softplus with threshold
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )

    # Compute g (decay factor, already includes exp)
    g = torch.exp(
        -torch.exp(A_log.unsqueeze(0).unsqueeze(0)) * softplus_x
    )  # [B, T, num_heads]

    # Compute beta (update gate)
    beta = 1.0 / (1.0 + torch.exp(-b))  # [B, T, num_heads]

    # Apply L2 normalization if needed
    if use_l2_norm:
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

    # Apply scaling to q
    q = q * scale_factor

    # Initialize output and intermediate states
    output = torch.zeros(B, T, num_heads, V, dtype=torch.float32, device=q.device)
    current_state = state.clone().to(
        state_dtype
    )  # [B, num_heads, K, V] stored in state_dtype

    if cache_intermediate_states:
        intermediate_states = torch.zeros(
            B, T, num_heads, K, V, dtype=state_dtype, device=q.device
        )
    else:
        intermediate_states = None

    # Process each time step sequentially
    for t in range(T):
        q_t = q[:, t]  # [B, num_heads, K]
        k_t = k[:, t]  # [B, num_heads, K]
        v_t = v[:, t]  # [B, num_heads, V]
        g_t = g[:, t]  # [B, num_heads]
        beta_t = beta[:, t]  # [B, num_heads]

        # Process each batch and head
        for b_idx in range(B):
            for h_idx in range(num_heads):
                q_h = q_t[b_idx, h_idx]  # [K]
                k_h = k_t[b_idx, h_idx]  # [K]
                v_h = v_t[b_idx, h_idx]  # [V]
                h_state = (
                    current_state[b_idx, h_idx].clone().to(torch.float32)
                )  # [K, V] read as fp32
                g_val = g_t[b_idx, h_idx]
                beta_val = beta_t[b_idx, h_idx]

                # Recurrent update (following Triton kernel)
                # 1. Apply decay
                h_state = h_state * g_val

                # 2. Compute prediction error: v - k^T @ h
                v_pred = k_h @ h_state  # [K] @ [K, V] = [V]
                v_new = v_h - v_pred

                # 3. Apply gating
                v_new = v_new * beta_val

                # 4. Update state: h = h + k ⊗ v_new
                h_state = h_state + k_h.unsqueeze(1) @ v_new.unsqueeze(
                    0
                )  # [K, V] + [K, 1] @ [1, V]

                # 5. Compute output: o = q^T @ h
                output[b_idx, t, h_idx] = q_h @ h_state  # [K] @ [K, V] = [V]

                # Update current state (cast back to state_dtype)
                current_state[b_idx, h_idx] = h_state.to(state_dtype)

                # Cache intermediate state if requested
                if cache_intermediate_states:
                    intermediate_states[b_idx, t, h_idx] = h_state.to(state_dtype)

    return output, current_state, intermediate_states


@torch.inference_mode()
def _flashinfer_reference_with_state_indices(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state_vk: torch.Tensor,
    state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    use_qk_l2norm: bool,
    state_dtype: torch.dtype = torch.float32,
    intermediate_pool_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T_len, _, K = q.shape
    _, _, HV, V = v.shape
    pool_size = state_vk.shape[0]
    device = q.device

    output = torch.zeros(B, T_len, HV, V, dtype=torch.float32, device=device)
    final_pool_kv = state_vk.transpose(-2, -1).clone().contiguous().to(state_dtype)
    intermediate_size = (
        pool_size if intermediate_pool_size is None else intermediate_pool_size
    )
    intermediate_kv = (
        torch.zeros(
            intermediate_size, T_len, HV, K, V, dtype=state_dtype, device=device
        )
        if cache_intermediate_states
        else None
    )

    for batch_idx in range(B):
        slot = int(state_indices[batch_idx].item())
        if slot < 0:
            continue

        ref_out, ref_state, ref_intermediate = verify_delta_rule(
            q=q[batch_idx : batch_idx + 1],
            k=k[batch_idx : batch_idx + 1],
            v=v[batch_idx : batch_idx + 1],
            state=final_pool_kv[slot : slot + 1],
            A_log=A_log,
            a=a[batch_idx : batch_idx + 1],
            dt_bias=dt_bias,
            b=b[batch_idx : batch_idx + 1],
            scale_factor=scale,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            use_l2_norm=use_qk_l2norm,
            cache_intermediate_states=cache_intermediate_states,
            state_dtype=state_dtype,
        )
        output[batch_idx].copy_(ref_out[0])
        if not disable_state_update:
            final_pool_kv[slot].copy_(ref_state[0])
        if intermediate_kv is not None:
            assert ref_intermediate is not None
            intermediate_kv[batch_idx].copy_(ref_intermediate[0])

    return (
        output,
        final_pool_kv.transpose(-2, -1).contiguous(),
        None
        if intermediate_kv is None
        else intermediate_kv.transpose(-2, -1).contiguous(),
    )


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


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[dtype]


def _tolerances(dtype: str) -> tuple[float, float, float, float]:
    if dtype == "bfloat16":
        return 1e-2, 1e-2, 1e-2, 1e-2
    return 1e-3, 1e-3, 1e-3, 1e-3


def _state_tolerances(dtype: str, state_dtype: torch.dtype) -> tuple[float, float]:
    _, _, atol_s, rtol_s = _tolerances(dtype)
    if state_dtype == torch.bfloat16:
        return 2e-2, 1e-2
    return atol_s, rtol_s


def _output_tolerances(dtype: str, state_dtype: torch.dtype) -> tuple[float, float]:
    atol_o, rtol_o, _, _ = _tolerances(dtype)
    if state_dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return atol_o, rtol_o


def _prepare_initial_state(
    *,
    pool_size: int,
    num_v_heads: int,
    head_size: int,
    state_dtype: torch.dtype,
    device: torch.device,
    use_noncontiguous_state: bool,
) -> torch.Tensor:
    if use_noncontiguous_state:
        return torch.randn(
            pool_size,
            num_v_heads,
            head_size,
            head_size,
            dtype=state_dtype,
            device=device,
        ).transpose(-2, -1)
    return torch.randn(
        pool_size,
        num_v_heads,
        head_size,
        head_size,
        dtype=state_dtype,
        device=device,
    )


def _make_strided_mtp_output(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    workspace = torch.empty(
        batch_size,
        seq_len,
        num_heads,
        head_size * 2,
        dtype=dtype,
        device=device,
    )
    output = workspace[..., :head_size]
    assert output.shape == (batch_size, seq_len, num_heads, head_size)
    assert not output.is_contiguous()
    return output


def _make_strided_random(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    scale: float = 0.1,
) -> torch.Tensor:
    workspace = torch.randn(
        *shape[:-1],
        shape[-1] * 2,
        dtype=dtype,
        device=device,
    )
    workspace.mul_(scale)
    tensor = workspace[..., : shape[-1]]
    assert tensor.shape == shape
    assert not tensor.is_contiguous()
    return tensor


def _make_random_empty_strided(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    scale: float = 0.1,
) -> torch.Tensor:
    tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
    tensor.normal_()
    tensor.mul_(scale)
    assert tensor.shape == shape
    assert tensor.stride() == stride
    return tensor


def _assert_close_large_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    msg: str,
    timestep_dim: int | None = None,
):
    """Manual assert_close for large tensors that avoids RuntimeError in error formatting.

    torch.testing.assert_close crashes with RuntimeError when trying to format
    error messages for tensors with >1B elements. This function computes the
    comparison manually and reports per-timestep error diagnostics on failure.
    """
    # Compare per-slice to avoid allocating huge temporary tensors
    if timestep_dim is not None and actual.ndim > timestep_dim:
        T = actual.shape[timestep_dim]
        per_t_stats = []
        any_violation = False
        for t in range(T):
            diff_t = (
                actual.select(timestep_dim, t).float()
                - expected.select(timestep_dim, t).float()
            ).abs()
            tol_t = atol + rtol * expected.select(timestep_dim, t).float().abs()
            violations_t = diff_t > tol_t
            count = violations_t.sum().item()
            total = violations_t.numel()
            per_t_stats.append(
                (t, diff_t.max().item(), diff_t.mean().item(), count, total)
            )
            if count > 0:
                any_violation = True
            del diff_t, tol_t, violations_t

        if not any_violation:
            return

        lines = [msg]
        for t, t_max, t_mean, t_count, t_total in per_t_stats:
            lines.append(
                f"  t={t}: max_abs={t_max:.6f}, mean={t_mean:.6f}, "
                f"violations={t_count}/{t_total} ({100 * t_count / t_total:.4f}%)"
            )
        lines.append(f"  Tolerances: atol={atol}, rtol={rtol}")
        raise AssertionError("\n".join(lines))
    else:
        diff = (actual.float() - expected.float()).abs()
        tol = atol + rtol * expected.float().abs()
        violations = diff > tol
        if not violations.any():
            return
        num_violations = violations.sum().item()
        total = violations.numel()
        raise AssertionError(
            f"{msg}\n"
            f"  Max abs error: {diff.max().item():.6f}, "
            f"Violations: {num_violations}/{total} ({100 * num_violations / total:.4f}%), "
            f"Tolerances: atol={atol}, rtol={rtol}"
        )


def _run_flashinfer_mtp_case(
    *,
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    cache_intermediate_states: bool = True,
    disable_state_update: bool = True,
    seed: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device = _get_runtime_device(allow_skip=True)
    _manual_seed_device(seed, device.type)
    torch_dtype = _torch_dtype(dtype)
    scale_value = 1.0 / math.sqrt(head_size) if scale == "auto" else float(scale)

    B = batch_size
    T_len = seq_len
    H = num_q_heads
    HK = num_k_heads
    HV = num_v_heads
    K = head_size
    V = head_size

    q = torch.randn(B, T_len, H, K, dtype=torch_dtype, device=device) * 0.1
    k = torch.randn(B, T_len, HK, K, dtype=torch_dtype, device=device) * 0.1
    v = torch.randn(B, T_len, HV, V, dtype=torch_dtype, device=device) * 0.1

    A_log = (
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1
        if alpha
        else torch.zeros(HV, dtype=torch.float32, device=device)
    )
    dt_bias = (
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1
        if alpha
        else torch.zeros(HV, dtype=torch.float32, device=device)
    )
    a = (
        torch.randn(B, T_len, HV, dtype=torch_dtype, device=device) * 0.1
        if alpha
        else torch.zeros(B, T_len, HV, dtype=torch_dtype, device=device)
    )
    b_tensor = (
        torch.randn(B, T_len, HV, dtype=torch_dtype, device=device) * 0.1
        if beta
        else torch.zeros(B, T_len, HV, dtype=torch_dtype, device=device)
    )

    initial_state = torch.randn(B, HV, V, K, dtype=torch.float32, device=device) * 0.01
    initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)
    intermediate_states_buffer = (
        torch.zeros(B, T_len, HV, V, K, dtype=torch.float32, device=device)
        if cache_intermediate_states
        else None
    )

    state_ref_input = initial_state.clone()
    output_kernel, final_state_kernel = mate.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=initial_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        state_layout="VK",
        state_indices=initial_state_indices,
        scale=scale_value,
        output=None,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=disable_state_update,
        use_qk_l2norm=True,
    )
    _synchronize_device(device.type)

    input_state_ref = state_ref_input.transpose(-2, -1).contiguous()
    output_ref, final_state_ref, intermediate_states_ref = verify_delta_rule(
        q=q,
        k=k,
        v=v,
        state=input_state_ref,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        scale_factor=scale_value,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,
        cache_intermediate_states=cache_intermediate_states,
    )

    atol_o, rtol_o, atol_s, rtol_s = _tolerances(dtype)
    torch.testing.assert_close(
        output_kernel.float(),
        output_ref.float(),
        atol=atol_o,
        rtol=rtol_o,
        msg=f"Output mismatch for FlashInfer MTP parity (B={B}, T={T_len}, dtype={dtype})",
    )

    if cache_intermediate_states and intermediate_states_buffer is not None:
        # Use manual comparison to avoid RuntimeError from torch.testing.assert_close
        # when formatting error messages for tensors with >1B elements (e.g. [512, 5, 32, 128, 128])
        _assert_close_large_tensor(
            intermediate_states_buffer.transpose(-2, -1).float(),
            intermediate_states_ref.float(),
            atol=atol_s,
            rtol=rtol_s,
            msg=(
                "Intermediate states mismatch for FlashInfer MTP parity "
                f"(B={B}, T={T_len}, dtype={dtype})"
            ),
            timestep_dim=1,
        )

    if not disable_state_update:
        torch.testing.assert_close(
            final_state_kernel.float(),
            final_state_ref.transpose(-2, -1).contiguous().float(),
            atol=atol_s,
            rtol=rtol_s,
            msg=f"Final state mismatch for FlashInfer MTP parity (B={B}, T={T_len}, dtype={dtype})",
        )


def _run_flashinfer_mtp_bitwise_case(
    *,
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int,
    scale: float | str,
    cache_intermediate_states: bool,
    seed: int,
    repeats: int = 20,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device = _get_runtime_device(allow_skip=True)
    _manual_seed_device(seed, device.type)
    torch_dtype = _torch_dtype(dtype)
    scale_value = 1.0 / math.sqrt(head_size) if scale == "auto" else float(scale)

    B = batch_size
    T_len = seq_len
    H = num_q_heads
    HK = num_k_heads
    HV = num_v_heads
    K = head_size
    V = head_size

    q = torch.randn(B, T_len, H, K, dtype=torch_dtype, device=device) * 0.1
    k = torch.randn(B, T_len, HK, K, dtype=torch_dtype, device=device) * 0.1
    v = torch.randn(B, T_len, HV, V, dtype=torch_dtype, device=device) * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(B, T_len, HV, dtype=torch_dtype, device=device) * 0.1
    b_tensor = torch.randn(B, T_len, HV, dtype=torch_dtype, device=device) * 0.1
    initial_state = torch.randn(B, HV, V, K, dtype=torch.float32, device=device) * 0.01
    initial_state_indices = torch.arange(B, dtype=torch.int32, device=device)

    baseline_o = None
    baseline_state = None
    baseline_intermediate = None

    for repeat_idx in range(repeats):
        state = initial_state.clone()
        state_ptr = state.data_ptr()
        intermediate_states_buffer = (
            torch.zeros(B, T_len, HV, V, K, dtype=torch.float32, device=device)
            if cache_intermediate_states
            else None
        )

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
            state_indices=initial_state_indices,
            scale=scale_value,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer,
            disable_state_update=True,
            use_qk_l2norm=True,
        )
        _synchronize_device(device.type)

        assert state.data_ptr() == state_ptr
        assert returned_state.data_ptr() == state_ptr

        if repeat_idx == 0:
            baseline_o = out.clone()
            baseline_state = returned_state.clone()
            baseline_intermediate = (
                intermediate_states_buffer.clone()
                if intermediate_states_buffer is not None
                else None
            )
            continue

        if not torch.equal(out, baseline_o):
            diff = (out.float() - baseline_o.float()).abs().max().item()
            raise AssertionError(
                f"MTP output is not bitwise stable at repeat={repeat_idx}, "
                f"max_abs_diff={diff}"
            )
        if not torch.equal(returned_state, baseline_state):
            diff = (returned_state - baseline_state).abs().max().item()
            raise AssertionError(
                f"MTP state is not bitwise stable at repeat={repeat_idx}, "
                f"max_abs_diff={diff}"
            )
        if intermediate_states_buffer is not None and not torch.equal(
            intermediate_states_buffer, baseline_intermediate
        ):
            diff = (
                (intermediate_states_buffer.float() - baseline_intermediate.float())
                .abs()
                .max()
                .item()
            )
            raise AssertionError(
                "MTP intermediate states are not bitwise stable at "
                f"repeat={repeat_idx}, max_abs_diff={diff}"
            )


def _run_mate_extra_mtp_case(
    *,
    batch_size: int,
    seq_len: int,
    num_q_heads: int = 16,
    num_k_heads: int = 16,
    num_v_heads: int = 32,
    dtype: str = "bfloat16",
    state_dtype: torch.dtype = torch.float32,
    dt_bias_dtype: torch.dtype = torch.float32,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    use_state_indices: bool,
    use_negative_state_indices: bool = False,
    use_preallocated_output: bool = False,
    use_strided_output: bool = False,
    use_strided_inputs: bool = False,
    use_strided_state_indices: bool = False,
    use_noncontiguous_state: bool = False,
    use_qk_l2norm: bool = True,
    state_pool_size: int | None = None,
    intermediate_pool_size: int | None = None,
    use_sglang_mtp_strides: bool = False,
    seed: int = 0,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    torch_dtype = _torch_dtype(dtype)
    device = _get_runtime_device(allow_skip=True)
    _manual_seed_device(seed, device.type)

    K = 128
    V = 128
    pool_size = (
        state_pool_size
        if state_pool_size is not None
        else max(batch_size * 2, 8)
        if use_state_indices
        else batch_size
    )
    scale = 1.0 / math.sqrt(K)

    if use_sglang_mtp_strides:
        q = _make_random_empty_strided(
            (batch_size, seq_len, num_q_heads, K),
            (24576, 8192, 128, 1),
            dtype=torch_dtype,
            device=device,
        )
        k = _make_random_empty_strided(
            (batch_size, seq_len, num_k_heads, K),
            (24576, 8192, 128, 1),
            dtype=torch_dtype,
            device=device,
        )
        v = _make_random_empty_strided(
            (batch_size, seq_len, num_v_heads, V),
            (24576, 8192, 128, 1),
            dtype=torch_dtype,
            device=device,
        )
    elif use_strided_inputs:
        q = _make_strided_random(
            (batch_size, seq_len, num_q_heads, K),
            dtype=torch_dtype,
            device=device,
        )
        k = _make_strided_random(
            (batch_size, seq_len, num_k_heads, K),
            dtype=torch_dtype,
            device=device,
        )
        v = _make_strided_random(
            (batch_size, seq_len, num_v_heads, V),
            dtype=torch_dtype,
            device=device,
        )
    else:
        q = (
            torch.randn(
                batch_size, seq_len, num_q_heads, K, dtype=torch_dtype, device=device
            )
            * 0.1
        )
        k = (
            torch.randn(
                batch_size, seq_len, num_k_heads, K, dtype=torch_dtype, device=device
            )
            * 0.1
        )
        v = (
            torch.randn(
                batch_size, seq_len, num_v_heads, V, dtype=torch_dtype, device=device
            )
            * 0.1
        )
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = (torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1).to(
        dt_bias_dtype
    )
    if use_strided_inputs:
        a = _make_strided_random(
            (batch_size, seq_len, num_v_heads),
            dtype=torch_dtype,
            device=device,
        )
        b = _make_strided_random(
            (batch_size, seq_len, num_v_heads),
            dtype=torch_dtype,
            device=device,
        )
    else:
        a = (
            torch.randn(
                batch_size, seq_len, num_v_heads, dtype=torch_dtype, device=device
            )
            * 0.1
        )
        b = (
            torch.randn(
                batch_size, seq_len, num_v_heads, dtype=torch_dtype, device=device
            )
            * 0.1
        )

    state = _prepare_initial_state(
        pool_size=pool_size,
        num_v_heads=num_v_heads,
        head_size=K,
        state_dtype=state_dtype,
        device=device,
        use_noncontiguous_state=use_noncontiguous_state,
    )
    state_before = state.clone()
    state_ref_input = state.clone()

    if use_state_indices:
        state_indices = torch.randperm(pool_size, device=device)[:batch_size].to(
            torch.int32
        )
        if use_strided_state_indices:
            state_indices_storage = torch.empty(
                batch_size * 2, dtype=torch.int32, device=device
            )
            state_indices_storage[0::2] = state_indices
            state_indices_storage[1::2] = -777
            state_indices = state_indices_storage[0::2]
            assert not state_indices.is_contiguous()
        if use_negative_state_indices:
            if not use_strided_state_indices:
                state_indices = state_indices.clone()
            state_indices[1::2] = -1
    else:
        state_indices = None

    intermediate_states_buffer = (
        torch.zeros(
            intermediate_pool_size if intermediate_pool_size is not None else pool_size,
            seq_len,
            num_v_heads,
            V,
            K,
            dtype=state_dtype,
            device=device,
        )
        if cache_intermediate_states
        else None
    )
    if use_strided_output:
        output = _make_strided_mtp_output(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_v_heads,
            head_size=V,
            dtype=torch_dtype,
            device=device,
        )
    elif use_preallocated_output:
        output = torch.empty(
            batch_size, seq_len, num_v_heads, V, dtype=torch_dtype, device=device
        )
    else:
        output = None

    out, returned_state = mate.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        state_layout="VK",
        state_indices=state_indices,
        scale=scale,
        output=output,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
    )
    _synchronize_device(device.type)

    assert returned_state.data_ptr() == state.data_ptr()
    if output is not None:
        assert out.data_ptr() == output.data_ptr()

    reference_indices = (
        torch.arange(batch_size, dtype=torch.int32, device=device)
        if state_indices is None
        else state_indices
    )
    ref_out, ref_state, ref_intermediate = _flashinfer_reference_with_state_indices(
        q=q,
        k=k,
        v=v,
        state_vk=state_ref_input,
        state_indices=reference_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
        state_dtype=state_dtype,
        intermediate_pool_size=(
            intermediate_states_buffer.shape[0]
            if intermediate_states_buffer is not None
            else None
        ),
    )

    atol_o, rtol_o = _output_tolerances(dtype, state_dtype)
    atol_s, rtol_s = _state_tolerances(dtype, state_dtype)
    torch.testing.assert_close(out.float(), ref_out, atol=atol_o, rtol=rtol_o)

    if disable_state_update:
        torch.testing.assert_close(
            returned_state.float(),
            state_before.float(),
            atol=0.0,
            rtol=0.0,
        )
    else:
        torch.testing.assert_close(
            returned_state.float(), ref_state.float(), atol=atol_s, rtol=rtol_s
        )

    if intermediate_states_buffer is not None:
        assert ref_intermediate is not None
        torch.testing.assert_close(
            intermediate_states_buffer.float(),
            ref_intermediate.float(),
            atol=atol_s,
            rtol=rtol_s,
        )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("cache_intermediate_states", [True, False])
@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("seq_len", [2, 4, 8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_flashinfer_verify_kernel_mtp(
    dtype: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    batch_size: int,
    seq_len: int,
    scale: float | str,
    alpha: bool,
    beta: bool,
    cache_intermediate_states: bool,
    seed: int = _get_default_seed(),
) -> None:
    _run_flashinfer_mtp_case(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        seq_len=seq_len,
        scale=scale,
        alpha=alpha,
        beta=beta,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=True,
        seed=seed,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("cache_intermediate_states", [True, False])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_flashinfer_verify_kernel_mtp_bitwise_stable(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    seq_len: int,
    scale: float | str,
    cache_intermediate_states: bool,
    seed: int = _get_default_seed(),
) -> None:
    _run_flashinfer_mtp_bitwise_case(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        seq_len=seq_len,
        scale=scale,
        cache_intermediate_states=cache_intermediate_states,
        seed=seed,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("seq_len", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_flashinfer_mtp_fp32_state_with_cache_and_state_update(
    dtype: str,
    batch_size: int,
    seq_len: int,
    seed: int = _get_default_seed(),
) -> None:
    _run_flashinfer_mtp_case(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        seq_len=seq_len,
        scale=1.0 / math.sqrt(128),
        alpha=True,
        beta=True,
        cache_intermediate_states=True,
        disable_state_update=False,
        seed=seed,
    )


MATE_EXTRA_MTP_CASES = [
    pytest.param(
        dict(
            batch_size=2,
            seq_len=2,
            cache_intermediate_states=False,
            disable_state_update=False,
            use_state_indices=False,
        ),
        id="identity_no_cache_update",
    ),
    pytest.param(
        dict(
            batch_size=1,
            seq_len=2,
            num_q_heads=8,
            num_k_heads=8,
            num_v_heads=16,
            cache_intermediate_states=True,
            disable_state_update=True,
            use_state_indices=False,
        ),
        id="identity_cache_readonly_h8_hv16",
    ),
    pytest.param(
        dict(
            batch_size=4,
            seq_len=4,
            cache_intermediate_states=True,
            disable_state_update=False,
            use_state_indices=True,
            use_negative_state_indices=True,
            use_preallocated_output=True,
            use_noncontiguous_state=True,
        ),
        id="pool_negative_preallocated_noncontiguous_state",
    ),
    pytest.param(
        dict(
            batch_size=2,
            seq_len=2,
            dtype="float16",
            cache_intermediate_states=True,
            disable_state_update=True,
            use_state_indices=True,
            use_strided_state_indices=True,
            use_strided_inputs=True,
            use_strided_output=True,
        ),
        id="pool_strided_io_indices_fp16_readonly",
    ),
    pytest.param(
        dict(
            batch_size=1,
            seq_len=3,
            num_q_heads=16,
            num_k_heads=16,
            num_v_heads=32,
            dt_bias_dtype=torch.bfloat16,
            cache_intermediate_states=True,
            disable_state_update=True,
            use_state_indices=True,
            state_pool_size=402,
            intermediate_pool_size=33,
            use_sglang_mtp_strides=True,
        ),
        id="sglang_pool402_intermediate33_readonly",
    ),
    pytest.param(
        dict(
            batch_size=2,
            seq_len=4,
            num_q_heads=16,
            num_k_heads=16,
            num_v_heads=64,
            dt_bias_dtype=torch.bfloat16,
            cache_intermediate_states=True,
            disable_state_update=False,
            use_state_indices=False,
            use_qk_l2norm=False,
        ),
        id="identity_hv64_bf16_dt_bias_no_qk_l2norm",
    ),
    pytest.param(
        dict(
            batch_size=2,
            seq_len=2,
            state_dtype=torch.bfloat16,
            cache_intermediate_states=True,
            disable_state_update=True,
            use_state_indices=False,
        ),
        id="bf16_state_identity_cache_readonly",
    ),
    pytest.param(
        dict(
            batch_size=4,
            seq_len=4,
            state_dtype=torch.bfloat16,
            cache_intermediate_states=True,
            disable_state_update=False,
            use_state_indices=True,
            use_negative_state_indices=True,
            use_preallocated_output=True,
        ),
        id="bf16_state_pool_negative_cache_update",
    ),
    pytest.param(
        dict(
            batch_size=2,
            seq_len=8,
            state_dtype=torch.bfloat16,
            cache_intermediate_states=False,
            disable_state_update=False,
            use_state_indices=False,
            use_qk_l2norm=False,
        ),
        id="bf16_state_t8_no_cache_update_no_qk_l2norm",
    ),
]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", MATE_EXTRA_MTP_CASES)
def test_mate_extra_mtp_api_case(
    case: dict,
    seed: int = _get_default_seed(),
) -> None:
    _run_mate_extra_mtp_case(**case, seed=seed)


if __name__ == "__main__":
    test_flashinfer_verify_kernel_mtp(
        dtype="bfloat16",
        batch_size=4,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        seq_len=4,
        scale=1.0,
        alpha=True,
        beta=True,
        cache_intermediate_states=True,
    )
