# MATE Gated Delta Network (GDN) Support Matrix

This document provides a comprehensive overview of the Gated Delta Network
(GDN) features, data types, and execution backends currently supported in
MATE. Use this matrix to verify configuration compatibility for production
deployment and optimization.

## Overview

MATE provides optimized kernels for Gated Delta Networks (GDN). MATE splits
GDN execution into two primary phases:

- Prefill: high-throughput processing of prompt contexts.
- Decode: low-latency generation, including standard single-token streaming
  and multi-token prediction (MTP).

## Decode

### Dispatch Matrix

| `state_layout` | `state.dtype` | `T` | Status | Notes |
| --- | --- | --- | --- | --- |
| `VK` | `bfloat16` | `1` | ❌ Not supported | Single-token BF16-state decode is not implemented yet |
| `VK` | `bfloat16` | `> 1` | ✅ Supported | TileLang MTP path, currently K=V=128 |
| `VK` | `float32` | `1` | ✅ Supported | Current active decode path |
| `VK` | `float32` | `> 1` | ✅ Supported | TileLang MTP path, currently K=V=128 |
| `KV` | `float32` | `1` | ❌ Not supported | KV backend is not implemented yet |
| `KV` | anything else | any | ❌ Not supported | Unsupported combination |

### Current MATE Active Path

Current MATE decode support is intentionally narrow:

- API: `mate.gated_delta_rule_decode(...)`
- Supported combinations:
  - `state_layout="VK"`, `state.dtype=float32`, `T=1`
  - `state_layout="VK"`, `state.dtype=float32`, `T>1`, `K=V=128`
  - `state_layout="VK"`, `state.dtype=bfloat16`, `T>1`, `K=V=128`
- Backend: TileLang
- State update: in place

### Current MATE Restrictions

| Item | MATE |
| --- | --- |
| `state_indices` | ✅ Supported on VK FP32 decode and VK MTP; negative entries are padding |
| BF16 state backend | ✅ Supported on VK MTP (`T>1`, `K=V=128`) |
| KV backend | ❌ Not supported |
| MTP (`T > 1`) | ✅ Supported for VK float32/bfloat16 state with K=V=128 |
| `intermediate_states_buffer` | ✅ Supported on VK MTP; dtype must match `state.dtype` |
| `disable_state_update` | ✅ Supported on VK MTP |

### Input Contract on the Active Path

On the currently supported MATE path:

- `q / k / v`: `float16` or `bfloat16`
- `A_log / dt_bias`: `float32`
- `a / b`: same dtype as `q`
- `state`: `float32` for single-token decode; `float32` or `bfloat16` for MTP
- `intermediate_states_buffer`: same dtype as `state`
- `output`: optional, supports `float16` / `bfloat16` / `float32`

## Prefill

### Prefill At a Glance

| Area | Status | Notes |
| --- | --- | --- |
| Backend | ✅ Supported | FlashInfer-aligned native MP31 prefill path on MUSA via `mate.gdn_prefill.chunk_gated_delta_rule` |
| Sequence Mode | ✅ Supported | Varlen prefill with `cu_seqlens` |
| Head Layout | ✅ Supported | `GQA` and `GVA` |
| Dtype (Q/K/V) | ✅ Supported | `fp16`, `bf16` |
| Gate Inputs | ✅ Supported | `g` and `beta` are float32 tensors; defaults to all-ones when omitted |
| Gate Space | ✅ Supported | `is_log_space=True` treats `g` as `log(alpha)`; `is_log_space=False` treats `g` as alpha in `(0, 1]` |
| Initial State | ✅ Supported | Optional `initial_state` with shape `[batch, head_sab, dim_v, dim_k]` (float32) |
| Final State Output | ✅ Supported | `output_final_state=True` returns `(output, final_state)` |
| Output Heads | ✅ Supported | `head_o = max(num_q_heads, num_v_heads)` |
| QK L2 Norm Option | ✅ Supported | `use_qk_l2norm_in_kernel=True` normalizes Q and K in place with a standalone TileLang kernel before KKT/prefill |
| Strided Q/K/V | ✅ Supported | Split-QKV views are supported when the last dimension is contiguous (`stride(-1) == 1`) |

### Prefill Shape Rules

| Item | Requirement |
| --- | --- |
| `q`, `k`, `v` rank | 3D tensors: `[total_tokens, heads, dim]` |
| Token count | `q.size(0) == k.size(0) == v.size(0)` |
| Q/K dim | `q.size(2) == k.size(2)` |
| Head layout | `GQA`: `num_v_heads == num_k_heads` and `num_q_heads % num_k_heads == 0`; `GVA`: `num_q_heads == num_k_heads` and `num_v_heads % num_q_heads == 0` |
| `cu_seqlens` | Required by public wrapper for varlen prefill |
| `chunk_size` | Must be exactly `64` on the current native path |
| Strides | `q`, `k`, and `v` may be non-contiguous split views, but each must satisfy `stride(-1) == 1` |
| QK L2 norm | If `use_qk_l2norm_in_kernel=True`, Q and K are modified in place before the KKT solve and fused prefill launch |

### Current Kernel Constraints

| Item | Status | Notes |
| --- | --- | --- |
| `dim_k` range | ✅ Required | Native MP31 kernel currently supports `dim_k <= 128` |
| `dim_v` range | ✅ Required | Native MP31 kernel currently supports `dim_v <= 128` |
| `chunk_size` | ✅ Required | Current native path is fixed to `chunk_size == 64` to match the FlashInfer-style variant chain |
| Device | ✅ Required | MUSA on MP31 |

### Not Supported Yet (Prefill)

| Feature | Status | Notes |
| --- | --- | --- |
| Non-GQA/GVA head mapping | ❌ Not supported | Only the two grouped layouts above are accepted |

### Notes

- Public API entry: `mate.gdn_prefill.chunk_gated_delta_rule`.
- `use_qk_l2norm_in_kernel=True` is an in-place operation on the input Q and K tensors. Pass cloned tensors if the original unnormalized values are still needed after the call.
- Strided support is intended for fused/split QKV layouts such as a single physical `[tokens, qkv_dim]` allocation split into Q, K, and V views. Arbitrary layouts with a non-contiguous last dimension are not supported.
