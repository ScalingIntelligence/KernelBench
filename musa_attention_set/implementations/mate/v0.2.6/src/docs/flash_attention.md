# FlashAttention3 Forward Compatibility

This document is a quick reference for the current FlashAttention-3-compatible **forward** coverage provided by MATE on MUSA.

## Compatibility Overview

If you already use FlashAttention-3 style Python APIs, start here to see what
works on MATE and how to try it.

1. Install ``flash_attn_3``.
2. Import ``flash_attn_interface``.
3. Call the usual forward APIs.
4. Check the matrix below for supported modes, dtypes, and limits.

This page covers how to get started with forward compatibility on MUSA. It
does not cover kernel internals, backward, or autograd behavior.

## Getting Started

Install the compatibility wrapper from the MUSA wheel source:

```bash
python -m pip install flash_attn_3 \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

Minimal forward example:

```python
import torch
from flash_attn_interface import flash_attn_varlen_func

device = "musa"
dtype = torch.bfloat16

q = torch.randn((96, 32, 128), device=device, dtype=dtype)
k = torch.randn((160, 8, 128), device=device, dtype=dtype)
v = torch.randn((160, 8, 128), device=device, dtype=dtype)

cu_seqlens_q = torch.tensor([0, 32, 96], device=device, dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 64, 160], device=device, dtype=torch.int32)

out = flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=64,
    max_seqlen_k=96,
    causal=False,
)
```

If your integration uses KV cache, paged KV, scheduler metadata, FP8, or
``only_qv``, check the compatibility matrix below before porting more code.

Minimal KV-cache example:

```python
import torch
from flash_attn_interface import flash_attn_with_kvcache

device = "musa"
dtype = torch.bfloat16

q = torch.randn((1, 4, 32, 128), device=device, dtype=dtype)
k_cache = torch.randn((8, 16, 8, 128), device=device, dtype=dtype)
v_cache = torch.randn((8, 16, 8, 128), device=device, dtype=dtype)
cache_seqlens = torch.tensor([32], device=device, dtype=torch.int32)
block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)

out, softmax_lse = flash_attn_with_kvcache(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    causal=True,
)
```

## At a Glance

| Area | Status | Notes |
| --- | --- | --- |
| Q Mode | ✅ Supported | `Normal`, `Ragged`, `Padded` |
| KV Mode | ✅ Supported | `Normal`, `Ragged`, `Padded`, `Paged` |
| Append New KV | ✅ Supported | `flash_attn_with_kvcache` appends via `k` / `v`; packed new KV is supported via `cu_seqlens_k_new` |
| RoPE Input | ✅ Supported | `flash_attn_with_kvcache` supports `rotary_cos` / `rotary_sin`, `Interleaved`, and `Non-interleaved` |
| Cache Index Options | ✅ Supported | `cache_batch_idx`, `cache_leftpad` |
| Mask Mode | ✅ Supported | `None`, `Causal`, `Local`, `Local + attention_chunk` |
| Score Mode | ✅ Supported | Standard softmax and `softcap` |
| Page Size | ✅ Supported | `1`, `16`, `64`, and arbitrary page sizes |
| Dtype | ✅ Supported | `bf16`, `fp16`, `torch.float8_e4m3fn`, and `torch.float8_e5m2` forward inputs; FP8 uses `q_descale` / `k_descale` / `v_descale` scaling |
| QV Input | ✅ Supported | The forward path supports an optional `qv` input, including FP8 inputs with `qv` |
| HeadDim | ✅ Supported | Any `headdim <= 512` |
| Optimization | ✅ Supported | `SplitKV`, `PackGQA`, `SchedulerMetadata` |
| Output | ✅ Supported | `out`, `softmax_lse` |

## MATE Extensions

| Extension | Status | Notes |
| --- | --- | --- |
| Context Parallel | ✅ Supported | `cp_world_size`, `cp_rank`, `cp_tot_seqused_k` |
| Learnable Sink | ✅ Supported | Supported on the local-attention path |

## Notes

- This page summarizes the compatibility surface, not every internal kernel detail.
- The statement `Any headdim <= 512` refers to the supported forward-path head-dimension range.
- FP8 forward support includes `torch.float8_e4m3fn` and `torch.float8_e5m2`; pass optional
  `q_descale`, `k_descale`, and `v_descale` tensors with shape
  `(batch_size, num_heads_kv)` when scale factors are required.
- When both `q` and the optional `qv` input are FP8, `q_descale` applies to
  both query tensors; `k_descale` and `v_descale` still apply to the KV inputs.
- RoPE is supported only when appending new KV through `k` / `v`; `rotary_dim` must be `<= headdim` and divisible by 16.
- `Local + attention_chunk` requires MUSA SDK >= 5.1.0.
- FP8 attention works on the forward path today. For best performance, use
  MUSA SDK 5.2.0 or newer when available.
- For wrapper-level usage, see the {doc}`FlashAttention wrapper page </wrappers/flash_attention_wrapper>`.
