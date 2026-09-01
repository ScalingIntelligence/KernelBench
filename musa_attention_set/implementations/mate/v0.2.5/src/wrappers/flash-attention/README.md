# FlashAttention-3 Compatibility Wrapper (flash_attn_3)

`flash_attn_3` is a compatibility wrapper package that preserves the official
FlashAttention-3 package surface while running on MUSA through MATE attention
operators.

## Overview

This wrapper is designed for projects that already target FlashAttention-3
style Python APIs. It helps run existing integrations on MUSA through MATE
with minimal code changes. The current compatibility target is the
`flash_attn_3` interface surface.

## Package and import

- Package name: `flash_attn_3`
- Public import path: `flash_attn_interface`
- Internal package path: `flash_attn_3`
- Runtime backend: MATE attention operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.4+musa`. Use `python -m pip show flash_attn_3` to distinguish
this wrapper from the native package.

For the current compatibility scope and known limitations, see
`docs/source/wrappers/flash_attention_forward_compatibility.md`.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.

## Build

Build a wheel from the `wrappers/flash-attention` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under `dist/`.

## Installation

For delivered packages, install from the external MUSA wheel source:

```bash
python -m pip install flash_attn_3 \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

This installs the matching `mate` dependency automatically.

For local wrapper development, install from source:

```bash
python -m pip install --no-build-isolation --no-deps -e ../.. -v
python -m pip install --no-build-isolation --no-deps -e .
```

Install a built local wheel:

```bash
python -m pip install --no-deps dist/flash_attn_3-*.whl
```

If you previously installed the legacy `mate-flash-attention` package, uninstall it before installing `flash_attn_3` so the
environment does not keep stale wrapper metadata.

```bash
python -m pip uninstall -y mate-flash-attention
python -m pip install --no-deps dist/flash_attn_3-*.whl
```

## Import

Import the package directly:

```python
import flash_attn_interface
```

Import individual APIs:

```python
from flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
```

## Public APIs

The wrapper currently exposes:

- `flash_attn_func`: FlashAttention-compatible dense QKV entry
- `flash_attn_varlen_func`: FlashAttention-compatible varlen / ragged FMHA entry
- `flash_attn_with_kvcache`: FlashAttention-compatible KV-cache entry, including paged KV cache use cases
- `get_scheduler_metadata`: helper for split-KV scheduler metadata preparation

## Quick Start

Minimal varlen FMHA example:

```python
import torch
from flash_attn_interface import flash_attn_varlen_func

device = "musa"
dtype = torch.bfloat16

num_heads_q = 32
num_heads_kv = 8
head_dim = 128

cu_seqlens_q = torch.tensor([0, 32, 96], device=device, dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 64, 160], device=device, dtype=torch.int32)

q = torch.randn((96, num_heads_q, head_dim), device=device, dtype=dtype)
k = torch.randn((160, num_heads_kv, head_dim), device=device, dtype=dtype)
v = torch.randn((160, num_heads_kv, head_dim), device=device, dtype=dtype)

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

Paged KV cache skeleton with scheduler metadata:

```python
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata

metadata = get_scheduler_metadata(
    batch_size=batch_size,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    num_heads_q=num_heads_q,
    num_heads_kv=num_heads_kv,
    headdim=head_dim_qk,
    headdim_v=head_dim_v,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    page_size=page_size,
    num_splits=0,
    pack_gqa=True,
)

out, lse, *_ = flash_attn_with_kvcache(
    q=q_unpad,
    k_cache=k_cache_paged,
    v_cache=v_cache_paged,
    cache_seqlens=cache_seqlens,
    page_table=page_table,
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    scheduler_metadata=metadata,
    return_softmax_lse=True,
)
```

## Tests

Wrapper-level tests are available in:

```text
tests/test_flash_attn.py
```

Run them from the `wrappers/flash-attention` directory:

```bash
pytest tests/test_flash_attn.py tests/test_interface_unit.py
```

## Notes

- This wrapper follows the FlashAttention-3 style Python package layout, and the current compatibility target is `flash_attn_3`
- The `flash_attn` top-level package is intentionally not shipped, so Transformers FA2 / FA4 package checks keep failing
- Actual feature coverage is documented in
  `docs/source/wrappers/flash_attention_forward_compatibility.md`
- For the authoritative operator behavior, refer to the corresponding MATE APIs under `mate.mha_interface`
