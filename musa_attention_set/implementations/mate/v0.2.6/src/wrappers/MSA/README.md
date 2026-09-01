# MSA Compatibility Wrapper (fmha_sm100)

`fmha_sm100` is a compatibility wrapper package that preserves the
`fmha_sm100` package name and import path while running on MUSA through MATE
MiniMax Sparse Attention (MSA) operators.

## Overview

This wrapper is designed for projects that already target the `fmha_sm100`
Python API. It keeps the public `fmha_sm100` import surface and forwards
supported calls to `mate.msa_interface`.

The current compatibility scope includes:

- `fmha_sm100.fmha_sm100_plan`
- `fmha_sm100.fmha_sm100`
- `fmha_sm100.sparse_topk_select`
- `fmha_sm100.sparse.sparse_decode_atten_func`
- `fmha_sm100.sparse.sparse_decode_atten_func`
- `fmha_sm100.sparse.sparse_fmha_plan`
- `fmha_sm100.sparse.sparse_fmha`

## Package and import

- Package name: `fmha_sm100`
- Import path: `fmha_sm100`
- Runtime backend: MATE MSA operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.5+musa`. Use `python -m pip show fmha_sm100` to distinguish this
wrapper from the native package.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.
- The current MSA kernels target Pinghu (MP31).

## Build

Build a wheel from the `wrappers/MSA` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under `dist/`.

## Installation

For delivered packages, install from the external MUSA wheel source:

```bash
python -m pip install fmha_sm100 \
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
python -m pip install --no-deps dist/fmha_sm100-*.whl
```

## Import

Import the package directly:

```python
import fmha_sm100
```

Import the main APIs:

```python
from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select
from fmha_sm100.sparse import sparse_decode_atten_func, sparse_fmha, sparse_fmha_plan
```

## Public APIs

The wrapper currently exposes:

- `fmha_sm100_plan(...)`
- `fmha_sm100(...)`
- `sparse_fmha_plan(...)`
- `sparse_fmha(...)`
- `sparse_decode_atten_func(...)`
- `sparse_topk_select(...)`

Use `fmha_sm100_plan(...)` and `fmha_sm100(...)` for the main forward path.

Use the sparse helpers for sparse prefill and sparse decode:

## Sparse decode behavior

`sparse_decode_atten_func(...)` has two execution paths:

- `q2k_indices=None`: runs dense all-KV paged decode
- `q2k_indices` is a tensor: runs sparse decode

Use `SparseDecodePagedAttentionWrapper.plan()` to cache decode metadata, then
use `run()` to execute the same decode path.

## Quick start

Main forward path:

```python
from fmha_sm100 import fmha_sm100, fmha_sm100_plan

plan = fmha_sm100_plan(
    qo_lens=qo_lens,
    kv_lens=kv_lens,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    page_size=128,
    causal=True,
)

out = fmha_sm100(
    q=q,
    k=k,
    v=v,
    plan_info=plan,
)
```

Sparse prefill:

```python
from fmha_sm100.sparse import sparse_fmha, sparse_fmha_plan

plan = sparse_fmha_plan(
    qo_lens=qo_lens,
    kv_lens=kv_lens,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    page_size=128,
    sparse_block_size=128,
    kv_block_num=16,
    causal=False,
)

out, lse = sparse_fmha(
    q=q,
    k=k,
    v=v,
    plan_info=plan,
    kv_indices=kv_indices,
    kv_block_indexes=kv_block_indexes,
)
```

Sparse decode:

```python
from fmha_sm100.sparse import sparse_decode_atten_func

out = sparse_decode_atten_func(
    q=q,
    k=k_cache,
    v=v_cache,
    q2k_indices=q2k_indices,
    page_table=page_table,
    seqused_k=seqused_k,
    seqlen_q=seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
```

## Compatibility limits

- Sparse decode currently requires `topK=16`.
- `sparse_topk_select(...)` supports `topk` values `4`, `8`, and `16`.
- `q2k_indices=None` uses dense all-KV paged decode, not sparse decode.
- Dense fallback decode does not support `return_softmax_lse=True`.
- `SparseK2qCsrBuilderSm100` is reserved and raises `NotImplementedError`.
- FP4 and NVFP4 helper paths are reserved and raise `NotImplementedError`.

## Direct API path

Use direct `mate` APIs when you need the lower-level MSA surface:

- `mate.msa`
- `mate.msa_plan`
- `mate.sparse_msa`
- `mate.sparse_msa_plan`
- `mate.sparse_topk_select`

## Notes

- This wrapper keeps the `fmha_sm100` import surface, but execution is
  provided by MATE on MUSA.
- This page covers the wrapper-facing workflow and compatibility scope. It
  does not cover kernel internals.
- For the authoritative operator behavior, refer to `mate.msa_interface`.
