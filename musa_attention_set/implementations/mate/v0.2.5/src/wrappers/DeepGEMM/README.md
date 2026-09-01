# DeepGEMM Compatibility Wrapper (deep-gemm)

`deep-gemm` is a compatibility wrapper package that preserves the
`deep_gemm` import path while running on MUSA through MATE GEMM operators.

## Overview

This wrapper is designed for projects that already target DeepGEMM-style
Python APIs. It helps run existing integrations on MUSA through MATE with
minimal code changes.

The current compatibility scope includes grouped GEMM, dense BF16 / FP8 GEMM,
FP8 einsum, HyperConnection prenorm GEMM, and MQA (multi-query attention)
logits APIs.

## Package and import

- Package name: `deep-gemm`
- Import path: `deep_gemm`
- Runtime backend: MATE GEMM and logits operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.4+musa`. Use `python -m pip show deep-gemm` to distinguish this
wrapper from the native package.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.

## Build

Build a wheel from the `wrappers/DeepGEMM` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under:

```text
dist/
```

## Installation

For delivered packages, install from the external MUSA wheel source:

```bash
python -m pip install deep-gemm \
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
python -m pip install --no-deps dist/deep_gemm-*.whl
```

If you previously installed the legacy `mate-deep-gemm` package, uninstall it
before installing `deep-gemm` so the environment does not keep stale wrapper
metadata.

```bash
python -m pip uninstall -y mate-deep-gemm
python -m pip install --no-deps dist/deep_gemm-*.whl
```

## Import

Import the package directly:

```python
import deep_gemm
```

Import individual APIs:

```python
from deep_gemm import (
    bf16_gemm_nt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    fp8_gemm_nt,
    fp8_einsum,
    tf32_hc_prenorm_gemm,
    get_paged_mqa_logits_metadata,
    fp8_paged_mqa_logits,
    fp8_mqa_logits,
)
```

## Public APIs

Dense BF16 GEMM:

- `bf16_gemm_nt`

Grouped GEMM:

- `m_grouped_bf16_gemm_nt_contiguous`
- `m_grouped_bf16_gemm_nt_masked`
- `m_grouped_fp8_gemm_nt_contiguous`
- `m_grouped_fp8_gemm_nt_masked`
- Legacy aliases: `fp8_m_grouped_gemm_nt_masked`, `bf16_m_grouped_gemm_nt_masked`

Dense FP8 GEMM:

- `fp8_gemm_nt`
- `fp8_einsum`

HyperConnection prenorm GEMM:

- `tf32_hc_prenorm_gemm`

MQA logits APIs:

- `get_paged_mqa_logits_metadata`
- `fp8_paged_mqa_logits`
- `fp8_mqa_logits`

Utility helpers re-exported from `deep_gemm.utils`:

- `bench`, `bench_kineto`, `calc_diff`
- `get_num_sms`, `set_num_sms`
- `get_tc_util`, `set_tc_util`
- `get_mk_alignment_for_contiguous_layout`
- `get_col_major_tma_aligned_tensor`
- `get_mn_major_tma_aligned_tensor`

Testing helpers are implemented in `mate.testing.deep_gemm` and are also
available from the upstream-style path:

- `deep_gemm.testing.bench`
- `deep_gemm.testing.bench_kineto`
- `deep_gemm.testing.calc_diff`

## Contiguous Grouped GEMM Alignment

`get_mk_alignment_for_contiguous_layout()` returns the M-axis padding alignment
used by DeepGEMM-compatible contiguous grouped GEMM wrappers. It defaults to
`128` and can be overridden with:

```bash
export MATE_DEEPGEMM_MK_ALIGNMENT=256
```

Only `128` and `256` are supported. Use the returned value when padding each
expert segment and building `m_indices` for
`m_grouped_{fp8,bf16}_gemm_nt_contiguous`. Set the environment variable before
starting Python; the helper reads and caches the value on first use.

## Quick Start

Minimal import example:

```python
import deep_gemm
```

An example script is provided at:

```text
examples/run_deep_gemm.py
```

## Examples

Run the bundled example:

```bash
python examples/run_deep_gemm.py
```

## Notes

- This wrapper preserves the DeepGEMM-style Python surface, but execution is provided by MATE on MUSA
- `get_paged_mqa_logits_metadata(..., block_kv, ...)` currently requires `block_kv == 64`
- The example script currently demonstrates the FP8 grouped GEMM path
