# SageAttention Compatibility Wrapper (sageattention)

`sageattention` is a compatibility wrapper that preserves the standard
SageAttention Python API surface, running on top of MATE's dense quantized
attention operators on MUSA.

## Overview

This wrapper is designed for projects that already target
SageAttention-style Python APIs, allowing you to run on MUSA through MATE
with minimal integration effort.

The current compatibility scope includes `sageattn` and
`sageattn_qk_int8_pv_fp8_cuda_sm90`.

## Package and import

- Package name: `sageattention`
- Import path: `sageattention`
- Runtime backend: MATE dense quantized attention operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.4+musa`. Use `python -m pip show sageattention` to distinguish
this wrapper from the native package.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA and the MUSA runtime environment are available.
- The target workload is configured to execute on MUSA devices.

## Build

Build a wheel from the `wrappers/SageAttention` directory:

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
python -m pip install sageattention \
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
python -m pip install --no-deps dist/sageattention-*.whl
```

## Import

Import the package directly:

```python
import sageattention
```

Import individual APIs:

```python
from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda_sm90
```

## Public APIs

The wrapper currently exposes:

- `sageattn`: primary SageAttention-compatible dense attention entry
- `sageattn_qk_int8_pv_fp8_cuda_sm90`: compatibility alias for the supported
  dense quantized attention path

## Quick Start

Minimal dense attention example:

```python
import torch
from sageattention import sageattn

device = "musa"
dtype = torch.bfloat16

q = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)
k = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)
v = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)

out = sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_dtype="int8",
)
```

FP8 output example:

```python
out_fp8, out_scale = sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_dtype="int8",
    fp8_output=True,
)

out_dequant = out_fp8.to(torch.float32) * out_scale
```

When `return_lse=True`, the FP8 output form returns
`(out_fp8, out_scale, lse)`. Without FP8 output, the return forms are `out` or
`(out, lse)`.

## Tests

Wrapper-level tests are available in:

```text
tests/test_sageattn_interface.py
```

Run them from the `wrappers/SageAttention` directory:

```bash
pytest tests/test_sageattn_interface.py
```

## Notes

- This wrapper currently supports the dense SageAttention path only
- Input tensors must be on the same MUSA device, use `torch.float16` or
  `torch.bfloat16`, and share the same dtype
- Supported public `tensor_layout` values are `"HND"` and `"NHD"`
- Supported head dimensions are positive values up to `128`
- `qk_quant_dtype` supports `int8` and `fp8`
- The default quantization recipe is `(128, 16, -1, 1)`; passing
  `quant_recipe` overrides `qk_quant_gran`
- Only `qk_quant_gran="per_thread"` is supported as a shortcut; other
  granularities should be expressed via an explicit supported `quant_recipe`
- `fp8_output=True` returns an FP8 tensor plus a `torch.float32` `out_scale`
  tensor; `out_scale` has the same public tensor layout as the output and a
  final scale dimension of `1`
- Unsupported in this wrapper package: varlen, KV-cache wrapper entrypoints,
  public INT8 wrapper entrypoints other than the SM90-compatible name, and
  low-level pre-quantized public APIs
