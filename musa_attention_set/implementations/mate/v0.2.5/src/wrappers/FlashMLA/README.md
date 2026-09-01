# FlashMLA Compatibility Wrapper (flash_mla)

`flash_mla` is a compatibility wrapper package that preserves the official
`flash_mla` package name and import path while running on MUSA through MATE
Multi-head Latent Attention (MLA) operators.

## Overview

This wrapper is designed for projects that already target the FlashMLA Python
API. It lets you run MLA dense decode, sparse decode, and sparse prefill
workloads on MUSA with minimal integration changes.

The current compatibility scope includes `FlashMLASchedMeta`,
`get_mla_metadata`, `flash_mla_with_kvcache`, and `flash_mla_sparse_fwd`.

## Package and import

- Package name: `flash_mla`
- Import path: `flash_mla`
- Runtime backend: MATE MLA operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.4+musa`. Use `python -m pip show flash_mla` to distinguish this
wrapper from the native package.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.

## Build

Build a wheel from the `wrappers/FlashMLA` directory:

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
python -m pip install flash_mla \
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
python -m pip install --no-deps dist/flash_mla-*.whl
```

## Import

Import the package directly:

```python
import flash_mla
```

Import individual APIs:

```python
from flash_mla import (
    FlashMLASchedMeta,
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_mla_sparse_fwd,
)
```

## Behavior

- `get_mla_metadata(...)` follows the current upstream FlashMLA Python interface and returns `(FlashMLASchedMeta(), None)`.
- The real scheduler tensors are initialized lazily on the first `flash_mla_with_kvcache(...)` call and cached inside `FlashMLASchedMeta`.
- Reusing the same `FlashMLASchedMeta` requires the same decode configuration across calls.
- `flash_mla_with_kvcache(...)` is the dense/sparse decode entry. The wrapper validates `FlashMLASchedMeta`, lazily materializes the real scheduler with `mate.flashmla.get_mla_metadata(...)`, and then forwards to `mate.flashmla.flash_mla_with_kvcache(...)`.
- `flash_mla_sparse_fwd(...)` is the sparse MLA prefill entry.


## Notes

- This wrapper keeps the official FlashMLA import surface, but execution is provided by MATE on MUSA.
- For the authoritative MLA operator behavior, refer to `mate.flashmla`.
