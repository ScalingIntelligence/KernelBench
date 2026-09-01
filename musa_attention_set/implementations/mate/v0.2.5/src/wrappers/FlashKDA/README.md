# FlashKDA Compatibility Wrapper (flash_kda)

`flash_kda` is a compatibility wrapper package that preserves the official
`flash_kda` package name and import path while running on MUSA through MATE
Kimi Delta Attention (KDA) operators.

## Overview

This wrapper is designed for projects that already target the FlashKDA Python
API. It keeps the public `flash_kda` import surface and forwards execution to
`mate.kda.chunk_kda`.

The current compatibility scope includes:

- `flash_kda.fwd`
- `flash_kda.get_workspace_size`

## Package and import

- Package name: `flash_kda`
- Import path: `flash_kda`
- Runtime backend: MATE KDA operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.4+musa`. Use `python -m pip show flash_kda` to distinguish this
wrapper from the native package.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.
- Use MUSA Toolkit / MTCC 5.1.0 or newer to build the current fused chunk KDA
  path. The 4.3.6 toolchain may fail to compile these kernels.

## Build

Build a wheel from the `wrappers/FlashKDA` directory:

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
python -m pip install flash_kda \
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
python -m pip install --no-deps dist/flash_kda-*.whl
```

## Import

Import the package directly:

```python
import flash_kda
```

Import individual APIs:

```python
from flash_kda import fwd, get_workspace_size
```

## Behavior

- `fwd(...)` preserves the FlashKDA Python call signature and forwards to
  `mate.kda.chunk_kda(...)` with `use_qk_l2norm_in_kernel=True`.
- `out` and `final_state` are treated as preallocated output buffers and are
  written in place, matching the official package surface.
- `get_workspace_size(...)` is preserved for compatibility and always returns
  `0`. The wrapper does not allocate or consume an explicit workspace tensor
  because MATE manages the kernel internals itself.
- `cu_seqlens` follows the MATE runtime behavior on MUSA and accepts both
  `torch.int32` and `torch.int64`.

## Notes

- This wrapper keeps the official FlashKDA import surface, but execution is
  provided by MATE on MUSA.
- For the authoritative operator behavior, refer to `mate.kda.chunk_kda`.
