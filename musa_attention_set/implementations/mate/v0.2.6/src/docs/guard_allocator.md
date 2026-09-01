# Guard Allocator Debugging

MATE provides a guarded MUSA allocator that can replace the default `torch_musa` allocator while you debug device-memory corruption in MUSA workloads. The guarded allocator reserves an extra virtual-memory page for every allocation, leaves that page unmapped, and fills the non-guarded slack region with a canary pattern. This makes out-of-bounds accesses much easier to localize than with the default caching allocator.

Use this allocator when you suspect:

- tensor overrun or underrun in a custom kernel
- stale pointer reuse across long-running MUSA workloads
- framework integration bugs that only reproduce on device memory

This is a debug tool, not a production allocator. Expect higher overhead and stricter limitations than the default `torch_musa` allocator.

## Detection Model

Each guarded allocation has two protections:

- one side uses an unmapped guard page and typically faults as soon as the kernel touches it
- the other side uses a canary-filled region and is checked when the allocation is released

Because only one side can use the unmapped page at a time, MATE exposes two modes:

| Mode | Immediate fault side | Deferred canary side | Best for |
| --- | --- | --- | --- |
| `tail` | overflow past the logical end of the allocation | underrun before the allocation | the common "write past the end" case |
| `head` | underrun before the logical start of the allocation | overflow past the end | negative indexing bugs and left-side underruns |

Recommended workflow:

1. Start with `tail` mode.
2. If the bug disappears or you only see canary failures on free, rerun with `head`.
3. Compare the two runs to decide whether the first invalid access is before or after the logical tensor range.

## Quick Start

Run any Python entry point with the guarded allocator installed before imports create MUSA allocations:

```bash
mate guard-run --mode tail -- python your_script.py
```

If you want allocator lifecycle logging as well:

```bash
mate guard-run --mode tail --log-allocations -- python your_script.py
```

For replaying a previously dumped workload:

```bash
mate replay --dir mate_dumps/20260310_xxxx_call0001 --guard-alloc --guard-mode tail
mate replay --dir mate_dumps/ --guard-alloc --guard-mode head
```

`mate replay --guard-alloc` only supports replay targets on MUSA devices.

## Python API

Install the allocator programmatically before the first MUSA allocation in the process:

```python
import torch

from mate.memory_debug import GuardAllocatorConfig, install_guard_allocator

install_guard_allocator(
    GuardAllocatorConfig(
        mode="tail",
        sync_on_free=True,
        log_allocations=False,
    )
)

x = torch.empty(4097, device="musa")
```

Key points:

- call `install_guard_allocator(...)` before any `torch.empty(..., device="musa")`, model load, or lazy module init that allocates on MUSA
- repeated installs with the same config are ignored
- repeated installs with a different config raise an error

`GuardAllocatorConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `mode` | `tail` | Which side gets the unmapped guard page: `tail` or `head` |
| `sync_on_free` | `True` | Synchronize the device before validating canaries during free |
| `log_allocations` | `False` | Print guarded alloc/free messages to stderr |

## Pytest Integration

The test-suite plugin in `tests/conftest.py` can enable the guarded allocator before test collection.

### One-off runs

```bash
pytest tests/test_memory_debug.py --guard-alloc --guard-mode tail
pytest tests/test_gemm.py -k fp8 --guard-alloc --guard-mode head
pytest tests/test_memory_debug.py --guard-alloc --guard-log-allocations
```

Supported pytest options:

- `--guard-alloc` / `--no-guard-alloc`
- `--guard-mode {tail,head}`
- `--guard-log-allocations` / `--no-guard-log-allocations`

### Enable it by default for pytest

If you want plain `pytest` to start in guarded mode for a debug session, export the public pytest env vars before launching the test process:

```bash
export MATE_PYTEST_GUARD_ALLOC=1
export MATE_PYTEST_GUARD_MODE=tail
pytest tests/test_gemm.py
```

To switch sides without editing shell history:

```bash
MATE_PYTEST_GUARD_ALLOC=1 MATE_PYTEST_GUARD_MODE=head pytest tests/test_gemm.py
```

You can also make it the default pytest option set through pytest itself:

```ini
[pytest]
addopts = --guard-alloc --guard-mode tail
```

or:

```bash
export PYTEST_ADDOPTS="--guard-alloc --guard-mode tail"
```

If a shell or `addopts` default enables guard mode, temporarily disable it with:

```bash
pytest --no-guard-alloc
```

### What pytest does automatically in guard mode

When guard mode is active, the plugin:

- installs the guarded allocator before test collection
- auto-skips parametrized test cases where `use_graph=True`
- auto-skips tests marked with `@pytest.mark.guard_incompatible`

Use the marker for tests that are known to be incompatible with this allocator:

```python
import pytest


@pytest.mark.guard_incompatible
def test_requires_regular_allocator():
    ...
```

Skip reason:

```text
guard allocator debug mode does not support MUSA graph capture
```

## Public vs Internal Environment Variables

The public pytest-facing controls are:

| Variable | Meaning |
| --- | --- |
| `MATE_PYTEST_GUARD_ALLOC` | Enable guarded pytest mode by default (`0` or `1`) |
| `MATE_PYTEST_GUARD_MODE` | Default pytest guard mode: `tail` or `head` |
| `MATE_PYTEST_GUARD_LOG_ALLOCATIONS` | Enable alloc/free logging during pytest (`0` or `1`) |

MATE also uses internal bootstrap variables such as `MATE_GUARD_ALLOCATOR_AUTO_INSTALL` and `MATE_GUARD_ALLOCATOR_MODE` when `mate guard-run` or the bootstrap shim prepares a child process. Treat those as implementation details rather than user-facing configuration.

## Limitations

- `torch.musa.MUSAGraph` capture is intentionally unsupported while the guarded allocator is active
- the allocator must be installed before the first MUSA allocation in the process
- one side faults immediately; the opposite side is detected later when the allocation is freed and its canary is checked
- debug overhead is higher than the default allocator because each allocation reserves extra virtual address space, may synchronize on free, and may force more deterministic failure behavior

## Troubleshooting

### "Guard allocator must be installed before the first MUSA allocation in the process"

Some earlier import or setup path already allocated on MUSA. Move guard installation earlier or use `mate guard-run` so the bootstrap happens before your program imports the code that allocates.

### "does not support torch.musa.MUSAGraph capture"

This is expected. Disable graph capture for the debug run, or rely on the pytest auto-skip behavior for graph-only tests.

### The process crashes without a Python traceback

That often means the unmapped guard page caught the first invalid device access. Rerun with `--guard-log-allocations` to correlate allocation ids and pointer ranges, or switch from `tail` to `head` to test the opposite side.

### The run only fails during free with a canary corruption message

The invalid access landed on the non-guarded side. Repeat the same workload with the opposite mode:

- `tail` canary corruption suggests trying `head`
- `head` canary corruption suggests trying `tail`

### JIT build or allocator bootstrap fails

Make sure the runtime can compile or load the guard allocator binary. The guard allocator is a host-only C++ shared library that calls MUSA runtime and driver APIs, so it needs the MUSA Toolkit headers/libraries and a working host compiler, but it does not need `MATE_MUSA_ARCH_LIST`. See [Environment Variables](environment_variables.md) for the full reference.
