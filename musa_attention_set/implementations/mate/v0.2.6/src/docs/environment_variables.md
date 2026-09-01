# Environment Variables

MATE reads configuration from the current process environment. Set variables
before launching the Python process or CLI command that should observe them.

Use `mate env` to print the MATE-related variables visible to the current
shell.

## API Logging and Dumps

The logging and dumping configuration is read when `mate.api_logging` is
imported.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_LOGLEVEL` | `0` | API logging level: `0`, `1`, `3`, `5`, or `10` |
| `MATE_LOGDEST` | `stdout` | Log destination: `stdout`, `stderr`, or a file path |
| `MATE_DUMP_DIR` | `mate_dumps` | Root directory for Level 10 tensor dumps |
| `MATE_DUMP_MAX_SIZE_GB` | `20` | Maximum total dump size in GB per process |
| `MATE_DUMP_MAX_COUNT` | `1000` | Maximum number of dumped calls per process |
| `MATE_DUMP_SAFETENSORS` | `0` | Save dumps as `safetensors` instead of `torch.save` |
| `MATE_DUMP_INCLUDE` | empty | Comma-separated `fnmatch` patterns to include for dumping |
| `MATE_DUMP_EXCLUDE` | empty | Comma-separated `fnmatch` patterns to exclude from dumping |

### Log-Level Meanings

- `0`: Disabled.
- `1`: Function names only.
- `3`: Function names plus structured inputs and outputs, including tensor VA ranges.
- `5`: Level 3 plus tensor statistics.
- `10`: Level 5 plus on-disk tensor dumping for replay.

```{note}
- `MATE_LOGDEST` supports `%i` in file paths. MATE replaces it with the current process ID.
- Dumps produced with `MATE_DUMP_SAFETENSORS=1` do not preserve original stride or non-contiguous layout information. Use the default `torch.save` format when replay must preserve strides.
- Level 10 logging writes full API inputs and outputs to disk. Do not enable it for sensitive workloads unless the dump directory is appropriately protected.
```

## JIT, AOT, and Cache

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_MUSA_ARCH_LIST` | auto-detect visible devices | MUSA architecture list used by JIT/AOT workflows; accepts space-separated `major.minor` values such as `3.1` or `3.1 4.0`. |
| `MATE_WORKSPACE_BASE` | home directory | Base directory for the MATE cache workspace. |
| `MATE_DISABLE_JIT` | `0` | Disable runtime JIT and require matching AOT modules. |
| `MATE_JIT_VERBOSE` | `0` | Show verbose ninja output for runtime JIT builds. |

Runtime loading prefers a matching AOT library when one exists.
`MATE_DISABLE_JIT=1` switches to AOT-only behavior and raises an error if no
matching AOT module exists. MATE does not provide an environment variable to
bypass AOT and force runtime JIT when a matching AOT module is present.

Set `MATE_MUSA_ARCH_LIST` explicitly for offline diagnostics when no MUSA device
is visible:

```bash
MATE_MUSA_ARCH_LIST=3.1 mate module-status
```

## MUBIN Artifacts

MATE can load MUBIN metadata and kernel artifacts from an installed
`mate-mubin` package or from a downloaded artifact cache. The following
variables control the downloaded-artifact path and repository behavior.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_MUBIN_DIR` | `~/.cache/mate/mubin` | Root directory for downloaded MUBIN artifacts. Use a directory dedicated to MATE artifacts. |
| `MATE_MUBIN_REPOSITORY` | `sw-mate-public-generic` | Repository name or HTTP(S) URL used as the artifact source. |
| `MATE_MUBIN_REPOSITORY_BASE_URL` | `https://sh-repo.mthreads.com/repo/repository` | Base URL prepended when `MATE_MUBIN_REPOSITORY` is a repository name rather than an HTTP(S) URL. |
| `MATE_MUBIN_NO_DOWNLOAD` | unset | Disable automatic metadata and kernel downloads when set to any nonempty value. |
| `MATE_MUBIN_VERIFY_DISABLED` | unset | Disable kernel-map and object hash verification when set to any nonempty value. |
| `MATE_MUBIN_DOWNLOAD_VERBOSE` | `0` | Print artifact download, cache reuse, and refresh details when enabled. |

Artifact source precedence is:

1. An installed `mate-mubin` package.
2. `MATE_MUBIN_DIR`, when set.
3. The default `~/.cache/mate/mubin` directory.

An installed `mate-mubin` package takes precedence even when `MATE_MUBIN_DIR`
is set. MATE trusts the installed package contents and does not fall back to the
downloaded cache. Without the package, MATE downloads the verified kernel map
when a MUBIN-backed module is first used, then downloads the selected `.o`
kernel object lazily.
Use `mate download-mubin` to fetch the complete payload in advance and
`mate list-mubins` to inspect the active source and module status.

`MATE_MUBIN_NO_DOWNLOAD` and `MATE_MUBIN_VERIFY_DISABLED` use nonempty-string
checks. Unset them to restore normal behavior; setting one to `0` still enables
that control.
`MATE_MUBIN_DOWNLOAD_VERBOSE` uses boolean parsing and treats an empty value,
`0`, `false`, `no`, and `off` as disabled.

```{warning}
`MATE_MUBIN_VERIFY_DISABLED` also disables verification of downloaded `.o`
objects before MATE loads them. Use the hash bypass only for isolated diagnosis,
not as a persistent deployment setting.
```

`mate env` displays the cache, download, verification, and verbosity controls.
Repository and repository-base overrides are not included in that command's
output; inspect those shell variables directly when diagnosing a custom source.

## Runtime Wrapper Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_DEEPGEMM_MK_ALIGNMENT` | `128` | Override the M-axis padding alignment returned by DeepGEMM wrapper `get_mk_alignment_for_contiguous_layout()`; supported values are `128` and `256` |

Use `MATE_DEEPGEMM_MK_ALIGNMENT` when DeepGEMM-compatible contiguous grouped
GEMM call sites need a non-default M-axis alignment. Callers should pad each
expert segment and build `m_indices` using the same value returned by
`get_mk_alignment_for_contiguous_layout()`. Set this variable before starting
the Python process; the helper reads and caches the value on first use.

## Compiler and Build Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_EXTRA_CFLAGS` | empty | Extra host compiler flags for JIT builds. |
| `MATE_EXTRA_MUSAFLAGS` | empty | Extra `mcc` flags for JIT builds. |
| `MATE_EXTRA_LDFLAGS` | empty | Extra linker flags for JIT builds. |
| `MATE_MCC` | auto-detected | Override the `mcc` compiler path used by JIT builds. |
| `MATE_MUBIN_SOURCE_DIR` | unset | Build `mate-mubin` from an existing complete artifact cache instead of downloading the payload. |

MATE JIT builds also honor common build-tool variables such as `CXX` for the
host C++ compiler and `MAX_JOBS` for ninja parallelism.


## Guard Allocator Debugging

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_GUARD_ALLOCATOR_AUTO_INSTALL` | unset | Internal bootstrap flag used by `mate guard-run` |
| `MATE_GUARD_ALLOCATOR_MODE` | unset | Internal guard allocator mode for bootstrap and replay |
| `MATE_GUARD_ALLOCATOR_SYNC_ON_FREE` | unset | Internal flag controlling device sync before guarded frees |
| `MATE_GUARD_ALLOCATOR_LOG_ALLOCATIONS` | unset | Internal flag controlling guarded alloc/free logging |

The `MATE_GUARD_ALLOCATOR_*` variables are internal bootstrap details and are
usually set by MATE itself when preparing guarded child processes. Use
`mate guard-run` or the Python `mate.memory_debug.install_guard_allocator()`
API for normal guard allocator debugging. The guard allocator is host-only C++
and does not need `MATE_MUSA_ARCH_LIST`; that variable only controls MATE
JIT/AOT modules that compile MUSA kernels.

## Diagnostic and Test-Only Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_PYTEST_GUARD_ALLOC` | `0` | Enable the guarded allocator by default for MATE repository pytest runs |
| `MATE_PYTEST_GUARD_MODE` | `tail` | Default MATE repository pytest guard allocator mode: `tail` or `head` |
| `MATE_PYTEST_GUARD_LOG_ALLOCATIONS` | `0` | Log guarded alloc/free events during MATE repository pytest runs |
| `MATE_DRY_RUN` | `0` | Internal test/diagnostic mode used by MATE tests; not intended as a normal user runtime setting |
| `MATE_PYTEST_SHARD_TOTAL` | `1` | Number of shards to dispatch all the tests to |
| `MATE_PYTEST_SHARD_INDEX` | `0` | Index of current shard |
| `MATE_PYTEST_SHARD_MODE` | `file` | Shard dispatch mode. Currently, only `file` mode is supported |

When `MATE_PYTEST_SHARD_TOTAL > 1`, `test_fmha.py` tests will dominate the last shard.

The `MATE_PYTEST_GUARD_*` variables are consumed by MATE's in-repository
pytest `conftest.py`; they are not installed as a general pytest plugin for
external projects. Prefer `mate guard-run` for user-facing guarded allocator
debug sessions.
