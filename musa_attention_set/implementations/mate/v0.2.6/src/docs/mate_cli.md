# MATE CLI Reference

MATE CLI provides command-line utilities for installation checks, configuration inspection, environment diagnostics, MUBIN artifact management, guarded-memory debugging, and Level 10 dump replay.

All examples below use `mate`, but `python -m mate` is equivalent after MATE is installed correctly.

For the end-to-end guarded allocator workflow, including pytest integration and troubleshooting, see [Guard Allocator Debugging](guard_allocator.md).

## Installation

Install the base package:

```bash
pip install mate
mate --help
python -m mate --help
```

For source builds or local wheels in an environment that already has the
correct MUSA-enabled `torch` / `torch_musa` stack installed, prefer
`--no-deps`. This is especially important with `--force-reinstall`, which can
otherwise replace the MUSA PyTorch stack with the official PyPI `torch`
package.

From a local wheel:

```bash
python -m pip install --no-deps /abs/path/mate-0.1.3-cp310-cp310-linux_x86_64.whl
python -m pip install --force-reinstall --no-deps /abs/path/mate-0.1.3-cp310-cp310-linux_x86_64.whl
```

If you plan to load or replay dumps written in `safetensors` format, install the CLI extra or install `safetensors` directly.

From a package index:

```bash
pip install "mate[cli]"
```

If you only have a local wheel, use the standard direct-reference syntax instead of `./wheel.whl[cli]`:

```bash
python -m pip install "mate[cli] @ file:///abs/path/mate-0.1.3-cp310-cp310-linux_x86_64.whl"
```

If preserving an existing MUSA PyTorch stack matters more than resolving extras
through pip metadata, install the wheel with `--no-deps` first and then install
`safetensors` explicitly:

```bash
python -m pip install --force-reinstall --no-deps /abs/path/mate-0.1.3-cp310-cp310-linux_x86_64.whl
python -m pip install safetensors
```

Notes:

- The `name[extra] @ file:///...` form is the recommended pip syntax for extras with a local wheel
- `--force-reinstall` without `--no-deps` can cause pip to reinstall `torch` from package metadata, which is usually wrong in a MUSA environment
- If your environment cannot access a package index, the extra dependency must also be available locally. For `mate[cli]`, that means `safetensors`
- Installing only the MATE wheel without resolving `safetensors` is still valid, but replaying `*.safetensors` dumps will fail until `safetensors` is installed
- MATE also requires a MUSA-enabled `apache-tvm-ffi` build. Install or build it from the MooreThreads MUSA fork at `https://github.com/MooreThreads/tvm-ffi` using a release tag that contains `musa` (for example `v0.1.11.post1+musa.1`); a plain upstream TVM-FFI package is treated as an invalid runtime dependency by `mate check`

## Quick Reference

| Command | Purpose |
| --- | --- |
| `mate show-config` | Show MATE, Python, PyTorch, MUSA, JIT, and AOT information |
| `mate module-status [--detailed]` | Show registered JIT/AOT module status without compiling |
| `mate list-modules [MODULE]` | List registered JIT/AOT modules or inspect one module |
| `mate export-compile-commands [PATH]` | Export compile commands for registered JIT modules |
| `mate clear-cache` | Remove the runtime JIT cache directory only |
| `mate install-mubin-wheel` | Install the `mate-mubin` wheel matching MATE |
| `mate download-mubin` | Download complete MUBIN artifacts for all supported modules |
| `mate list-mubins` | Show the active MUBIN source and artifact status by module |
| `mate clear-mubin` | Remove the configured downloaded MUBIN artifact directory |
| `mate env` | Show relevant environment variables and their current values |
| `mate check` | Validate a usable MATE runtime environment |
| `mate guard-run -- COMMAND` | Launch a child command with the guarded MUSA allocator |
| `mate replay --dir PATH` | Replay one Level 10 dump or a directory of dumps |
| `mate list-dumps [PATH]` | List dump directories under a root directory |

Useful global options:

- `mate --help`
- `mate --version`

## Commands

### `show-config`

Show runtime and installation information:

```bash
mate show-config
mate show-config --json
```

The command reports:

- MATE version and git version
- Python version
- PyTorch version
- `torch_musa` version and whether MUSA is available
- Resolved JIT MUSA architecture list, cache key, and whether the value came from `MATE_MUSA_ARCH_LIST` or auto-detection
- JIT workspace/cache/generated-source directories, AOT directory, JIT toggles, and registered module counts
- `apache-tvm-ffi` version and whether it is a MUSA-enabled build
- MUSA device count and device names
- AOT library directory and a short sample of detected `.so` files
- Installed package versions for `mate`, `torch`, and `torch_musa`

If `apache-tvm-ffi` is missing or is not a MUSA-enabled build, `mate show-config` highlights the status in red and prints an installation hint.
If JIT architecture resolution is unavailable, `mate show-config` still succeeds and reports the reason in yellow; the JSON output includes `musa_arch_available=false` plus the error text.

Use `--json` when the output will be consumed by scripts.

### `env`

Print MATE-related environment variables:

```bash
mate env
```

This command is a read-only view of the current process environment. It is useful for checking whether shell exports are in place before launching a workload.
For variable meanings, defaults, and timing details, see {doc}`Environment Variables </environment_variables>`.

### JIT/AOT diagnostics

These commands register the default JIT/AOT module specs for diagnostics, but they do not compile kernels or run ninja:

```bash
MATE_MUSA_ARCH_LIST=3.1 mate module-status
MATE_MUSA_ARCH_LIST=3.1 mate module-status --detailed
MATE_MUSA_ARCH_LIST=3.1 mate list-modules
MATE_MUSA_ARCH_LIST=3.1 mate list-modules gemm_ops
MATE_MUSA_ARCH_LIST=3.1 mate export-compile-commands
```

`module-status` reports whether each module is backed by an AOT library, an existing runtime JIT `.so`, or no compiled library. `list-modules MODULE` prints JSON for a single module, including source files and expected AOT/JIT paths. `export-compile-commands` writes `compile_commands.json` by default for clangd/IDE tooling.

Use `MATE_MUSA_ARCH_LIST=3.1` for offline diagnostics when no MUSA device is visible. See {doc}`Environment Variables </environment_variables>` for details.

### `clear-cache`

Remove runtime JIT cache files without touching AOT libraries:

```bash
mate clear-cache
```

This command only clears the runtime JIT cache. It does not remove packaged or
downloaded MUBIN artifacts.

### MUBIN artifact management

MUBIN-backed operators use artifacts from an installed `mate-mubin` package
when one is available. Otherwise, MATE uses the directory selected by
`MATE_MUBIN_DIR`, which defaults to `~/.cache/mate/mubin`.

#### `install-mubin-wheel`

Install the `mate-mubin` package matching the running MATE public version:

```bash
mate install-mubin-wheel
```

The command runs the active Python interpreter's pip with `--upgrade`,
`--no-deps`, and the MUSA wheel index. It does not modify the installed MATE or
PyTorch packages. Use `--dry-run` to inspect the exact command:

```bash
mate install-mubin-wheel --dry-run
```

Use `--mate-version VERSION` to select an explicit MATE version and
`--index-url URL` when the optional wheel is hosted on another Python package
index. The installed MATE distribution contains the external kernel maps and
`.o` artifacts.

#### `download-mubin`

Download complete metadata and kernel payloads for `gemm`, `flash_attention`,
`flash_mla`, and `sage_attention`:

```bash
mate download-mubin
```

The command writes to `MATE_MUBIN_DIR` or the default downloaded-artifact
cache. It downloads the cache even when an installed `mate-mubin` package is
the active artifact source; it does not modify that package.

#### `list-mubins`

Show the active artifact source, root directory, and status of each supported
module:

```bash
mate list-mubins
```

The reported statuses are:

| Status | Meaning |
| --- | --- |
| `Downloaded` | Metadata and the complete kernel payload are present and verified |
| `Metadata only` | The kernel map is verified, but the complete kernel payload is not present and verified |
| `Incomplete` | The module directory exists but the kernel map is missing or fails verification |
| `Missing` | No artifact directory exists for the module |

An installed `mate-mubin` package takes precedence over `MATE_MUBIN_DIR` and
the default cache. Installed package contents are trusted and MATE does not
fall back to downloaded artifacts.

#### `clear-mubin`

Remove the configured downloaded-artifact root recursively:

```bash
mate clear-mubin
```

This command removes `MATE_MUBIN_DIR`, or `~/.cache/mate/mubin` when the
variable is unset. Point `MATE_MUBIN_DIR` at a directory dedicated to MUBIN
artifacts. An installed `mate-mubin` package is not modified or uninstalled.

#### Prepare an offline runtime

When `mate-mubin` is not installed, prefetch the complete cache before network
access is removed:

```bash
export MATE_MUBIN_DIR=/opt/mate/mubin
mate download-mubin
mate list-mubins
export MATE_MUBIN_NO_DOWNLOAD=1
```

Keep `MATE_MUBIN_DIR` set to the same location in the runtime environment.
Every required module should report `Downloaded`; `Metadata only` is not enough
for kernels that have not already been fetched.

### `check`

Validate whether the current environment is usable for MATE:

```bash
mate check
```

The command checks:

- Python version (`>= 3.9`)
- PyTorch importability
- `torch_musa` importability
- MUSA device availability
- `apache-tvm-ffi` presence and whether its version is MUSA-enabled
- MATE importability
- Presence of AOT libraries

Behavior:

- Hard errors cause exit code `1`
- Missing `apache-tvm-ffi` or installing a non-MUSA TVM-FFI build is a hard error
- Warnings such as "MUSA not available" or "AOT libraries not found" do not fail the command

### `replay`

Replay API calls captured from Level 10 logging:

```bash
mate replay --dir mate_dumps/20260310_xxxx_call0001
mate replay --dir mate_dumps/
mate replay --dir dumps/ --device cpu
mate replay --dir dumps/ --no-run
mate replay --dir dumps/ --no-compare
mate replay --dir dumps/ --verbose
mate replay --dir dumps/ --guard-alloc --guard-mode tail
```

Supported options:

| Option | Meaning |
| --- | --- |
| `--dir PATH` | Dump directory or dump-root directory to replay |
| `--device {musa,cpu,musa:N}` | Target device for loading tensors and running replay |
| `--run/--no-run` | Execute the resolved function or only reconstruct arguments |
| `--compare/--no-compare` | Compare execution results with dumped outputs |
| `-v, --verbose` | Print metadata and argument summaries for a single dump |
| `--guard-alloc/--no-guard-alloc` | Install the guarded allocator before replay on MUSA |
| `--guard-mode {tail,head}` | Select which side should use the unmapped guard page |

Replay mode depends on the path passed to `--dir`:

- If `PATH/metadata.jsonl` exists, MATE treats `PATH` as a single dump directory.
- Otherwise, MATE scans the immediate subdirectories under `PATH` and replays each one that contains `metadata.jsonl`.

Single-dump replay:

- Honors `--run`, `--compare`, and `--verbose`
- Can be used for argument reconstruction only with `--no-run`
- Prints a detailed summary for one function call

Directory replay:

- Replays each immediate child dump in sorted order
- Preserves stateful objects across calls by tracking constructor-created objects internally
- Summarizes pass/fail status for the full sequence
- Currently always executes with output comparison for batch replay; `--no-run`, `--no-compare`, and `--verbose` do not change batch behavior

Comparison behavior:

- Output comparison only checks tensor outputs
- A single tensor result is compared as `result`
- Tuple or list results are compared item by item for tensor elements only
- Comparison uses `torch.allclose(..., rtol=1e-3, atol=1e-3)`
- If the dump has inputs but no saved outputs, comparison is treated as a mismatch

Practical notes:

- `--device cpu` is useful for validating dump loading and argument reconstruction, but execution only succeeds if the target API supports CPU tensors
- Replaying `safetensors` dumps requires `safetensors` to be installed
- Dumps produced with `MATE_DUMP_SAFETENSORS=1` lose original stride and non-contiguous layout information; see {doc}`Environment Variables </environment_variables>` for dump format details
- `--guard-alloc` only works with MUSA replay targets and must be enabled before replay starts reconstructing tensors

### `guard-run`

Launch a child command with the guarded allocator installed before the target process imports code that might allocate on MUSA:

```bash
mate guard-run -- python your_script.py
mate guard-run --mode tail -- python your_script.py
mate guard-run --mode head --log-allocations -- python reproduce_bug.py
```

Supported options:

| Option | Meaning |
| --- | --- |
| `--mode {tail,head}` | Select which side of each allocation uses the unmapped guard page |
| `--log-allocations` | Print guarded alloc/free events to stderr in the child process |

Practical notes:

- `tail` is the recommended starting point because it immediately faults on writes past the logical end of an allocation
- `head` is useful when you suspect left-side underruns or negative indexing bugs
- this command bootstraps the allocator through internal environment variables before the child imports `torch_musa`
- detailed usage and pytest guidance live in [Guard Allocator Debugging](guard_allocator.md)

### `list-dumps`

List Level 10 dump directories:

```bash
mate list-dumps
mate list-dumps mate_dumps/
mate list-dumps mate_dumps/ --details
```

Behavior:

- Default root directory is `mate_dumps`
- Only immediate child directories are scanned
- A directory is considered a dump if it contains `metadata.jsonl`
- The command reads the last record in each `metadata.jsonl` file and displays its status

With `--details`, the output includes:

- Function name
- Execution status
- Timestamp
- Number of input tensors
- Number of output tensors

## Level 10 Dump Layout

When Level 10 API logging is enabled, MATE writes one subdirectory per dumped API call under the configured dump root:

```text
mate_dumps/
├── session.jsonl
├── 20260310_153045_123_pid43210_flash_attn_with_kvcache_call0001/
│   ├── metadata.jsonl
│   ├── inputs.pt
│   └── outputs.pt
└── 20260310_153045_456_pid43210_gemm_fp8_nt_groupwise_call0001/
    ├── metadata.jsonl
    ├── inputs.safetensors
    └── outputs.safetensors
```

Notes:

- Directory names include timestamp, process id, function name, and call sequence
- `metadata.jsonl` is append-only JSONL; the last line is the latest state of the dump
- `session.jsonl` aggregates the same metadata records at the dump-root level
- If a process crashes after inputs are written but before outputs are saved, the dump directory may contain only `metadata.jsonl` plus input files
- Typical `execution_status` values are `inputs_saved` and `completed`

## Environment Variables

Use `mate env` to inspect current values. For the complete list of variables,
defaults, and detailed behavior, see {doc}`Environment Variables </environment_variables>`.

## Usage Examples

### Verify an installation

```bash
mate check
mate show-config
mate env
```

### Capture a single failing API call

```bash
export MATE_LOGLEVEL=10
export MATE_DUMP_DIR=./debug_dumps
python your_script.py

mate list-dumps ./debug_dumps
mate replay --dir ./debug_dumps/20260310_xxxx_call0001 --verbose
```

### Run a debug session with the guarded allocator

```bash
mate guard-run --mode tail -- python reproduce_bug.py
mate replay --dir ./debug_dumps/20260310_xxxx_call0001 --guard-alloc --guard-mode head
MATE_PYTEST_GUARD_ALLOC=1 MATE_PYTEST_GUARD_MODE=tail pytest tests/test_gemm.py
```

This workflow is described in more detail in [Guard Allocator Debugging](guard_allocator.md).

### Limit dumps to selected APIs

```bash
export MATE_LOGLEVEL=10
export MATE_DUMP_DIR=./filtered_dumps
export MATE_DUMP_INCLUDE="*gemm*,*attention*"
export MATE_DUMP_EXCLUDE="*benchmark*"
python your_script.py
```

### Replay a full session

```bash
mate replay --dir ./debug_dumps/
```

This mode is useful when later calls depend on objects created by earlier calls.

### Write logs to a per-process file

```bash
export MATE_LOGLEVEL=5
export MATE_LOGDEST=/tmp/mate_%i.log
python your_script.py
```

## Exit Codes

Command exit behavior is not identical across all subcommands:

- `mate show-config`: returns `0` on successful execution
- `mate env`: returns `0` on successful execution
- `mate download-mubin`, `mate list-mubins`, and `mate clear-mubin`: currently report ordinary failures in command output and do not guarantee a non-zero exit code
- `mate check`: returns `1` only when hard errors are found
- `mate guard-run`: returns the child process exit code, including signal-based termination
- `mate replay`: returns `1` on replay failure, execution failure, mismatch, or invalid replay setup
- `mate list-dumps`: informational command; missing directories or empty results are currently reported in output and do not force a non-zero exit code

## Troubleshooting

- `Failed to load MATE API logging module`: ensure the installed MATE package is complete and importable in the current Python environment
- `safetensors not installed`: install `safetensors`, or reinstall from a package index with `pip install "mate[cli]"`, or reinstall from a local wheel with `python -m pip install "mate[cli] @ file:///abs/path/your-mate.whl"`
- `apache-tvm-ffi ... is not a MUSA-enabled build`: reinstall or rebuild `apache-tvm-ffi` from `https://github.com/MooreThreads/tvm-ffi` and use a release tag that contains `musa`, such as `v0.1.11.post1+musa.1`; the upstream public build is not compatible with MATE
- `No dumps found`: `mate list-dumps` and batch replay only scan immediate child directories, not nested trees recursively
- `compare_outputs=True but no output file found`: the dump is incomplete, often because the original process crashed after saving inputs
- `AOT libraries not found`: MATE may still work in JIT mode, but startup behavior can differ from an AOT-enabled installation
- `Kernel map hash mismatch` or `Hash mismatch for MATE MUBIN artifact`: clear and redownload the MUBIN cache, then confirm the configured repository; do not disable hash verification as a normal workaround
- `MATE MUBIN artifact is not available locally`: unset `MATE_MUBIN_NO_DOWNLOAD` to permit retrieval, or run `mate download-mubin` before entering the offline environment
- MUBIN download failures: set `MATE_MUBIN_DOWNLOAD_VERBOSE=1`, then check `MATE_MUBIN_REPOSITORY`, `MATE_MUBIN_REPOSITORY_BASE_URL`, and network access
- Replay mismatches do not always mean argument reconstruction failed; they can also reflect runtime differences, device differences, or numerical drift
- Guard allocator failures, pytest defaults, and graph-capture limitations are documented in [Guard Allocator Debugging](guard_allocator.md)

## Security Note

Level 10 logging writes full API inputs and outputs to disk. Do not enable it for sensitive workloads unless the dump directory is adequately protected.
