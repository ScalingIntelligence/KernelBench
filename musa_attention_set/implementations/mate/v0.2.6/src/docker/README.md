# MATE development container

The public development image is based on
`registry.mthreads.com/mcconline/musa_sdk:5.2.0-devel-ubuntu22.04-s5000`.
It contains the MUSA Python stack, build and test tools, muAlg, muThrust,
Miniforge/Python 3.10, Node LTS, Codex, Claude Code, OpenCode, and common
terminal productivity and network-diagnostic tools. MATE source is
bind-mounted at runtime; it is never copied into or installed while the image
is built.

The image is split into two layers:

- `mate-dev-base:local` contains the reusable, UID/GID-independent toolchain
  under `/opt/mate`.
- `mate-dev:uid-<uid>-gid-<gid>` is a small per-user layer containing
  `devuser`, its home directory, and a system-site-packages virtual
  environment.

Changing `USER_UID` or `USER_GID` therefore rebuilds only the user layer, not
Apt, Conda, PyPI, or Node dependencies.

## Prerequisites

Install Docker with the Moore Threads `mthreads` runtime and Docker Compose
v2. The runtime must be registered in the Docker daemon before starting the
container.

## Build and start

Run the helpers from the repository root:

```bash
make -C docker build
make -C docker up
make -C docker shell
```

`build` evaluates the shared base and user layers, using Docker's cache for an
unchanged base. The individual targets are also available:

```bash
make -C docker base-build
make -C docker user-build
```

The Makefile passes the current host UID/GID, uses a per-user image tag, and
sets a per-user Compose project name. To invoke Compose directly on a shared
Docker daemon, set all of them explicitly:

```bash
export MATE_USER_UID="$(id -u)"
export MATE_USER_GID="$(id -g)"
export MATE_DEV_IMAGE="mate-dev:uid-${MATE_USER_UID}-gid-${MATE_USER_GID}"
export COMPOSE_PROJECT_NAME="mate-${MATE_USER_UID}"
docker compose build dev
docker compose up -d --no-build dev
```

`make -C docker up` builds the user image only when it is missing. Run
`make -C docker build` explicitly after changing a Dockerfile or pinned
dependency.

The container uses host networking and IPC, the `mthreads` runtime,
`SYS_PTRACE`, an unconfined seccomp profile, and development-oriented memlock
and stack limits. It does not request `SYS_ADMIN`. Existing `video` and
`render` groups are used without changing their GIDs. If the base image lacks
the standard render group, it is created at GID 109; no configurable
`VIDEO_GID` or `RENDER_GID` remapping is performed.

## Ports and additional mounts

The default service uses host networking. A process listening on port `8000`
inside the container is therefore already reachable at `localhost:8000` on
the host; do not add a Compose `ports` entry while `network_mode: host` is in
effect.

For explicit Docker port publishing or per-developer mounts, copy the ignored
local override template and provide its inputs:

```bash
cp compose.local.yaml.example compose.local.yaml
export MATE_DEV_HOST_PORT=8888
export MATE_EXTRA_SOURCE_DIR=/absolute/path/on/host
export MATE_COMPOSE_OVERRIDE=compose.local.yaml
make -C docker config
make -C docker up
```

The example switches the development service to bridge networking, publishes
host port `8888` to container port `8888`, and mounts the extra directory at
`/workspace/extra`. Edit the copied file for additional ports or mount
targets. If only an extra mount is needed, remove `network_mode` and `ports`
from the copy to retain host networking. `compose.local.yaml` is ignored by
Git and the Docker build context so host-specific paths are not published or
embedded in an image.

## Initialize the checkout

Initialization is deliberately explicit so opening a container never changes
the checkout or Git hooks:

```bash
make -C docker setup
```

This initializes submodules and performs the documented editable install with
`--no-build-isolation --no-deps`. It does not install pre-commit hooks.
The editable installation is written to `/home/devuser/.venv`; the shared
Conda environment and MUSA packages under `/opt/mate` remain unchanged.
Individual operations remain available inside the container:

```bash
git submodule update --init --recursive
python -m pip install --no-build-isolation --no-deps -e . -v
```

## Test and lint

```bash
make -C docker test
make -C docker lint
make -C docker config
make -C docker down
```

The project-scoped `mate-dev-cache` volume persists compiler and Python
caches. For the Makefile defaults its Docker name is
`mate-<uid>_mate-dev-cache`. Remove it explicitly only when a fully cold
environment is required.

## VS Code Dev Containers

Open the repository and choose **Dev Containers: Reopen in Container**. The
configuration reuses `compose.yaml`, runs as `devuser`, and mounts the host
`~/.ssh` directory read-only. No post-create command mutates the checkout.
VS Code uses `/home/devuser/.venv/bin/python` and can update the thin user
layer's UID without touching the shared tools under `/opt/mate`.

## Reuse a base image from a registry

Base images may be distributed with an immutable version tag or digest. The
build never pushes automatically. To consume an existing base image:

```bash
export MATE_DEV_BASE_IMAGE=registry.internal.example/mate-dev-base:musa5.2.0-py310-v1
make -C docker base-pull
export MATE_DEV_BASE_CONTEXT="docker-image://${MATE_DEV_BASE_IMAGE}"
make -C docker user-build
```

Without `MATE_DEV_BASE_CONTEXT`, Compose builds or reuses the local
`dev-base` service. A direct BuildKit invocation can supply the same named
context with `--build-context mate_dev_base=docker-image://...`.

## Package sources

General Python packages are installed from the Tsinghua PyPI mirror. This
includes MKL 2024.0.0, which supplies the `.so.2` and iJIT ABI required by the
MUSA torch wheel. The MUSA-specific torch, torch_musa, TVM-FFI, DLPack,
TileLang, and Triton wheels are installed in a separate `--no-deps` step from:

```text
https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

The current development base pins `apache-tvm-ffi` to
`0.1.11.post1+musa.1` and `tilelang-musa` to `0.1.12+musa.2`. The TileLang
local version satisfies MATE's `tilelang-musa==0.1.12` project requirement.

The build runs `pip check` and validates every MUSA distribution version from
installed package metadata. It does not import GPU modules during the build.
The base environment is root-owned. Additional Python packages belong in the
user virtual environment, and additional global npm packages use the
user-owned `~/.local` prefix. Only the MKL/OpenMP libraries required by Torch
are exposed through `/opt/mate/runtime-lib`; the complete Conda library
directory is not placed on the final runtime library path, so it cannot
override Ubuntu OpenSSL for Git and other system tools.

The base also provides `rg`, `fd`, `fzf`, `bat`, `tree`, `lsof`, `strace`,
`zip`, `ip`, `ss`, `ping`, `nc`, `dig`, and `nslookup`. Ubuntu packages name
the `fd` and `bat` executables `fdfind` and `batcat`; stable `fd` and `bat`
aliases are installed under `/opt/mate/bin`.
