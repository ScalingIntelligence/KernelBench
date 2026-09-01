![Torch MUSA_Logo](docs/pdf/images/torch_musa.png)
--------------------------------------------------------------------------------

> **Language**: English | [中文](README_zh.md)

<!-- toc -->

- [Overview](#overview)
- [Usage](#usage)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Docker Image](#docker-image)
  - [From Python wheels](#from-python-wheels)
  - [From Source](#from-source)
- [MUSA Supported Repositories](#musa-supported-repositories)
  - [torchvision](#torchvision)
  - [torchaudio](#torchaudio)
  - [Other Repositories](#other-repositories)
- [Trouble Shooting](#trouble-shooting)
- [License](#license)

<!-- tocstop -->

## Overview

**torch_musa** is an extended Python package based on PyTorch. Combined with PyTorch, users can take advantage of the strong power of Moore Threads graphics cards through **torch_musa**.

**torch_musa**'s APIs are consistent with PyTorch in format, which allows users accustomed to PyTorch to migrate smoothly to **torch_musa**, so for the usage users can refer to [PyTorch Official Doc](https://docs.pytorch.org/docs/stable/index.html), all you need is just switch the backend string from `cpu` or `cuda` to `musa`.

**torch_musa** also provides a bundle of tools for users to conduct cuda-porting, building musa extension and debugging. Please refer to [README.md](torch_musa/utils/README.md).

For some customize optimizations, like **Dynamic Double Casting** and **Unified Memory Management**, please refer to [README.md](torch_musa/README.md).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate. No wrapper code needs to be written. Refer to [ResNet50 example](torch_musa/examples/cpp/README.md).

--------------------------------------------------------------------------------

## Usage

**We recommend users refer to [torchada](https://github.com/MooreThreads/torchada), which enable your torch scripts written with CUDA run directly on our MUSA platform.**

Now `import torch` would automatically load torch_musa, and in most cases, one just need to switch the backend from **cuda** to **musa**:
```Python
import torch
torch.musa.is_available()  # should be True

# Creating tensors:
a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
b = torch.tensor([1.2, 2.3], dtype=torch.float32, device='cpu').to('musa')
c = torch.tensor([1.2, 2.3], dtype=torch.float32).musa()

# Also some cuda modules or functions:
torch.backends.mudnn.allow_tf32 = True
event = torch.musa.Event()
stream = torch.musa.Stream()
```

For distribute training or inference, initialize your process group with backend **mccl**:
```Python
import torch.distributed as dist

dist.init_process_group("mccl", rank=rank, world_size=world_size)
```

### Docker Image
We provide several released docker images and they can be easily found in [mcconline](https://mcconline.mthreads.com/repo),
and encourage developers to utilize torch_musa within docker environment.

Pull the docker image with `docker pull`, create a container with command below and there you go.
```bash
# For example, start a S80 docker with Python3.10
docker run -it --privileged \
  --pull always --network=host \
  --env MTHREADS_VISIBLE_DEVICES=all \
  registry.mthreads.com/mcconline/musa-pytorch-release-public:latest /bin/bash
```
During its initial startup, it performs a self-check, so you can see the MUSA environment is ready or not.


## Installation
### Prerequisites
Before installing torch_musa, here are some software packages that need to be installed on your machine first:
- **[MUSA-SDK](https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)**, including MUSA Driver, musa_toolkit, muDNN and MCCL(*S4000 only*);
- **Math-Libs**, including [muThrust](https://github.com/MooreThreads/muThrust) and [muAlg](https://github.com/MooreThreads/muAlg);
- (For Docker users) **[KUAE Cloud Native Toolkits](https://developer.mthreads.com/sdk/download/CloudNative?equipment=&os=&driverVersion=&version=)**, including Container-Toolkit, MTML, sGPU;

 Run `musaInfo` to check if the MUSA environment is set correctly, if it outputs error log instead of device information, refer to [Trouble Shotting](#trouble-shooting).

### From Python wheels
Download torch & torch_musa wheels from our [Release Page](https://github.com/MooreThreads/torch_musa/releases),
please make sure you have all prerequisites installed.

### From Source
Firstly, clone the torch_musa repository:
```bash
git clone https://github.com/MooreThreads/torch_musa.git
cd torch_musa
```

then, we need to set `MUSA_HOME`, `PATH`, `LD_LIBRARY_PATH` and `PYTORCH_REPO_PATH` for building torch_musa:
```bash
export MUSA_HOME=path/to/musa/install/path  # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
export PATH=$MUSA_HOME/bin:$PATH

# if PYTORCH_REPO_PATH is not set, PyTorch will be downloaded outside this directory when building with build.sh
export PYTORCH_REPO_PATH=/path/to/PyTorch
```

To build torch_musa, run:
```bash
bash build.sh -c  # clean cache then build PyTorch and torch_musa from scratch
```

Some important building parameters are as follows:
 - --torch/-t: build original PyTorch only
 - --musa/-m: build torch_musa only
 - --debug: build in debug mode
 - --asan: build in asan mode
 - --clean/-c: clean everything built and build
 - --wheel/-w: generate wheels

For example, if one has built PyTorch and only needs to build torch_musa and generate wheel, run:
```bash
bash build.sh -m -w
```
For **S80/S3000** users, since the MCCL library is not provided for such architectures, please add `USE_MCCL=0` whilst building torch_musa:
```bash
USE_MCCL=0 bash build.sh -c
```

## MUSA Supported Repositories

### torchvision
For torch_musa v2.7.0 and later, install torchvision from [our repository](https://github.com/MooreThreads/vision):
```shell
git clone https://github.com/MooreThreads/vision -b 0.26.0.post1 --depth 1
cd vision && python setup.py install
```

Otherwise, install torchvision from [torch repository](https://github.com/pytorch/vision):
```shell
git clone https://github.com/pytorch/vision -b ${version} --depth 1
cd vision && python setup.py install
```
the `version` depends on torch version, for example you have torch v2.5.0, `${version}` should be `0.20.0`.

### torchaudio
Install torchaudio from [torch source](https://github.com/pytorch/audio):
```shell
git clone https://github.com/pytorch/audio.git -b release/${version} --depth 1
cd audio && python setup.py install
```
the `version` is same as the torch version.

### Other Repositories
Many repositories have supported MUSA backend upstream,
like [Transformers](https://github.com/huggingface/transformers.git), [Accelerate](https://github.com/huggingface/accelerate.git),
you can install them from PyPi with `pip install [repo-name]`.

For others that haven't supported musa, we musified them and put into our [GitHub](https://github.com/MooreThreads), here's the list:
| Repo | Branch | Link |
| :-- | :-: | :-: |
| pytorch3d | musa-dev | https://github.com/MooreThreads/pytorch3d |
| pytorch_sparse | master | https://github.com/MooreThreads/pytorch_sparse |
| pytorch_scatter | master | https://github.com/MooreThreads/pytorch_scatter |
| pytorch_cluster | musa-dev | https://github.com/MooreThreads/pytorch_cluster |
| torchvision | 0.26.0.post1 | https://github.com/MooreThreads/vision |
| pytorch_lightning | musa-dev | https://github.com/MooreThreads/pytorch-lightning |

Each repository can be built and install via `python setup.py install` or `pip install . --no-build-isolation`.

If users find any question about these repos, please file issues in torch_musa, and if anyone  musify a repository, you can
submit a Pull Request that helping us to expand this list.

## Trouble Shooting

- **`torch.musa.is_available()` returns False**
  - **Possible cause**: MUSA driver/runtime is not installed or not loaded correctly, or required shared libraries are not found in the current environment.
  - **Suggested fix**: Make sure [MUSA-SDK](https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=) is installed and available, and check that `MUSA_HOME` and `LD_LIBRARY_PATH` (or corresponding paths inside your container) include directories for `musa_toolkit`/`muDNN`, etc.

- **Import error: `ImportError: libmusa...so: cannot open shared object file` / missing `libmudnn.so` / `libmccl.so`**
  - **Possible cause**: Shared library search path is not configured, or the wheel/package does not match the system libraries.
  - **Suggested fix**: Configure (example):
    - `export MUSA_HOME=/usr/local/musa`
    - `export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH`
    and ensure your driver/toolchain versions are compatible with the torch_musa version you are using.

- **Import error: `ImportError: libmkl...so: cannot open shared object file` / missing `libmkl.so`**
  - **Possible cause**: `libmkl.so` is needed from pytorch, and you could be using pre-built wheel with MKL installed.
  - **Suggeseted fix**: Uninstall torch, and rebuild it from source (run `bash build.sh -t`).

- **Distributed init fails: `dist.init_process_group("mccl", ...)` (backend not available / not found)**
  - **Possible cause**: MCCL is not installed, or the current architecture does not support it.
  - **Suggested fix**: Make sure MCCL is installed (*S4000 only*). On architectures without MCCL (e.g. **S80/S3000**), build with `USE_MCCL=0` when needed and use a backend that is available in your environment.

- **When building from source, PyTorch is downloaded outside the repo / unexpected build location**
  - **Possible cause**: `PYTORCH_REPO_PATH` is not set.
  - **Suggested fix**: Set `export PYTORCH_REPO_PATH=/path/to/PyTorch` before building (see “From Source” above).

- **No device visible inside container / “no device” (or similar) at runtime**
  - **Possible cause**: Device nodes are not passed into the container, or device visibility is not configured.
  - **Suggested fix**: Start the container using the example flags in this README, and ensure `MTHREADS_VISIBLE_DEVICES=all` (or a specific device ID list) is set.

## License
torch_musa has a BSD-style license, as found in the [LICENSE](LICENSE) file.
