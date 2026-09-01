# Paddle MUSA

[中文](#中文) | [English](#english)

---

## 中文

### 项目介绍

Paddle MUSA 是摩尔线程面向飞桨（PaddlePaddle）深度学习框架打造的 MUSA GPU 硬件后端适配项目。项目基于 PaddlePaddle 主线源码与 Custom Device 扩展机制，将摩尔线程 GPU 的计算、存储、通信与运行时能力接入飞桨生态，使开发者能够在保持飞桨主线 API 使用方式基本不变的前提下，将训练、推理和模型部署任务迁移到 MUSA 设备上运行。

Paddle MUSA 围绕飞桨框架与摩尔线程的S5000显卡硬件能力之间的全链路打通，覆盖设备发现与管理、运行时调度、算子 kernel 注册与调用、MUDNN/MCCL 等基础库集成、分布式通信、性能分析以及模型验证等关键环节。通过自定义的[规则化机制](backends/musa/hack/README.md)管理和 CUDA 的差异，降低 Paddle 主线代码升级适配成本。

当前项目已完成对 PaddlePaddle 源码仓库主线的适配，支持与 PaddlePaddle 主框架联合构建并产出单一安装包，便于在训练、推理和模型验证场景中直接使用。适配内容覆盖运行时、设备上下文、基础算子注册、MUSA/MUDNN/MCCL 依赖集成、Paddle 主线规则化源码适配以及单元测试与模型冒烟验证等模块。

### 适配状态

| 项目 | 当前状态 |
| --- | --- |
| 主线适配 | 支持 PaddlePaddle 主线源码联合构建，随主线持续适配 |
| 算子适配规模 | 972 类算子 |
| 算子适配覆盖率 | 91%（以 Paddle GPU/CUDA 1067 类算子为 100% 基准统计） |
| 模型适配规模 | 44 个飞桨生态模型 |
| 覆盖领域 | 6 类模型与应用场景 |
| 认证情况 | 已取得百度飞桨官方三级适配认证 |

### 已验证模型与覆盖领域

目前 Paddle MUSA 已适配并验证 44 个飞桨生态模型，覆盖大语言模型、多模态、OCR、视觉分类、检测、分割、姿态估计、检索识别与推荐系统等主流场景。

| 领域 | 代表模型 |
| --- | --- |
| 大语言模型与多模态模型 | ERNIE-4.5、LLaMA、PaddleOCR-VL 等 |
| OCR 系列模型 | PP-OCRv3、PP-OCRv4、PP-OCRv5 等 |
| 视觉分类模型 | AlexNet、VGG11、ResNet18、ResNet50、PP-LCNet、PP-HGNetV2 等 |
| 检测、分割与姿态估计模型 | SOLOv2、OCRNet、DMNet、UNet、PP-TinyPose 等 |
| 检索与识别模型 | PP-ShiTuV2、MobileFaceNet 等 |
| 推荐系统模型 | DLRM、DeepFM 等 |

### 环境准备

#### 基础依赖

请确保构建环境中已安装或可访问以下组件：

- MUSA SDK 4.3.5 及以上版本，推荐使用 MUSA SDK 5.1.0；
- Python 3.10；
- GCC / G++、CMake、Make、Git、pip 等基础构建工具；
- 足够的磁盘空间与内存用于编译 PaddlePaddle 主框架。

#### Docker 镜像

为简化环境搭建，提供了以下预配置的 Docker 镜像：

- **编译构建镜像**  
  包含编译 Paddle-MUSA 所需的完整依赖，可直接用于构建与安装。  
  `registry.mthreads.com/mcctest/ai/musa-paddle-dev:4.3.5_paddle_musa_release_0_deb_2026-07-02_ubuntu`

- **模型体验镜像**  
  已安装 Paddle-MUSA 并集成已验证的模型代码，覆盖 6 类应用场景。镜像内提供了各领域的运行说明：  
  `registry.mthreads.com/mcctest/ai/musa-paddle-dev:paddle_musa_model_zoo_release_20260703`  
  - OCR、检测、分割等视觉模型：`/home/PaddleX/README_MUSA_MODEL_RUN.md`  
  - 大语言模型与多模态模型：`/home/PaddleFormers/README_MUSA_MODEL_RUN.md`  
  - 推荐系统模型：`/home/PaddleRec/PaddleRec/README_MUSA_CRITEO.md`

#### 获取源码

```bash
git clone --recursive <paddle_musa_repo_url> paddle_musa
cd paddle_musa

git submodule sync
git submodule update --remote --init --recursive
```

本仓库依赖 PaddlePaddle 主框架源码。通常目录结构如下：

```text
paddle_musa/
├── Paddle/
└── backends/
    └── musa/
```

### 编译与安装

推荐使用联合构建方式，该方式会将 PaddlePaddle 主框架与 paddle_musa 后端一起构建，并安装最终生成的 wheel 包。为保证首次构建或主线升级后的构建环境干净一致，建议先执行清理与规则化源码适配，再执行联合构建：

```bash
cd backends/musa

# 1. 清理 PaddlePaddle 与 paddle_musa 历史构建产物，避免旧的 CMake 缓存、wheel 或中间文件影响新构建
bash tools/build.sh -c

# 2. 重置 Paddle 源码并应用 MUSA 规则化源码适配（JSON 替换规则与 .hackdiff 结构化替换规则），确保 Paddle 源码处于 MUSA 适配后的预期状态
bash tools/build.sh -ap

# 3. 执行联合构建并安装最终 wheel 包
bash tools/build.sh -u
```

其中，`-c` 用于清理历史构建产物，降低缓存和旧文件导致的构建问题；`-ap` 用于在构建前将 MUSA 规则化源码适配应用到 Paddle 主线源码，包括可复用的 JSON 语义替换规则和面向结构化改动的 `.hackdiff` 替换规则，避免主线源码与后端适配不一致；`-u` 用于联合构建 PaddlePaddle 与 paddle_musa，并产出推荐使用的单一安装包。

构建完成后，脚本会从 `Paddle/build/python/dist` 中选择最新的 wheel 包并执行安装。默认包名为 `paddlepaddle-musa`，也可以通过 `PADDLE_PYTHON_PACKAGE_NAME` 环境变量覆盖。

#### 其他构建命令

```bash
# 构建并安装 PaddlePaddle 主框架与 paddle_musa 后端
bash tools/build.sh -a

# 仅构建并安装 PaddlePaddle 主框架
bash tools/build.sh -p

# 仅构建并安装 paddle_musa 后端（依赖 PaddlePaddle 已构建完成）
bash tools/build.sh -m

# 仅应用 Paddle 主线规则化源码适配
bash tools/build.sh -ap

# 清理 PaddlePaddle 与 paddle_musa 构建产物
bash tools/build.sh -c
```

> 推荐优先使用 `bash tools/build.sh -u`。仅在需要调试独立插件构建流程时再使用 `-m`。

### 第一次构建注意事项

首次构建耗时较长，建议在构建前确认以下事项：

1. **子模块完整同步**：执行 `git submodule sync && git submodule update --remote --init --recursive`，确保 PaddlePaddle 主框架源码与第三方依赖完整。
2. **MUSA 环境可用**：确认驱动、运行时、MUSA Toolkit、MUDNN、MCCL 已正确安装，并且相关库路径可被 CMake 与运行时加载。
3. **Paddle 源码会被应用规则化源码适配**：构建脚本会在需要时对 `Paddle/` 源码应用 MUSA 适配规则，包括 JSON 语义替换规则与 `.hackdiff` 结构化替换规则；如果本地修改过 Paddle 源码，请先保存修改，避免被重置或覆盖。
4. **推荐从干净构建开始**：如遇到主线升级、依赖变化或 CMake 缓存异常，可执行 `bash tools/build.sh -c` 后重新执行联合构建。
5. **资源要求较高**：PaddlePaddle 主框架编译对 CPU、内存和磁盘空间要求较高，建议使用高并发编译环境并预留充足磁盘空间。

### 安装验证

安装完成后，可以通过以下方式确认 MUSA 后端已被 Paddle 识别：

```bash
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
```

期望输出包含：

```text
['musa']
```

也可以运行简单模型进行功能验证：

```bash
cd backends/musa
python tests/test_MNIST_model.py
```

运行过程中应能看到 loss 整体下降、accuracy 整体上升的训练日志。

### 运行单元测试

```bash
cd backends/musa

# 运行后端单元测试
bash tools/build.sh -t
```

### 基本使用方式

安装完成后，飞桨 Python API 的使用方式保持不变。用户只需将设备设置为 `musa`，即可在 MUSA 后端上创建张量并运行模型。

```python
import paddle

paddle.set_device("musa")

x = paddle.randn([2, 3])
y = paddle.nn.functional.relu(x)
print(y)
```

### 参与贡献

欢迎围绕以下方向参与贡献：

- 补充更多 MUSA 算子；
- 增加单元测试、模型测试和性能回归测试；
- 扩展 PaddleX、PaddleOCR、PaddleFormers、PaddleDetection、PaddleSeg、PaddleRec 等生态模型；
- 改进文档、构建脚本和开发者体验。

### 许可证

本项目遵循 Apache License 2.0 许可证。

---

## English

### Introduction

Paddle MUSA is Moore Threads' MUSA GPU backend adaptation project for the PaddlePaddle deep learning framework. Built on the PaddlePaddle mainline source tree and Custom Device extension mechanism, it integrates Moore Threads GPU compute, memory, communication, and runtime capabilities into the Paddle ecosystem, enabling training, inference, and model deployment workloads to run on MUSA devices while keeping standard Paddle APIs largely unchanged for users.

Paddle MUSA currently adapts to the Moore Threads S5000 accelerator card. It connects the PaddlePaddle framework with S5000 hardware capabilities end to end, covering device discovery and management, runtime scheduling, operator kernel registration and dispatch, MUDNN/MCCL library integration, distributed communication, performance profiling, and model validation. It manages differences from CUDA through a custom [rule-based mechanism](backends/musa/hack/README.md), reducing the adaptation cost of Paddle mainline code upgrades.

The project has completed adaptation to the mainline of the PaddlePaddle source repository. It supports joint build with the PaddlePaddle framework and produces a single installation package for training, inference, and model validation scenarios. The adaptation covers runtime, device context, basic operator registration, MUSA/MUDNN/MCCL dependency integration, Paddle mainline rule-based source adaptation, unit tests, and model smoke tests.

### Adaptation Status

| Item | Status |
| --- | --- |
| Mainline adaptation | Supports joint build with PaddlePaddle mainline source and continuous mainline adaptation |
| Operator adaptation scale | 975 operator categories |
| Operator coverage | 91.38% based on Paddle GPU/CUDA's 1067 operator categories as 100% |
| Model adaptation scale | 44 Paddle ecosystem models |
| Covered domains | 6 model and application domains |
| Certification | Obtained Baidu PaddlePaddle official Level-3 adaptation certification |

### Verified Models and Domains

Paddle MUSA has adapted and verified 44 Paddle ecosystem models, covering mainstream scenarios including large language models, multimodal models, OCR, visual classification, detection, segmentation, pose estimation, retrieval and recognition, and recommendation systems.

| Domain | Representative Models |
| --- | --- |
| Large language and multimodal models | ERNIE-4.5, LLaMA, PaddleOCR-VL, etc. |
| OCR models | PP-OCRv3, PP-OCRv4, PP-OCRv5, etc. |
| Visual classification models | AlexNet, VGG11, ResNet18, ResNet50, PP-LCNet, PP-HGNetV2, etc. |
| Detection, segmentation, and pose estimation models | SOLOv2, OCRNet, DMNet, UNet, PP-TinyPose, etc. |
| Retrieval and recognition models | PP-ShiTuV2, MobileFaceNet, etc. |
| Recommendation models | DLRM, DeepFM, etc. |

### Environment Preparation

#### Requirements

Please make sure the following components are available in your build environment:

- MUSA SDK 4.3.5 or later; MUSA SDK 5.1.0 is recommended;
- Python 3.10;
- GCC/G++, CMake, Make, Git, pip, and other common build tools;
- Sufficient disk space and memory for building the PaddlePaddle framework.

#### Get the Source Code

```bash
git clone --recursive <paddle_musa_repo_url> paddle_musa
cd paddle_musa

git submodule sync
git submodule update --remote --init --recursive
```

This repository depends on the PaddlePaddle framework source code. The expected directory layout is usually:

```text
paddle_musa/
├── Paddle/
└── backends/
    └── musa/
```

### Build and Installation

The recommended build mode is the union build, which builds PaddlePaddle and the paddle_musa backend together and installs the generated wheel package. To keep the build environment clean and deterministic, especially for the first build or after a Paddle mainline upgrade, it is recommended to clean previous build artifacts and apply the rule-based source adaptation before running the union build:

```bash
cd backends/musa

# 1. Clean historical PaddlePaddle and paddle_musa build artifacts to avoid stale CMake cache, wheels, or intermediate files
bash tools/build.sh -c

# 2. Reset the Paddle source tree and apply MUSA rule-based source adaptations (JSON replacement rules and .hackdiff structured replacement rules) so the Paddle source matches the expected MUSA-adapted state
bash tools/build.sh -ap

# 3. Run the union build and install the final wheel package
bash tools/build.sh -u
```

Here, `-c` cleans historical build artifacts and reduces issues caused by stale cache or old files; `-ap` applies MUSA rule-based source adaptations to the Paddle mainline source before building, including reusable JSON semantic replacement rules and `.hackdiff` structured replacement rules, preventing inconsistencies between the Paddle mainline source and backend adaptation; `-u` builds PaddlePaddle and paddle_musa together and produces the recommended single installation package.

After the build finishes, the script installs the latest wheel from `Paddle/build/python/dist`. The default package name is `paddlepaddle-musa`; it can be overridden by setting the `PADDLE_PYTHON_PACKAGE_NAME` environment variable.

#### Other Build Commands

```bash
# Build and install both PaddlePaddle and paddle_musa
bash tools/build.sh -a

# Build and install PaddlePaddle only
bash tools/build.sh -p

# Build and install paddle_musa only; requires PaddlePaddle to be built first
bash tools/build.sh -m

# Apply Paddle mainline rule-based source adaptations only
bash tools/build.sh -ap

# Clean PaddlePaddle and paddle_musa build artifacts
bash tools/build.sh -c
```

> `bash tools/build.sh -u` is recommended for normal development and validation. Use `-m` only when debugging the standalone plugin build flow.

### Notes for the First Build

The first build can take a long time. Please check the following items before building:

1. **Synchronize submodules**: run `git submodule sync && git submodule update --remote --init --recursive` to ensure the PaddlePaddle source tree and third-party dependencies are complete.
2. **Check the MUSA environment**: make sure the driver, runtime, MUSA Toolkit, MUDNN, and MCCL are installed correctly and their libraries can be found by CMake and the runtime linker.
3. **Rule-based source adaptations may be applied to Paddle**: the build scripts may apply MUSA adaptation rules to the `Paddle/` source tree, including JSON semantic replacement rules and `.hackdiff` structured replacement rules. Save your local Paddle changes before building to avoid losing modifications.
4. **Start from a clean build when needed**: after a Paddle mainline upgrade, dependency change, or CMake cache issue, run `bash tools/build.sh -c` and then rebuild with the union build mode.
5. **Reserve enough resources**: building PaddlePaddle requires significant CPU, memory, and disk resources.

### Installation Verification

After installation, verify that PaddlePaddle can discover the MUSA backend:

```bash
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
```

The expected output should include:

```text
['musa']
```

You can also run a simple model test:

```bash
cd backends/musa
python tests/test_MNIST_model.py
```

The training log should show an overall decreasing loss trend and an overall increasing accuracy trend.

### Run Unit Tests

```bash
cd backends/musa

# Run backend unit tests
bash tools/build.sh -t
```

### Basic Usage

After installation, PaddlePaddle Python APIs remain unchanged. Set the device to `musa` to create tensors and run models on the MUSA backend.

```python
import paddle

paddle.set_device("musa")

x = paddle.randn([2, 3])
y = paddle.nn.functional.relu(x)
print(y)
```

### Contributing

Contributions are welcome in the following areas:

- Add more MUSA kernels for Paddle operators;
- Improve rule-based source adaptations for Paddle mainline upgrades;
- Add unit tests, model tests, and performance regression tests;
- Extend PaddleX, PaddleOCR, PaddleFormers, PaddleDetection, PaddleSeg, PaddleRec, and other Paddle ecosystem models;
- Improve documentation, build scripts, and developer experience.

### License

This project is licensed under the Apache License 2.0.
