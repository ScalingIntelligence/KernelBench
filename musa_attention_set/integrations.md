# MUSA Attention 集成进展（不重复归档内核）

下列项目证明 MUSA Attention 已进入更多框架，但核心算子来自本集合已归档的 MATE、
TorchMUSA 或通用 ggml fattn 路径，因此不计为新的独立内核。

| 项目 | 公开进展 | 核心来源 | 处理 |
| --- | --- | --- | --- |
| [torchada](https://github.com/MooreThreads/torchada/tree/90d8d2a4429f46660f96abf6bed66dad9f15bf3d) | 将 SGLang/vLLM 的 CUDA/FA 接口重定向到 MUSA `flash_attn_interface`，并声明 FlexAttention 兼容 | MATE、TorchMUSA | 元数据登记 |
| [SGLang MUSA roadmap](https://github.com/sgl-project/sglang/issues/16565) | FA3/MATE 后端、上下文并行修复、FlashInfer sampling 等 PR 进展 | MATE、torchada | 跟踪 issue/PR，不复制 |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni/issues/2347) | MUSA FlashAttention 3 via MATE；diffusion backend 已把 MUSA 纳入 FA 路径 | MATE | 跟踪官方 roadmap，不复制 |
| [InfiniCore](https://github.com/InfiniTensor/InfiniCore) | 实验性 Moore GPU `--flash-attn` 接入，文档固定 MATE v0.1.3 | MATE v0.1.3 | 记录依赖，不复制 |
| [Paddle-MUSA](https://github.com/MooreThreads/paddle_musa) | Paddle FA API、前后向注册和 KV-cache 桥接 | muDNN/MATE | 相关桥接已选择性归档 |
| [vLLM-MUSA](https://github.com/MooreThreads/vllm-musa) | MATE/FA3/FlashMLA 后端；另含 DeepSeek-V4/GLM-5.2 专用 `.mu` 内核 | MATE + 自有内核 | 自有内核已选择性归档 |

登记项不自动继承“正确”“更快”或“适配 S4000”的结论。每个框架仍需要按其固定依赖、
模型、硬件和输入配置独立验证。
