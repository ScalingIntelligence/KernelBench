# 检索记录与信源审计

检索截止日期：2026-09-01。

## 检索范围

本次使用中英文关键词组合检索 GitHub、Gitee、官方文档页和上游项目引用，关键词
包括 `MUSA attention`、`FlashAttention MUSA`、`FMHA MUSA`、`FlashMLA MUSA`、
`paged attention MUSA`、`sparse attention MUSA`、`scaled_dot_product_attention MUSA`
及具体源码后缀 `.mu`。随后对候选项目执行浅克隆，以 Git 对象中的 commit、标签、
目录树和根许可证作为最终证据。

优先级如下：

1. MooreThreads 官方组织仓库；
2. 原仓库 README 明确指定的维护后继仓库；
3. 上游框架官方仓库中的 MUSA 后端；
4. 能被原始算法项目交叉确认的 MUSA 移植。

博客、聚合页面、搜索摘要和学生作业仓库只用于发现线索，不作为源码归档依据。

## 关键核验结论

- `MooreThreads/mate` v0.2.6 是当前集中式高性能 Attention 库，根 README 指定
  S5000、MUSA SDK 4.3.5+ 和 TorchMUSA 2.7+；其 MUTLASS 子模块固定到本目录已
  归档的 `78b349...`。
- `MooreThreads/mutlass` 当前公开 FMHA、Paged FMHA、MLA，README 将这些路径标为
  MP31 WarpSpecialized 实现，并给出 MUSA Toolkit 4.3.4 最低要求。
- `MooreThreads/MT-flashMLA` 是 DeepSeek FlashMLA 的官方 Moore Threads 对应实现；
  DeepSeek 原仓库的 Community Support 也指向该仓库。它固定 MUTLASS v0.2.0。
- `MooreThreads/torch_musa` 的公开 SDPA 代码区分 math 与 flash 模式；其 README
  明确说明 flash 路径只覆盖部分 head dimension/half，并且所归档版本的说明不支持
  backward，因此不能泛化为训练可用。
- `MooreThreads/tilelang_musa` 已声明停止维护并迁移至 `tile-ai/tilelang-musa`。
  旧快照仍有 S4000/M1000/S5000 与 DSA/NSA 的公开代码价值；新仓库是后续更新源，
  因而两者分别归档而不互相覆盖。
- 当前 `MooreThreads/vllm-musa` 不只是 MATE 调用层：`csrc/musa/attention` 还公开了
  DeepSeek-V4 sparse FlashMLA、indexer/cache/fused-QKV 等 `.mu` 内核，所以这些路径
  单独归档；普通 FA/MLA 调用则仍记作 MATE 集成。
- `MooreThreads/paddle_musa` 的 FA 前后向与 KV-cache MATE 桥接属于框架后端，
  官方 README 的当前目标硬件是 S5000。
- `ggml-org/llama.cpp` 的 `ggml-musa/CMakeLists.txt` 会将通用 `ggml-cuda` fattn
  模板按 `mp_21;mp_22;mp_31` 目标用 MUSA 编译，因此归档了桥接文件和实际 fattn
  源码；这属于可移植后端路径，不宣称是 MooreThreads 独立手写内核。

## 仅登记、不重复归档

SGLang、vLLM-Omni、InfiniCore、torchada 等项目公开了 MUSA Attention 接入，但核心
计算复用 MATE/FlashAttention 兼容包。重复复制这些调用层会夸大独立实现数量，故只在
[integrations.md](integrations.md) 登记。

## 未纳入源码的候选

- Gitee 上出现了带 `csrc_musa/attention/paged_attention_v1.mu` 的教学/作业派生仓库，
  但当前 MooreThreads 官方 vLLM-MUSA 可见历史无法交叉确认该文件的原始 commit；
  在来源链不闭合前不归档。
- CUDA、ROCm、MetaX、DCU 等非 MUSA 实现即使算法同名，也不属于本集合。
- 只发布二进制 wheel、没有对应公开源码的算子，不作为 `source-archived` 项目。
- MusaCoder 模型和论文属于 kernel 生成研究，不是可固定版本的 Attention 算子库；
  KernelBench 内已有生成候选继续按本仓库现有基线管理。

“全网检索”只能覆盖检索时公开、可索引且可访问的内容，不能证明不存在未索引、已删除、
私有或仅二进制发布的实现。新增可信来源时应保留相同的 commit/许可证/哈希证据链。
