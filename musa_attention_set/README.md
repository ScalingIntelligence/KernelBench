# MUSA Attention Set

本目录归档截至 **2026-09-01** 能从公开一手信源核验到的 MUSA Attention
算子实现、框架接入和有版本意义的历史快照。目标是保留源码、固定版本和
许可证证据，而不是把“公开了代码”误写成“已经在本仓库硬件上证明更快”。

## 当前收录

| 来源 | 收录版本 | 主要内容 | 定位 |
| --- | --- | --- | --- |
| MATE | v0.2.5、v0.2.6 | FA3、FMHA、Paged KV、FlashMLA、稀疏 MLA、KDA、SageAttention、MSA | 当前 S5000/MP31 主力库 |
| MUTLASS | v0.2.0、MATE 固定版本 | 早期 MP31 FA forward；当前 FMHA、Paged FMHA、MLA | MATE/MT-flashMLA 依赖 |
| MT-flashMLA | 固定 commit | BF16/FP16 MLA decode、Paged KV | 早期 MP31 独立实现 |
| TorchMUSA | v2.9.1、2.11.0.post1 | PyTorch SDPA 与 muDNN FlashAttention 接入 | 框架后端 |
| TileLang-MUSA | 旧仓库最终快照、新维护仓库快照 | FA 前后向、线性/块稀疏/NSA/MLA 等 DSL 实现 | S4000 覆盖与新 MP31 后端并存 |
| vLLM-MUSA | v0.28.0-dev 固定 commit | DeepSeek-V4 稀疏 FlashMLA/index/cache、GLM-5.2 indexer | 推理专用新内核与调度 |
| Paddle-MUSA | v3.5.0.r1 | FA 前后向注册、MATE KV-cache 桥接 | Paddle S5000 后端 |
| llama.cpp | 固定 commit | 由 MUSA 后端编译的 ggml FlashAttention 模板 | mp_21/mp_22/mp_31 可移植路径 |

完整字段见 [manifest.csv](manifest.csv)，检索与取舍依据见
[research_notes.md](research_notes.md)，复用同一内核的框架进展见
[integrations.md](integrations.md)。

## 目录约定

每个版本目录都包含：

- `metadata.json`：上游仓库、完整 commit、标签、硬件/软件条件和收录范围；
- `LICENSE.upstream`：该快照的上游许可证原文；
- `src/`：完整或明确标注为 selective 的源码快照；
- `SHA256SUMS`：`src/` 与许可证的逐文件 SHA-256。

完整快照仍可能依赖 MUSA SDK、闭源 muDNN、PyTorch/Paddle 主仓库或单独归档的
MUTLASS。选择性快照用于研究、移植和版本比较，不保证脱离上游仓库即可独立构建。

## 状态含义

- `provenance-verified-not-built`：已从上游官方仓库固定 commit 并核对许可证，尚未在本地 MUSA 环境编译。
- `upstream-claimed-tested-devices-not-locally-built`：上游明确列出测试设备，但本仓库没有重复该实验。
- `source-archived-selective`：只归档 Attention 相关路径，非完整项目镜像。

本次导入没有新增任何 `benchmarked` 或 `optimized` 状态。性能结论必须补充设备、
MUSA SDK、TorchMUSA、dtype、shape、基线、预热和重复次数后才能升级。

## 完整性检查

```bash
python musa_attention_set/scripts/verify_snapshots.py
```

更新上游快照时，应新建版本目录而不是覆盖旧目录，然后更新 `manifest.csv`、
`metadata.json` 并执行：

```bash
python musa_attention_set/scripts/verify_snapshots.py --write
python musa_attention_set/scripts/verify_snapshots.py
```

## 与 KernelBench 的关系

这些上游实现覆盖的 API、dtype、布局和硬件代际并不相同，不能直接等价为
`KernelBench/level1/97_ScaledDotProductAttention.py` 的可替换答案。后续适配应按
题目/变体在独立分支中增加调用包装、正确性测试和基准数据，不修改本目录中的
原始快照。
