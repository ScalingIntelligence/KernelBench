[English](./README.md)

# MUTLASS 0.3.0

*MUTLASS 0.3.0 - 2025年12月*

MUTLASS (MUSA Templates for Linear Algebra Subroutines)是用于在MUSA(**M**eta-computing **U**nified **S**ystem **A**rchitecture)上实现高性能矩阵乘法运算的纯头文件库，采用了与实现muDNN类似的分层分解和数据搬运策略。

参考[快速入门指南](./media/docs/quickstart.md)来快速入门使用。

注意：MUTLASS使用了在CUTLASS 3.x引入的CuTe库做为后端，因此与大多数CUTLASS 2.x的实现并不兼容。

# MUTLASS 0.3.0新增

MUTLASS 0.3.0是MUTLASS的一次版本更新，添加了：

- TME im2col 的支持

- 适用于 MP31 架构的 warp specialized 矩阵乘法核心循环实现。

- 在 Library 中实例化了针对 MP31 架构的 FP8 (e4m3, m5m2) GEMM。

- persistent tile schedule 的支持

- 完善了适用于 MP31 架构的 FP8 scale GEMM 的实现

- 适用于 MP31 架构的 warp specialized [FMHA](./experimental/fmha/fmha_fwd.mu)(fused multi-head attention) 与 [Paged FMHA](./experimental/fmha/paged_fmha_fwd.mu) 的实现。

- 适用于 MP31 架构的 warp specialized [MLA](./experimental/fmha/mla.mu) 的实现。



最低要求：

- 架构：曲院

- 编译器：MCC 4.3.4

- MUSA工具包：4.3.4


**参考[变更日志](./CHANGELOG.md)获取更详细的发布及更新信息。**

# 性能

<p align="center"><img src=media/images/mutlass-0.2.0-gemm-performance.png></p>

上图展示了MUTLASS相对muDNN的TensorCore矩阵乘法性能。TF32数据类型的性能会在下一次发布时进一步优化。

# 文档

- [快速入门指南](./media/docs/quickstart.md) - 编译和运行MUTLASS


# 编译MUTLASS

MUTLASS是一个模板纯头文件库，因此在被其他项目使用时不需要单独编译。用户应用将MUTLASS的`include/`目录指定到项目头文件路径中即可使用。

MUTLASS的单元测试、实例和工具都使用CMake进行编译构建。编译构建所需要的最低CMake版本在[快速入门指南](./media/docs/quickstart.md)中给出。

在MUTLASS中创建一个单独的build目录，并执行CMake即可编译。默认情况下，MUTLASS会编译MUSA架构2.2和3.1的实现。
