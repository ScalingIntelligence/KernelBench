[English](./README.md)

# MUTLASS 0.2.0

*MUTLASS 0.2.0 - 2025年2月*

MUTLASS(MUSA Templates for Linear Algebra Subroutines)是用于在MUSA(**M**eta-computing **U**nified **S**ystem **A**rchitecture)上实现高性能矩阵乘法运算的纯头文件库，采用了与实现muDNN类似的分层分解和数据搬运策略。

参考[快速入门指南](./media/docs/quickstart.md)来快速入门使用。

注意：MUTLASS使用了在CUTLASS 3.x引入的CuTe库做为后端，因此与大多数CUTLASS 2.x的实现并不兼容。

# MUTLASS 0.2.0新增

MUTLASS 0.2.0是MUTLASS的一次版本更新，添加了：

- MP31特性：

  - 支持丰富数据类型的Squad-level MMA(SQMMA)和Warp-level MMA原语，包含TF32/FP16/BF16/[FP8](./examples/02_mp31_fp8_gemm_with_collective_builder)/S8等多种精度。

  - Tensor Memory Engine(TME)及[RobustBufferAccess](./test/unit/mute/mp31/mp31_robust_buffer_access.mu)原语。

- 新适用于MP31架构的矩阵乘法核心循环及后处理实现，基于TME和SQMMA实现高性能的矩阵乘法计算。

- 新适用于MP31架构算子的Tile调度器，用于实现更好的线程组调度。

- 新的*experimental*目录，用于存放尚未稳定或可能在未来有重大改变的代码实现。

  - 针对MP31架构的[FlashAttention前向原型](./experimental/mp31_flash_attention_fwd/)，运用了TME、RobustBufferAccess和SQMMA等新特性。

- 新的[Groupwise Scaling FP8矩阵乘法](./examples/03_mp31_fp8_scaling_gemm/)。

- 将后端库从CUTLASS/CuTe 3.5.0升级到CUTLASS/CuTe 3.6.0。


最低要求：

- 架构：曲院

- 编译器：MCC 4.0.0

- MUSA工具包：4.0.0


**参考[变更日志](./CHANGELOG.md)获取更详细的发布及更新信息。**

# 性能

# 文档

- [快速入门指南](./media/docs/quickstart.md) - 编译和运行MUTLASS


# 编译MUTLASS

MUTLASS是一个模板纯头文件库，因此在被其他项目使用时不需要单独编译。用户应用将MUTLASS的`include/`目录指定到项目头文件路径中即可使用。

MUTLASS的单元测试、实例和工具都使用CMake进行编译构建。编译构建所需要的最低CMake版本在[快速入门指南](./media/docs/quickstart.md)中给出。

在MUTLASS中创建一个单独的build目录，并执行CMake即可编译。默认情况下，MUTLASS会编译MUSA架构2.2和3.1的实现。
