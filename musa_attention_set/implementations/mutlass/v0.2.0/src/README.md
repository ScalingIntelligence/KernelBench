[中文版](./README_CN.md)

# MUTLASS 0.2.0

_MUTLASS 0.2.0 - February 2025_

MUTLASS(MUSA Templates for Linear Algebra Subroutines) is a header-only library for implementing high-performance matrix-matrix multiplication (GEMM) within MUSA(**M**eta-computing **U**nified **S**ystem **A**rchitecture). It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement muDNN.

See the [Quick Start Guide](./media/docs/quickstart.md) to get started quickly.

Note: MUTLASS uses the CuTe library, introduced in CUTLASS 3.x, as the backend, and thus is incompatible with most implementations of CUTLASS 2.x.

# What's New in MUTLASS 0.2.0

MUTLASS 0.2.0 is an update to MUTLASS adding:

- MP31 Features:
  - Squad-level MMA(SQMMA) and Warp-level MMA primitives with rich data types (TF32/FP16/BF16/[FP8](./examples/02_mp31_fp8_gemm_with_collective_builder)/S8 etc.).
  - Tensor Memory Engine(TME) and [RobustBufferAccess](./test/unit/mute/mp31/mp31_robust_buffer_access.mu) primitives.
- New GEMM mainloop and epilogue targeting MP31 architecture that achieve high performance with TME and SQMMA.
- New tile scheduler to support CTA swizzle for MP31 kernels.
- New experimental directory housing the implementations that are not yet stable and may have significant changes in the future.
  - [Prototype of Flash Attention Forward](./experimental/mp31_flash_attention_fwd/) targeting MP31 architecture with TME, RobustBufferAccess and SQMMA.
- New [FP8 GEMM with groupwise scaling](./examples/03_mp31_fp8_scaling_gemm/).
- Upgrade the backend from CUTLASS/CuTe 3.5.0 to CUTLASS/CuTe 3.6.0.


Minimum requirements:

- Architecture: Quyuan

- Compiler: MCC 4.0.0

- MUSA Toolkit version: 4.0.0


**See the [CHANGELOG](./CHANGELOG.md) for a detailed listing of releases and updates.**

# Performance


<p align="center"><img src=media/images/mutlass-0.2.0-gemm-performance.png></p>

The above figure shows the relative performance of the tensorop GEMM compared with muDNN. The performance of TF32 data type be futher optimized in the next release.

# Documentation

- [Quick Start Guide](./media/docs/quickstart.md) - build and run MUTLASS

# Building MUTLASS

MUTLASS is a header-only template library and does not need to be built to be used by other projects. Client applications should target MUTLASS's `include/` directory in their include paths.

MUTLASS unit tests, examples, and utilities can be build with CMake. The minimum version of CMake is given in the [QuickStart guide](./media/docs/quickstart.md).

Create a build directory within the MUTLASS project, then run CMake. By default MUTLASS will build kernels for MUSA architecture versions 2.2 and 3.1.
