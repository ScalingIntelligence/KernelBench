[中文版](./README_CN.md)

# MUTLASS 0.3.0

_MUTLASS 0.3.0 - December 2025_

MUTLASS (MUSA Templates for Linear Algebra Subroutines) is a header-only library for implementing high-performance matrix-matrix multiplication (GEMM) within MUSA(**M**eta-computing **U**nified **S**ystem **A**rchitecture). It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement muDNN.

See the [Quick Start Guide](./media/docs/quickstart.md) to get started quickly.

Note: MUTLASS uses the CuTe library, introduced in CUTLASS 3.x, as the backend, and thus is incompatible with most implementations of CUTLASS 2.x.

# What's New in MUTLASS 0.3.0

MUTLASS 0.3.0 is an update to MUTLASS adding:

- Tensor Memory Engine (TME) im2col primitives.
- New WarpSpecialized GEMM mainloop targeting MP31 architecture.
- New instances of FP8 (e4m3, m5m2) GEMM in Library targeting MP31 architecture.
- New persistent tile schedule.
- Refine FP8 scale GEMM implementation for MP31 architecture.
- New WarpSpecialized [FMHA](./experimental/fmha/fmha_fwd.mu) and [Paged FMHA](./experimental/fmha/paged_fmha_fwd.mu) implementation for MP31 architecture.
- New WarpSpecialized [MLA](./experimental/fmha/mla.mu) implementation for MP31 architecture.


Minimum requirements:

- Architecture: Quyuan

- Compiler: MCC 4.3.4

- MUSA Toolkit version: 4.3.4


**See the [CHANGELOG](./CHANGELOG.md) for a detailed listing of releases and updates.**

# Performance


<p align="center"><img src=media/images/mutlass-0.2.0-gemm-performance.png></p>

The above figure shows the relative performance of the tensorop GEMM compared with muDNN. The performance of TF32 data type will be further optimized in the next release.

# Documentation

- [Quick Start Guide](./media/docs/quickstart.md) - build and run MUTLASS

# Building MUTLASS

MUTLASS is a header-only template library and does not need to be built to be used by other projects. Client applications should target MUTLASS's `include/` directory in their include paths.

MUTLASS unit tests, examples, and utilities can be build with CMake. The minimum version of CMake is given in the [QuickStart guide](./media/docs/quickstart.md).

Create a build directory within the MUTLASS project, then run CMake. By default MUTLASS will build kernels for MUSA architecture versions 2.2 and 3.1.
