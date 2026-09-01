# MooreThreads MUTLASS Changelog

## [0.3.0](https://github.com/MooreThreads/mutlass/releases/tag/v0.3.0) (2025-12-19)

- New Features:
  - Tensor Memory Engine (TME) im2col primitives.
  - New warp specialized GEMM mainloop targeting MP31 architecture.
  - New instances of FP8 (e4m3, m5m2) GEMM in Library targeting MP31 architecture.
  - New persistent tile schedule.
  - New Warp specialized [FMHA](./experimental/fmha/fmha_fwd.mu) and [Paged FMHA](./experimental/fmha/paged_fmha_fwd.mu) implementation for MP31 architecture.
  - New Warp specialized [MLA](./experimental/fmha/mla.mu) implementation for MP31 architecture.
- Bug fixing and improvements
  - Refine FP8 scale GEMM implementation for MP31 architecture.


## [0.2.0](https://github.com/MooreThreads/mutlass/releases/tag/v0.2.0) (2025-02-26)

- MP31 Features:
  - Squad-level MMA(SQMMA) and Warp-level MMA primitives with rich data types (TF32/FP16/BF16/[FP8](./examples/02_mp31_fp8_gemm_with_collective_builder)/S8 etc.).
  - Tensor Memory Engine(TME) and [RobustBufferAccess](./test/unit/mute/mp31/mp31_robust_buffer_access.mu) primitives.
- New GEMM mainloop and epilogue targeting MP31 architecture that achieve high performance with TME and SQMMA.
- New tile scheduler to support CTA swizzle for MP31 kernels.
- New experimental directory housing the implementations that are not yet stable and may have significant changes in the future.
  - [Prototype of Flash Attention Forward](./experimental/mp31_flash_attention_fwd/) targeting MP31 architecture with TME, RobustBufferAccess and SQMMA.
- New [FP8 GEMM with groupwise scaling](./examples/03_mp31_fp8_scaling_gemm/).
- Upgrade the backend from CUTLASS/CuTe 3.5.0 to CUTLASS/CuTe 3.6.0.


## [0.1.1](https://github.com/MooreThreads/mutlass/releases/tag/v0.1.1) (2024-09-30)

- [MuTe](./include/mute), a core library and backend adapted from CUTLASS CuTe
- Quyuan Features
  - MMA primitives: TensorFloat32, BFloat16, Float16, INT8
- FMA/MMA GEMM Kernels targeting the Quyuan architecture
  - Note: this is a beta release. Further updates to MUTLASS will include performance improvements, feature enablement, and possible breaking changes to the API
- MUTLASS Profiler, Library, and Utilities
- Two examples that demonstrate the usage of the [low-level API](./examples/00_basic_gemm) and the [collective builders](./examples/01_quyuan_gemm_with_collective_builder) to build GEMM kernelS
