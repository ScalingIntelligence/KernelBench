## Technical Report: CUDA Kernel Optimization for Square Matrix Multiplication

### Introduction

This report details the optimization efforts for square matrix multiplication within a PyTorch architecture. The primary goal was to replace the standard `torch.matmul` operation with a custom CUDA kernel to improve performance while maintaining correctness. The target backend is CUDA, utilizing FP32 precision. Performance is measured by runtime in milliseconds, with lower values indicating better optimization.

### Preprocessing

The primary "preprocessing" step involved dynamically loading custom CUDA C++ code into the PyTorch environment. This was achieved using `torch.utils.cpp_extension.load_inline()`, which compiles and links the CUDA kernel and its C++ wrapper at runtime, making it callable from Python. The C++ wrapper also includes input validation to ensure tensors are on the CUDA device, are of `float32` type, are 2D, and have compatible square dimensions.

### Modeling Methods

The core optimization strategy focused on implementing a tiled matrix multiplication CUDA kernel.

#### 1. Initial Tiled CUDA Kernel Implementation

The `torch.matmul` operation was identified as a performance bottleneck. A custom `matmul_kernel` was developed to replace it, leveraging shared memory for efficient data reuse.

- **Tiled Matrix Multiplication:** The output matrix `C` is divided into tiles, with each thread block responsible for computing one tile.
- **Shared Memory Usage:** Within each block, sub-matrices (tiles) of input matrices `A` and `B` are cooperatively loaded into `__shared__` memory (`s_A` and `s_B`). This reduces global memory accesses.
- **Tile Size:** A `TILE_SIZE` of 32 was defined, leading to `32x32` shared memory tiles and `32x32` thread blocks. This configuration uses 8192 bytes of shared memory per block (4096 bytes for `s_A`, 4096 bytes for `s_B`), well within typical GPU limits.
- **Synchronization:** `__syncthreads()` was used to ensure all threads within a block completed loading data into shared memory before computation and finished computation before loading the next tiles.
- **Boundary Handling:** Zero-padding was applied during shared memory loading to handle cases where matrix dimensions `N` are not exact multiples of `TILE_SIZE`.
- **PyTorch Integration:** A `ModelNew` class was created, inheriting from `nn.Module`, which encapsulates the compiled `matmul_cuda` function from the extension.

#### 2. Loop Unrolling Optimization

An attempt was made to further optimize the initial kernel by applying `#pragma unroll` to the inner loop responsible for accumulating `Cvalue`. This directive aimed to instruct the CUDA compiler to unroll the loop, potentially exposing more instruction-level parallelism and reducing loop overhead for the dot product calculation within each shared memory tile.

### Results Discussion

#### Initial Kernel Performance

The initial custom CUDA kernel for tiled matrix multiplication was successfully implemented, compiled, and passed correctness checks. Its measured runtime was **37.700 ms**. This established a baseline performance for the custom kernel.

#### Loop Unrolling Impact

The application of `#pragma unroll` to the inner computation loop did not yield any observable performance improvement. The runtime remained **37.700 ms**. This suggests that either the compiler already effectively unrolled this small loop, or the performance bottleneck lies elsewhere (e.g., memory access patterns, global memory latency, or overall kernel launch overhead) rather than in the loop's instruction-level parallelism.

### Future Work

Given the current performance and the limited impact of loop unrolling, future optimization efforts could explore:

- **Tile Size Tuning:** Experimenting with different `TILE_SIZE` values to find an optimal balance between shared memory usage, thread block occupancy, and global memory access patterns.
- **Shared Memory Access Patterns:** Optimizing shared memory access to minimize bank conflicts, which can degrade performance.
- **Register Blocking:** Implementing register blocking to further reduce shared memory accesses and increase data reuse within registers.
- **Warp-Level Optimizations:** Exploring more advanced tiling strategies, such as warp-level tiling, to improve utilization.
- **Asynchronous Memory Operations:** Investigating the use of asynchronous memory copy operations (e.g., `cuda::memcpy_async`) to overlap computation with data transfer.
- **Non-Square Matrix Support:** Extending the kernel to efficiently handle non-square matrix multiplications.