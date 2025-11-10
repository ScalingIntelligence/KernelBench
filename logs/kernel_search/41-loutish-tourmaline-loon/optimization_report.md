## Technical Report: Custom CUDA Kernel Optimization for Square Matrix Multiplication

### Introduction

This report details the empirical findings and technical decisions made during the optimization of a PyTorch architecture for square matrix multiplication. The primary goal was to replace the standard `torch.matmul` with custom CUDA kernels to improve performance while maintaining correctness. Kernels were evaluated based on compilation success, correctness across multiple random inputs, and runtime performance in milliseconds.

### Preprocessing

For all custom CUDA kernel implementations, input tensors `A` and `B` were consistently moved to the CUDA device and ensured to be contiguous in memory using `.to(torch.kCUDA).contiguous()`. This is crucial for direct memory access via `data_ptr<float>()` within the CUDA kernels. Output tensors `C` were allocated on the CUDA device with matching data types and dimensions. Input validation checks (e.g., 2D tensors, floating-point type, matching inner dimensions, square matrices) were progressively added to the C++ wrapper functions to ensure robust kernel execution.

### Modeling Methods

Several custom CUDA kernel designs were implemented and evaluated.

#### 1. Basic Element-wise Kernel (Block Size 32)

*   **Design:** The initial approach involved a basic matrix multiplication kernel where each CUDA thread was responsible for computing a single element of the output matrix `C`. This required an inner loop within each thread to perform the dot product (`sum += A[row * N + k] * B[k * N + col]`). A 2D grid and block configuration (`dim3 block_dim(32, 32)`) was used to map threads to output matrix dimensions.
*   **Implementation:** The kernel `matmul_kernel` was defined, and a C++ wrapper `matmul_cuda` handled tensor preparation and kernel launch using `torch.utils.cpp_extension.load_inline`.

#### 2. Basic Element-wise Kernel (Block Size 16)

*   **Design:** This iteration maintained the element-wise computation strategy but adjusted the thread block size to `16x16` (`dim3 block_dim(16, 16)`). This change aimed to explore the impact of block dimensions on performance. Robust input validation using `TORCH_CHECK` macros was also integrated into the C++ wrapper.
*   **Implementation:** Similar to the first attempt, but with `BLOCK_SIZE_X = 16` and `BLOCK_SIZE_Y = 16`.

#### 3. Basic Element-wise Kernel (Block Size 16, Refined)

*   **Design:** This was a minor refinement of the previous basic element-wise kernel, retaining the `16x16` block size. The core logic remained the same, focusing on correctness and establishing a baseline.
*   **Implementation:** The kernel `basic_matmul_kernel` and wrapper `basic_matmul_cuda` were compiled, identical in logic to the previous `16x16` attempt.

#### 4. Tiled Matrix Multiplication with Shared Memory

*   **Design:** To mitigate high global memory traffic in the basic kernels, a tiled matrix multiplication strategy was introduced. Each thread block was designed to cooperatively load sub-tiles of matrices `A` and `B` into shared memory (`__shared__ float As[BLOCK_SIZE][BLOCK_SIZE]`, `__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]`). Threads then performed partial computations using these shared memory tiles, iterating over the K-dimension. `__syncthreads()` calls ensured data consistency. A `BLOCK_SIZE` of 32 was used.
*   **Implementation:** The `matmul_tiled_kernel` was developed, featuring shared memory arrays and the tiling loop. Boundary checks were included during data loading and result writing.

#### 5. Tiled Matrix Multiplication with Shared Memory and Padding

*   **Design:** Building on the tiled approach, this optimization specifically addressed shared memory bank conflicts. It was identified that accessing columns of the `Bs` shared memory tile could lead to conflicts when `BLOCK_SIZE` is 32. To resolve this, the `Bs` array was padded in its column dimension (`__shared__ float Bs[BLOCK_SIZE][PADDED_BLOCK_SIZE]`, where `PADDED_BLOCK_SIZE` is `BLOCK_SIZE + 1`). This ensures that consecutive column accesses by different threads map to distinct shared memory banks. `As` was also padded for consistency.
*   **Implementation:** The `matmul_tiled_kernel` was modified to use `PADDED_BLOCK_SIZE` for shared memory declarations.

### Results Discussion

The performance of each kernel implementation was measured in milliseconds:

*   **Basic Element-wise Kernel (Block Size 32):** 66.500 ms
*   **Basic Element-wise Kernel (Block Size 16):** 76.100 ms
*   **Basic Element-wise Kernel (Block Size 16, Refined):** 76.100 ms
*   **Tiled Matrix Multiplication with Shared Memory:** 37.700 ms
*   **Tiled Matrix Multiplication with Shared Memory and Padding:** 47.000 ms

The initial basic element-wise kernels (Attempts 1-3) showed relatively high runtimes, indicating significant global memory bandwidth limitations. The `16x16` block size performed slightly worse than `32x32` in the basic kernel, which might be due to less occupancy or less efficient memory coalescing for the specific problem size.

The introduction of shared memory tiling (Attempt 4) dramatically improved performance, reducing runtime from approximately 66-76 ms to 37.7 ms. This confirms the effectiveness of shared memory for data reuse and reduction of global memory accesses.

However, the subsequent attempt to mitigate shared memory bank conflicts by padding `Bs` (Attempt 5) resulted in a performance *degradation*, increasing the runtime to 47.0 ms. This suggests that while bank conflicts might have been present, the padding itself introduced other overheads (e.g., increased shared memory usage, potentially affecting occupancy, or the specific access pattern for `Bs[k][threadIdx.x]` might not have been the primary source of severe conflicts or was already somewhat mitigated by the compiler). It's possible the padding itself caused cache misses or other inefficiencies that outweighed the benefit of conflict avoidance for this specific kernel and hardware.

The **Tiled Matrix Multiplication with Shared Memory (without explicit padding for bank conflicts)** yielded the best performance at 37.7 ms.

### Future Work

1.  **Optimized Shared Memory Access for `Bs`:** Re-evaluate the bank conflict issue for `Bs`. Instead of simple padding, consider transposing `Bs` in shared memory or using a more sophisticated access pattern to ensure coalesced reads and conflict-free access.
2.  **Register Blocking:** Implement register blocking (or "cache blocking") where small sub-tiles are loaded into registers for further reuse, reducing shared memory accesses.
3.  **Warp-level Optimizations:** Explore warp-level primitives and intrinsics (e.g., `__shfl_sync`) for more efficient data exchange within warps.
4.  **Asynchronous Memory Operations:** Utilize asynchronous memory copies (e.g., `cuda::memcpy_async`) to overlap global memory fetches with computation.
5.  **Different Block Sizes:** Systematically benchmark various `BLOCK_SIZE` values for the tiled kernel to find the optimal configuration for different matrix dimensions.
6.  **Non-Square Matrices:** Extend the kernel to handle non-square matrix multiplication and varying dimensions (M, N, K).