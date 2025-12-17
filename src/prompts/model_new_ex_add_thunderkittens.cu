#include "kittens.cuh"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace kittens;

constexpr int BLOCK_SIZE = 16;
#define NUM_WORKERS (1)
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

struct add_globals {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using tile_gl = gl<bf16, 1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B;
    tile_gl C;
};

__global__ void add_tk(const __grid_constant__ add_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Cs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> C_reg;
    
    int col = blockIdx.x;
    int row = blockIdx.y;
    
    // Load A and B tiles from global to shared
    kittens::warp::load(As, g.A, {0, 0, row, col});
    kittens::warp::load(Bs, g.B, {0, 0, row, col});
    __syncthreads();
    
    // Load from shared to register
    kittens::warp::load(A_reg, As);
    kittens::warp::load(B_reg, Bs);
    __syncthreads();
    
    // Element-wise add: C = A + B
    kittens::warp::add(C_reg, A_reg, B_reg);
    __syncthreads();
    
    // Store result back to global
    kittens::warp::store(g.C, C_reg, {0, 0, row, col});
}

void dispatch_add(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N) {
    using tile_gl = add_globals::tile_gl;
    tile_gl a_arg{(bf16*)A.data_ptr(), nullptr, nullptr, (size_t)M, (size_t)N};
    tile_gl b_arg{(bf16*)B.data_ptr(), nullptr, nullptr, (size_t)M, (size_t)N};
    tile_gl c_arg{(bf16*)C.data_ptr(), nullptr, nullptr, (size_t)M, (size_t)N};
    add_globals g{a_arg, b_arg, c_arg};
    
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 50480;
    cudaFuncSetAttribute(add_tk, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    add_tk<<<blocks, NUM_THREADS, mem_size>>>(g);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernels, m) {
    m.doc() = "ThunderKittens element-wise add kernel";
    m.def("dispatch_add", &dispatch_add, "Element-wise add using ThunderKittens");
}
