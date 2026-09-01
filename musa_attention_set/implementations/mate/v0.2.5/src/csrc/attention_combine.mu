#include <mutlass/fast_math.h>

#include "attention_combine.hpp"
#include "mate/attention/flash_mla/mpxx_mla_combine.hpp"
#include "tvm/ffi/error.h"

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                     \
  [&] {                                                                  \
    if (NUM_SPLITS <= 32) {                                              \
      constexpr static int NAME = 32;                                    \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 64) {                                       \
      constexpr static int NAME = 64;                                    \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 96) {                                       \
      constexpr static int NAME = 96;                                    \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 128) {                                      \
      constexpr static int NAME = 128;                                   \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 160) {                                      \
      constexpr static int NAME = 160;                                   \
      return __VA_ARGS__();                                              \
    } else {                                                             \
      TVM_FFI_THROW(ValueError) << "Invalid num_splits: " << NUM_SPLITS; \
    }                                                                    \
  }()

template <typename T, bool VarlenQ>
void run_mla_combine_kernel(const mate::flash_mla::MlaCombineParams& params, musaStream_t stream) {
  static constexpr int HEAD_DIM_V = 512;  // Since only this head dimension is supported by Flash MLA
  MLA_NUM_SPLITS_SWITCH(params.num_mp_parts, NUM_SPLITS, [&] {
    constexpr int    BLOCK_SIZE_M = 8;
    constexpr int    NUM_THREADS  = BLOCK_SIZE_M * 32;
    constexpr size_t smem_size    = BLOCK_SIZE_M * (NUM_SPLITS + 1) * sizeof(float);

    dim3 grid;
    grid.x = params.batch_size;
    grid.y = mutlass::ceil_div(params.h_k * params.max_q_seq_per_hk, BLOCK_SIZE_M);
    grid.z = 1;
    mate::flash_mla::mpxx_mla_combine_kernel<T, HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS, NUM_THREADS, VarlenQ>
        <<<grid, NUM_THREADS, smem_size, stream>>>(params);
  });
}

template void run_mla_combine_kernel<mutlass::half_t, false>(const mate::flash_mla::MlaCombineParams& params,
                                                             musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::half_t, true>(const mate::flash_mla::MlaCombineParams& params,
                                                            musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::bfloat16_t, false>(const mate::flash_mla::MlaCombineParams& params,
                                                                 musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::bfloat16_t, true>(const mate::flash_mla::MlaCombineParams& params,
                                                                musaStream_t                             stream);

#undef MLA_NUM_SPLITS_SWITCH
