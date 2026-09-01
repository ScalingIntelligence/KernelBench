#pragma once

#include <mutlass/array.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>

#include <mute/tensor.hpp>

#include "mpxx_params.hpp"

namespace mate::flash_mla {

using namespace mute;

template <typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS, bool is_varlen_q>
__global__ void __launch_bounds__(NUM_THREADS, 1) mpxx_mla_combine_kernel(const MlaCombineParams params) {
  // grid_shape: [batch_size, num_q_heads*s_q / BLOCK_SIZE_M]
  // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result
  static_assert(NUM_THREADS / 32 == BLOCK_SIZE_M);  // The number of warps == block_size_m
  const int batch_idx   = blockIdx.x;
  const int m_block_idx = blockIdx.y;
  const int warp_idx    = threadIdx.x / 32;
  const int lane_idx    = threadIdx.x % 32;

  const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
  const int end_split_idx   = __ldg(params.num_splits_ptr + batch_idx + 1);
  const int my_num_splits   = end_split_idx - start_split_idx;
  // TODO: device assert under DEBUG macro
  // FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);
  if (my_num_splits == 1) {
    return;
  }

  const int max_num_q_seqs = params.max_q_seq_per_hk * params.h_k;
  int       q_seq_start    = 0;
  int       q_seq_end      = params.max_q_seq_per_hk / params.h_r;
  if constexpr (is_varlen_q) {
    q_seq_start = *(params.seqlens_q_ptr + batch_idx);
    q_seq_end   = *(params.seqlens_q_ptr + batch_idx + 1);
  }
  const int num_q_seqs           = (q_seq_end - q_seq_start) * params.h_r * params.h_k;
  const int num_cur_valid_q_seqs = min(BLOCK_SIZE_M, num_q_seqs - m_block_idx * BLOCK_SIZE_M);
  Tensor gLseAccum = make_tensor(make_gmem_ptr((float*)params.softmax_lseaccum_ptr + start_split_idx * max_num_q_seqs +
                                               m_block_idx * BLOCK_SIZE_M),
                                 Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
                                 make_stride(max_num_q_seqs, _1{}));
  // BHQ
  int glse_offset = batch_idx * max_num_q_seqs;
  if constexpr (is_varlen_q) {
    // HQ'
    glse_offset = q_seq_start * params.h_r;
  }
  Tensor gLse = make_tensor(make_gmem_ptr((float*)params.softmax_lse_ptr + glse_offset + m_block_idx * BLOCK_SIZE_M),
                            Shape<Int<BLOCK_SIZE_M>>{},
                            Stride<_1>{});

  extern __shared__ float __attribute__((aligned(256))) smem_buf[];

  Tensor sLseScale = make_tensor(
      make_smem_ptr(smem_buf), Shape<Int<BLOCK_SIZE_M>, Int<MAX_SPLITS>>{}, Stride<Int<MAX_SPLITS + 1>, _1>{}
      // +1 to avoid bank conflict
  );
  //  Read gLseAccum into sLseScale
  {
#pragma unroll 4
    for (int elem_idx = threadIdx.x; elem_idx < my_num_splits * BLOCK_SIZE_M; elem_idx += NUM_THREADS) {
      int split_idx                 = elem_idx / BLOCK_SIZE_M;
      int seq_idx                   = elem_idx % BLOCK_SIZE_M;
      sLseScale(seq_idx, split_idx) = seq_idx < num_cur_valid_q_seqs ? gLseAccum(split_idx, seq_idx) : -INFINITY;
    }
    __syncthreads();
  }

  if (warp_idx >= num_cur_valid_q_seqs) return;

  // Warp #i gathers LseAccum for seq #i
  {
    constexpr int NUM_LSE_PER_THREAD = mute::ceil_div(MAX_SPLITS, 32);
    float         local_lse[NUM_LSE_PER_THREAD];
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
      const int split_idx = i * 32 + lane_idx;
      local_lse[i]        = split_idx < my_num_splits ? sLseScale(warp_idx, split_idx) : -INFINITY;
    }

    float max_lse = -INFINITY;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) max_lse = max(max_lse, local_lse[i]);
    MUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2)
      max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

    float sum_lse = 0;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
    MUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

    float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : logf(sum_lse) + max_lse;
    if constexpr (is_varlen_q) {
      // B H Q
      // H, TOTAL_Q
      // m_block_idx --> max_q_per_hk
      int    cur_lse_idx   = q_seq_start * params.h_r + m_block_idx * BLOCK_SIZE_M + warp_idx;
      float* out_lse       = (float*)params.softmax_lse_ptr;
      int    cur_lse_q_idx = (m_block_idx * BLOCK_SIZE_M + warp_idx) / params.h_r;
      int    cur_lse_h_idx = (m_block_idx * BLOCK_SIZE_M + warp_idx) % params.h_r;

      bool is_valid_q = q_seq_start + cur_lse_q_idx < q_seq_end;
      if (is_valid_q && lane_idx == 0) {
        out_lse[cur_lse_h_idx * params.total_q + q_seq_start + cur_lse_q_idx] = global_lse;
      }
    } else {
      if (lane_idx == 0) gLse(warp_idx) = global_lse;
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
      const int split_idx = i * 32 + lane_idx;
      if (split_idx < my_num_splits) sLseScale(warp_idx, split_idx) = expf(local_lse[i] - global_lse);
    }
  }

  __syncwarp();

  // Warp #i accumulates activation for seq #i
  {
    const int64_t row_offset_oaccum =
        (int64_t)(start_split_idx * max_num_q_seqs + m_block_idx * BLOCK_SIZE_M + warp_idx) * HEAD_DIM_V;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<MAX_SPLITS>, Int<HEAD_DIM_V>>{},
                                 make_stride(max_num_q_seqs * HEAD_DIM_V, _1{}));

    static_assert(HEAD_DIM_V % 32 == 0);
    constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / 32;
    float         result[ELEMS_PER_THREAD];
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) result[i] = 0.0f;

#pragma unroll 2
    for (int split = 0; split < my_num_splits; ++split) {
      float lse_scale = sLseScale(warp_idx, split);
      if (lse_scale != 0.f) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
          result[i] += lse_scale * gOaccum(split, lane_idx + i * 32);
        }
      }
    }

    const int q_seq_idx = m_block_idx * BLOCK_SIZE_M + warp_idx;

    if constexpr (is_varlen_q) {
      if (q_seq_idx >= q_seq_end * params.h_r) return;  // TODO: if hr != 128
    }
    const int k_head_idx = q_seq_idx / params.max_q_seq_per_hk;
    int       o_offset   = batch_idx * params.o_batch_stride + k_head_idx * params.o_head_stride +
                   (q_seq_idx % params.max_q_seq_per_hk) * params.o_row_stride;
    if constexpr (is_varlen_q) {
      // TOTAL_Q, H, D
      o_offset = ((q_seq_start * params.h_r + q_seq_idx)) * params.o_row_stride;
    }

    auto o_ptr_local = reinterpret_cast<ElementT*>(params.o_ptr) + o_offset;

    Tensor gO = make_tensor(make_gmem_ptr(o_ptr_local), Shape<Int<HEAD_DIM_V>>{}, Stride<_1>{});

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) gO(lane_idx + i * 32) = (ElementT)result[i];
  }
}

}  // namespace mate::flash_mla
