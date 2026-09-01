/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <mutlass/numeric_conversion.h>

#include "fwd_params.hpp"
#include "online_softmax.hpp"


namespace mutlass {

template <>
struct NumericArrayConverter<mutlass::half_t, float, 4, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<mutlass::half_t, 4>;
  using source_type = Array<float, 4>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    using v4f16_t = _Float16 __attribute__((vector_size(8)));
    using v4f32_t = float __attribute__((vector_size(16)));

    Array<mutlass::half_t, 4> result;

    reinterpret_cast<v4f16_t&>(result) = __builtin_convertvector(*reinterpret_cast<v4f32_t const*>(source.data()), v4f16_t);
    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericArrayConverter<mutlass::half_t, float, 8, FloatRoundStyle::round_to_nearest> {
  using result_type = Array<mutlass::half_t, 8>;
  using source_type = Array<float, 8>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    Array<mutlass::half_t, 8> result;
    NumericArrayConverter<mutlass::half_t, float, 4, round_style> convert_vector_;

    Array<mutlass::half_t, 4> *result_ptr = reinterpret_cast<Array<mutlass::half_t, 4>*>(&result);
    Array<float, 4> const *source_ptr  = reinterpret_cast<Array<float, 4> const*>(&source);

    result_ptr[0] = convert_vector_(source_ptr[0]);
    result_ptr[1] = convert_vector_(source_ptr[1]);

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

} // namespace mutlass

template <
  class Element,
  class GmemLayoutQ,
  class TmeLoadQ,
  class GmemLayoutKey,
  class KeyDesc,
  class GmemLayoutV,
  class TmeLoadV,
  class GmemLayoutO,
  class GmemLayoutS,
  bool  EnableSoftmax,
  bool  EnableCausal,
  class KernelTraits
>
__global__ __launch_bounds__(KernelTraits::NumThreads, 1)
void flash_atten_fwd(GmemLayoutQ    gmem_layout_Q,
                            TmeLoadQ       tme_load_Q,
                            GmemLayoutKey  gmem_layout_Key,
                            KeyDesc        key_desc,
                            Element const* ptr_Key,
                            GmemLayoutV    gmem_layout_V,
                            TmeLoadV       tme_load_V,
                            GmemLayoutO    gmem_layout_O,
                            Element*       ptr_O,
                            GmemLayoutS    gmem_layout_S,
                            Element*       ptr_S,
                            int n_tiles,
                            float rln2_scale) {

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310))

  __shared__ char __attribute__((aligned(256))) smem_buf[sizeof(typename KernelTraits::SharedStorage)];
  auto& shared_storage = *reinterpret_cast<typename KernelTraits::SharedStorage*>(smem_buf);

  int h_coord_q = blockIdx.y;
  int h_coord_kv = h_coord_q % size<2>(gmem_layout_V);
  int b_coord = blockIdx.z;

  const int warp_idx = mutlass::canonical_warp_idx();

  using TileShapeQK = typename KernelTraits::TileShapeQK;
  using TileShapePV = typename KernelTraits::TileShapePV;
  constexpr int FragmentSize = KernelTraits::FragmentSize;
  using PermuteAlignmentType = mute::uint_bit_t<sizeof_bits_v<Element> * FragmentSize>;
  using ElementAccum = typename KernelTraits::ElementAccum;

  auto tile_shape_Q = select<0, 2>(TileShapeQK{});

  auto tile_shape_Key = typename KernelTraits::KeyPermuteTile{};

  auto tile_shape_V = select<2, 1>(TileShapePV{});
  auto tile_shape_O = Shape<Int<get<0>(TileShapeQK{})>, Int<get<1>(TileShapePV{})>>{};
  auto tile_shape_S = select<0, 1>(TileShapeQK{});

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()),   typename KernelTraits::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()),   typename KernelTraits::SmemLayoutK{});
  Tensor sLse = make_tensor(make_smem_ptr(shared_storage.smem_lse.data()),   typename KernelTraits::SmemLayoutLse{});
  Tensor sAlpha = make_tensor(make_smem_ptr(shared_storage.smem_alpha.data()),   typename KernelTraits::SmemLayoutAlpha{});
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem_s.data()),   typename KernelTraits::SmemLayoutS{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()),   typename KernelTraits::SmemLayoutV{});
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), typename KernelTraits::SmemLayoutVt{});

  typename KernelTraits::TiledMmaQK tiled_mma0;
  typename KernelTraits::TiledMmaPV tiled_mma1;
  typename KernelTraits::TiledMmaQKPermute tiled_mma0_perm;

  auto thr_mma0 = tiled_mma0.get_thread_slice(threadIdx.x % KernelTraits::PNumThreads);
  auto thr_mma1 = tiled_mma1.get_thread_slice(threadIdx.x % KernelTraits::PNumThreads);
  auto thr_mma0_perm = tiled_mma0_perm.get_thread_slice(threadIdx.x % KernelTraits::PNumThreads);

  constexpr int Rows = size<0>(typename KernelTraits::TileShapeQK{}) / (KernelTraits::PNumThreads / size(typename KernelTraits::TiledMmaQK::ThrID{})) / 16;

  Tensor alpha       = make_tensor<ElementAccum>(Shape<Int<Rows>>{});
  Tensor lse         = make_fragment_like(alpha);
  Tensor row_sum     = make_fragment_like(alpha);
  Tensor row_max     = make_fragment_like(alpha);
  Tensor old_row_max = make_fragment_like(alpha);

  if constexpr (EnableSoftmax) {
    fill(lse, ElementAccum(0.0f));
    fill(alpha, ElementAccum(0.0f));
    fill(old_row_max, -mutlass::platform::numeric_limits<ElementAccum>::infinity());
  } else {
    fill(lse, ElementAccum(1.0f));
  }


  constexpr int num_warps_QK = KernelTraits::CNumThreads/ mutlass::NumThreadsPerWarp;
  constexpr int num_warps_PV = KernelTraits::PNumThreads/ mutlass::NumThreadsPerWarp;
  bool is_pv_ws = warp_idx < num_warps_PV;
  bool is_qk_ws = !is_pv_ws;

  using BarrierStorage = typename KernelTraits::BarrierStorage;
  mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
  BarrierStorage* barrier_ids = reinterpret_cast<BarrierStorage*>(0);

  typename KernelTraits::PipeLineQ q_pipeline(reinterpret_cast<uint64_t>(&barrier_ids->pipeline_Q));
  typename KernelTraits::PipeLineAlpha alpha_pipeline(reinterpret_cast<uint64_t>(&barrier_ids->pipeline_Alpha));
  uint32_t alpha_phase = 0;
  if (warp_idx == 0) {
    q_pipeline.init(1 /*warp num*/);
    alpha_pipeline.init(num_warps_PV /*warp num*/);
  }

  // TME LOAD V
  typename KernelTraits::PipeLineV::Params v_pipe_params;
  v_pipe_params.transaction_bytes = KernelTraits::TmeTransactionBytesV;
  v_pipe_params.num_consumers = num_warps_PV;
  v_pipe_params.num_producers = 1;
  mutlass::Mp31PipelineTmeAsync<KernelTraits::VStages> v_pipeline(v_pipe_params, reinterpret_cast<uint64_t>(&barrier_ids->pipeline_V));

  // ldgsts K
  typename KernelTraits::PipeLineK::Params k_pipe_params;
  k_pipe_params.producer_arv_count = num_warps_PV;
  k_pipe_params.consumer_arv_count = num_warps_QK;
  mutlass::Mp31PipelineAsync<KernelTraits::KStages> k_pipeline(k_pipe_params, reinterpret_cast<uint64_t>(&barrier_ids->pipeline_K));

  // TCE
  typename KernelTraits::PipeLineGemm::Params gemm_pipe_params;
  gemm_pipe_params.producer_arv_count = num_warps_PV;
  gemm_pipe_params.consumer_arv_count = num_warps_QK;
  mutlass::Mp31PipelineAsync<1> gemm_pipeline(gemm_pipe_params, reinterpret_cast<uint64_t>(&barrier_ids->pipeline_Gemm));

  mutlass::PipelineState<KernelTraits::KStages> k_load = mutlass::make_producer_start_state<mutlass::PipelineState<KernelTraits::KStages>>();
  mutlass::PipelineState<KernelTraits::KStages> k_read;
  mutlass::PipelineState<KernelTraits::KStages> alpha_write;
  mutlass::PipelineState<KernelTraits::KStages> alpha_read;
  mutlass::PipelineState<KernelTraits::KStages> lse_alpha_load = mutlass::make_producer_start_state<mutlass::PipelineState<KernelTraits::KStages>>();
  mutlass::PipelineState<KernelTraits::KStages> lse_alpha_read;
  mutlass::PipelineState<KernelTraits::VStages> v_load = mutlass::make_producer_start_state<mutlass::PipelineState<KernelTraits::VStages>>();
  mutlass::PipelineState<KernelTraits::VStages> v_read;
  mutlass::PipelineState<1> QK = mutlass::make_producer_start_state<mutlass::PipelineState<1>>();
  mutlass::PipelineState<1> PV;

  __syncthreads();

  // TME load Q
  Tensor mQ = tme_load_Q.get_tme_tensor(shape(gmem_layout_Q));
  auto cta_tme_q = tme_load_Q.get_slice(0);
  Tensor gQ = local_tile(mQ, tile_shape_Q, make_coord(blockIdx.x, 0, h_coord_q, b_coord));

  Tensor tQgQ = cta_tme_q.partition_S(gQ);
  Tensor tQsQ = cta_tme_q.partition_D(sQ);

  // TME load V
  Tensor mV = tme_load_V.get_tme_tensor(shape(gmem_layout_V));
  auto cta_tme_v = tme_load_V.get_slice(0);
  Tensor gV = local_tile(mV, tile_shape_V, make_coord(_, 0, h_coord_kv, b_coord));

  int tile_iter_v = 0;
  Tensor tVgV = cta_tme_v.partition_S(gV);
  Tensor tVsV = cta_tme_v.partition_D(sV);

  // LDGSTS Load Key
  Tensor mK = make_tensor(make_gmem_ptr(ptr_Key), gmem_layout_Key);
  Tensor gK = local_tile(mK, tile_shape_Key, make_coord(_, 0, h_coord_kv, b_coord));

  TiledCopy tiled_copy_key = typename KernelTraits::GmemTiledCopyK{};
  ThrCopy thr_copy_key = tiled_copy_key.get_thread_slice(threadIdx.x % KernelTraits::PNumThreads);

  Tensor tKgK = thr_copy_key.partition_S(gK);
  Tensor tKsK = thr_copy_key.partition_D(sK);

  int tile_iter_k = 0;

  // MMA0
  Tensor tCsQ = thr_mma0.partition_A(sQ);
  Tensor tCrQ = thr_mma0.make_fragment_A(tCsQ);

  Tensor tCsK = thr_mma0.partition_B(sK);
  Tensor tCrK = thr_mma0.make_fragment_B(tCsK);

  // MMA1
  Tensor tCsS = thr_mma1.partition_A(sS);
  Tensor tCrS = thr_mma1.make_fragment_A(tCsS);

  Tensor tCsV = thr_mma1.partition_B(sVt);
  Tensor tCrV = thr_mma1.make_fragment_B(tCsV);
 
  auto tiled_copy_r2s = make_tiled_copy_C(Copy_Atom<UniversalCopy<PermuteAlignmentType>, Element>{}, tiled_mma0_perm);
  auto thr_copy_r2s = tiled_copy_r2s.get_thread_slice(threadIdx.x % KernelTraits::PNumThreads);
  Tensor tSsS = thr_copy_r2s.partition_D(sS);

  auto tiled_copy_lse_r2s = make_tiled_copy_C_atom(
    Copy_Atom<UniversalCopy<uint_bit_t<sizeof_bits_v<ElementAccum>>>, ElementAccum>{},
    tiled_mma0
  );
  auto thr_copy_lse_r2s = tiled_copy_lse_r2s.get_thread_slice(threadIdx.x % KernelTraits::PNumThreads);
  Tensor tSsL = thr_copy_lse_r2s.partition_D(sLse);
  Tensor tSsA = thr_copy_lse_r2s.partition_D(sAlpha);

  int inner_loop_cnt = n_tiles;
  if constexpr (EnableCausal) {
      inner_loop_cnt = min(size<0>(gmem_layout_V) + size<0>(tile_shape_V) - 1,
                            (blockIdx.x + 1) * size<0>(tile_shape_Q) + size<0>(tile_shape_V) - 1) / size<0>(tile_shape_V);
  }
  if(is_pv_ws) {
    auto oob_mn = make_tuple(size<0>(gmem_layout_Q) - blockIdx.x * size<0>(tile_shape_Q), size<1>(gmem_layout_V));
    auto accum_o = partition_fragment_C(tiled_mma1, tile_shape_O);
    // clear(accum_o);
    Tensor mO = make_tensor(make_gmem_ptr(ptr_O), gmem_layout_O);
    Tensor gO = local_tile(mO, tile_shape_O, make_coord(blockIdx.x, 0, h_coord_q, b_coord));
    Tensor tOgO = thr_mma1.partition_C(gO);
    Tensor cO = make_identity_tensor(make_shape(size<0>(tile_shape_O), size<1>(tile_shape_O)));
    Tensor tCcO = thr_mma1.partition_C(cO);

    if (warp_idx == 0) {
      q_pipeline.arrive_and_expect_tx(KernelTraits::TmeTransactionBytesQ);
      uint32_t bar_id = q_pipeline.get_barrier_id();
      copy(tme_load_Q.with(bar_id), tQgQ, tQsQ);
    }

    if (warp_idx == 0) {
      v_pipeline.producer_acquire(v_load);
      uint32_t bar_id = v_pipeline.producer_get_barrier_id(v_load);
      copy(tme_load_V.with(bar_id), tVgV(_,_,_,tile_iter_v), tVsV(_,_,_,v_load.index()));
      ++tile_iter_v;
      ++v_load;
    }

    if constexpr (EnableSoftmax) {
      alpha_pipeline.wait(alpha_phase);
      alpha_phase ^= 1;
      copy(tSsA(_, _, _, alpha_read.index()), alpha);
      ++alpha_read;
      update_accum(accum_o, alpha); // update accum_o
    }

    // SV GEMM
    gemm_pipeline.consumer_wait(PV);
    v_pipeline.consumer_wait(v_read);

    mute::gemm(tiled_mma1, tCrS, tCrV(_,_,_,v_read.index()), accum_o);

    for (int loop = 1; loop < inner_loop_cnt; ++loop) {

      if constexpr (KernelTraits::VStages == 2) {
        if (warp_idx == 0) {
          v_pipeline.producer_acquire(v_load);
          uint32_t bar_id = v_pipeline.producer_get_barrier_id(v_load);
          copy(tme_load_V.with(bar_id), tVgV(_,_,_,tile_iter_v), tVsV(_,_,_,v_load.index()));
          ++tile_iter_v;
          ++v_load;
        }
      }

      warpsquad_wait();
      v_pipeline.consumer_release(v_read);
      ++v_read;
      gemm_pipeline.consumer_release(PV);
      ++PV;

      if constexpr (KernelTraits::VStages == 1) {
        if (warp_idx == 0) {
          v_pipeline.producer_acquire(v_load);
          uint32_t bar_id = v_pipeline.producer_get_barrier_id(v_load);
          copy(tme_load_V.with(bar_id), tVgV(_,_,_,tile_iter_v), tVsV(_,_,_,v_load.index()));
          ++tile_iter_v;
          ++v_load;
        }
      }

      if constexpr (EnableSoftmax) {
        alpha_pipeline.wait(alpha_phase);
        alpha_phase ^= 1;
        copy(tSsA(_, _, _, alpha_read.index()), alpha);
        ++alpha_read;
        update_accum(accum_o, alpha); // update accum_o
      }
  
      gemm_pipeline.consumer_wait(PV);
      v_pipeline.consumer_wait(v_read);

      mute::gemm(tiled_mma1, tCrS, tCrV(_,_,_,v_read.index()), accum_o);
    }
  
    copy(tSsL(_, _, _, 0), lse);

    warpsquad_wait();
    v_pipeline.consumer_release(v_read);
    ++v_read;
    gemm_pipeline.consumer_release(PV);
    ++PV;

    if constexpr (EnableSoftmax) {
      apply_softmax_normalizer(accum_o, lse); // update accum_o
    }

    // STG O
    #pragma unroll
    for(int i = 0; i < size(accum_o); ++i) {
      if (elem_less(tCcO(i), oob_mn)) {
        tOgO(i) = Element(accum_o(i));
      }
    }

  }


  if (is_qk_ws) {

    auto accum_s = partition_fragment_C(tiled_mma0, tile_shape_S);
    auto accum_softmax = partition_fragment_C(tiled_mma0, tile_shape_S);
    auto accum_cvt = make_fragment_like<Element>(accum_softmax);
    Tensor tSrS = thr_copy_r2s.retile_S(accum_cvt);
    Tensor tCvt_frg = recast<mutlass::Array<Element, FragmentSize>>(accum_cvt);
    Tensor tAcc_frg = recast<mutlass::Array<ElementAccum, FragmentSize>>(accum_softmax);
    auto oob_n = size<0>(gmem_layout_Key)-(n_tiles - 1) * size<1>(tile_shape_S);
    Tensor cS = make_identity_tensor(tile_shape_S);
    Tensor tCcS = thr_copy_r2s.partition_D(cS);

    k_pipeline.producer_acquire(k_load);
    copy(tiled_copy_key.with(key_desc), tKgK(_,_,_,tile_iter_k), tKsK(_,_,_,k_load.index()));
    ++tile_iter_k;

    ldgsts_wait();
    k_pipeline.producer_commit(k_load);
    ++k_load;

    q_pipeline.wait(/*phase*/0); // wait Q
    k_pipeline.consumer_wait(k_read);
    mute::gemm(tiled_mma0, tCrQ, tCrK(_,_,_,k_read.index()), accum_s);
    if constexpr (KernelTraits::KStages == 2) {
      if (n_tiles > 1) {
        k_pipeline.producer_acquire(k_load);
        copy(tiled_copy_key.with(key_desc), tKgK(_,_,_,tile_iter_k), tKsK(_,_,_,k_load.index()));
        ++tile_iter_k;
      }
    }


    warpsquad_wait();
    k_pipeline.consumer_release(k_read);
    ++k_read;
    if constexpr (KernelTraits::KStages == 1) {
      if (n_tiles > 1) {
        k_pipeline.producer_acquire(k_load);
        copy(tiled_copy_key.with(key_desc), tKgK(_,_,_,tile_iter_k), tKsK(_,_,_,k_load.index()));
        ++tile_iter_k;
      }
    }
    
    Tensor tCcMask = thr_mma0_perm.partition_C(cS);
    float margin = size<0>(gmem_layout_Q) / size<0>(gmem_layout_V);
    for (int loop = 1; loop < inner_loop_cnt; ++loop) {

      if constexpr (EnableCausal) { 
        if (blockIdx.x * size<0>(tile_shape_Q) / (loop * size<0>(tile_shape_V)) < margin) {
          int m_idx = blockIdx.x * size<0>(tile_shape_Q);
          int n_idx = (loop - 1) * size<0>(tile_shape_V);
          add_mask(accum_s, tCcMask, m_idx, n_idx, margin);
        }
      }

      if constexpr (EnableSoftmax) {
        online_softmax_scale(accum_s, accum_softmax, rln2_scale);
        online_softmax_row_max<ElementAccum>(accum_softmax, row_max, old_row_max, alpha);
      } else {
        online_softmax_scale(accum_s, accum_softmax, float(1.0f));
      }
      // QK GEMM
      clear(accum_s);

      ldgsts_wait();
      k_pipeline.producer_commit(k_load);
      ++k_load;

      k_pipeline.consumer_wait(k_read);

      mute::gemm(tiled_mma0, tCrQ(_,_,_), tCrK(_,_,_,k_read.index()), accum_s);
      if constexpr (KernelTraits::KStages == 2) {
        if (loop != n_tiles - 1) {
          k_pipeline.producer_acquire(k_load);
          copy(tiled_copy_key.with(key_desc), tKgK(_,_,_,tile_iter_k), tKsK(_,_,_,k_load.index()));
          ++tile_iter_k;
        }
      }

      if constexpr (EnableSoftmax) {
        online_softmax_exp2(accum_softmax);
        online_softmax_row_sum<ElementAccum>(accum_softmax, row_sum, lse, alpha);
      }

      // accum_softmax -> tsrs
      MUTE_UNROLL
      for (int i = 0; i < size(tCvt_frg); ++i) {
        tCvt_frg(i) = mutlass::NumericArrayConverter<Element, ElementAccum, FragmentSize, mutlass::FloatRoundStyle::round_to_nearest>{}(tAcc_frg(i));
      }

      // STS S
      gemm_pipeline.producer_acquire(QK);
      copy(tiled_copy_r2s, tSrS, tSsS);

      warpsquad_wait();

      k_pipeline.consumer_release(k_read);
      ++k_read;

      if constexpr (KernelTraits::KStages == 1) {
        if (loop != n_tiles - 1) {
          k_pipeline.producer_acquire(k_load);
          copy(tiled_copy_key.with(key_desc), tKgK(_,_,_,tile_iter_k), tKsK(_,_,_,k_load.index()));
          ++tile_iter_k;
        }
      }

      if constexpr (EnableSoftmax) {
        
        copy(alpha, tSsA(_, _, _, alpha_write.index()));
        ++alpha_write;
        __threadfence_block();
        alpha_pipeline.arrive();
      }
      __threadfence_block();
      gemm_pipeline.producer_commit(QK);
      ++QK;
    }

    if constexpr (EnableCausal) { 
      int m_idx = blockIdx.x * size<0>(tile_shape_Q);
      int n_idx = (inner_loop_cnt-1) * size<0>(tile_shape_V);
      add_mask(accum_s, tCcMask, m_idx, n_idx, margin);
    }

    if constexpr (EnableSoftmax) {
      online_softmax_scale_last<ElementAccum>(accum_s, accum_softmax, rln2_scale, tCcS, oob_n);
      online_softmax_row_max<ElementAccum>(accum_softmax, row_max, old_row_max, alpha);
    } else {
      online_softmax_scale(accum_s, accum_softmax, float(1.0f));
    }

    if constexpr (EnableSoftmax) {
      online_softmax_exp2(accum_softmax);
      online_softmax_row_sum<ElementAccum>(accum_softmax, row_sum, lse, alpha);
    }

    if constexpr (EnableSoftmax) {
      copy(lse, tSsL(_, _, _, 1));
      copy(alpha, tSsA(_, _, _, alpha_write.index()));
      ++alpha_write;
      __threadfence_block();
      alpha_pipeline.arrive();
    }
    MUTE_UNROLL
    for (int i = 0; i < size(tCvt_frg); ++i) {
      tCvt_frg(i) = mutlass::NumericArrayConverter<Element, ElementAccum, FragmentSize, mutlass::FloatRoundStyle::round_to_nearest>{}(tAcc_frg(i));
    }

    // STS
    gemm_pipeline.producer_acquire(QK);
    copy(tiled_copy_r2s, tSrS, tSsS);
    __threadfence_block();
    gemm_pipeline.producer_commit(QK);
    ++QK;
  }

#endif
}
