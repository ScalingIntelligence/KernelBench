#pragma once

#include <mute/arch/simd_mp31.hpp>

#include "simd_helper.hpp"
#include "utils.hpp"

namespace mate::attention::fmha {

template <int Rows, bool HasLearnableSink, int MaxOffset = 0>
struct Softmax {
  using Element = float;
  using TensorT = decltype(make_tensor<Element>(Shape<Int<Rows>>{}));

  TensorT row_max, row_sum;

  float const sm_scale;
  float const sm_scale_log2;

  MUTLASS_DEVICE
  Softmax(float const softmax_scale_, float const softmax_scale_log2_)
      : sm_scale(softmax_scale_), sm_scale_log2(softmax_scale_log2_) {};

  // TODO: more simd
  template <bool IsFirst, bool CheckInf, class AccQK, class TiledMmaQK>
  MUTLASS_DEVICE auto online_softmax(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk) {
    auto          reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank            = decltype(rank(reduction_target_qk))::value;

    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));

    static_assert(size<0>(acc_qk_mn) % 2 == 0, "M must be a multiple of 2");
    static_assert(size<1>(acc_qk_mn) % 4 == 0, "N must be a multiple of 4");

    static constexpr int BurstGranularityM = size<0>(acc_qk_mn) % 4 == 0 ? 4 : 2;
    static constexpr int BurstGranularityN = 4;

    using MVecType = vector_type<Element, BurstGranularityM>;

    TensorT correction_scales;
    Tensor  row_max_prev = make_fragment_like(row_max);

    if constexpr (IsFirst) {
      mute::fill(correction_scales, Element{1.f});
    } else {
      mute::copy(row_max, row_max_prev);
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      // RowMax
      float4 row_max_cur;

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        float4 vec_acc =
            make_float4(acc_qk_mn(i, j + 0), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));

        if (j == 0) {
          row_max_cur = vec_acc;
        } else {
          mute::max(row_max_cur, vec_acc, row_max_cur);
        }
      }
      float2 v2_row_max_cur;
      mute::max(v2_row_max_cur, make_float2(row_max_cur.x, row_max_cur.y), make_float2(row_max_cur.z, row_max_cur.w));
      row_max(i) = max(v2_row_max_cur.x, v2_row_max_cur.y);

      for_each(make_seq<red_rank>{}, [&](auto r) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
          row_max(i) = max(row_max(i), __shfl_xor_sync(uint32_t(-1), row_max(i), stride<r>(reduction_target_qk) * j));
        }
      });

      if (!IsFirst) {
        row_max(i) = max(row_max_prev(i), row_max(i));
      }

      bool const is_all_masked = row_max(i) == (-std::numeric_limits<Element>::infinity());
      Element    max_scaled    = row_max(i);
      if constexpr (CheckInf) {
        max_scaled = is_all_masked ? Element{0} : row_max(i);
      }

      // if (i == 0 && threadIdx.x == 128) {
      //   printf("first:%d acc:%f %f %f %f\n", IsFirst, acc_qk_mn(i, 0), acc_qk_mn(i, 1), acc_qk_mn(i, 2), acc_qk_mn(i,
      //   3));
      // }

      // Exp
      Element scale_max = max_scaled * sm_scale_log2;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        acc_qk_mn(i, j + 0) = acc_qk_mn(i, j + 0) * sm_scale_log2 - scale_max + MaxOffset;
        acc_qk_mn(i, j + 1) = acc_qk_mn(i, j + 1) * sm_scale_log2 - scale_max + MaxOffset;
        acc_qk_mn(i, j + 2) = acc_qk_mn(i, j + 2) * sm_scale_log2 - scale_max + MaxOffset;
        acc_qk_mn(i, j + 3) = acc_qk_mn(i, j + 3) * sm_scale_log2 - scale_max + MaxOffset;

        float4 v4f32_src =
            make_float4(acc_qk_mn(i, j + 0), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));

#if defined(MATE_FMHA_USE_SCALAR_EXP2)
        v4f32_src.x = exp2f(v4f32_src.x);
        v4f32_src.y = exp2f(v4f32_src.y);
        v4f32_src.z = exp2f(v4f32_src.z);
        v4f32_src.w = exp2f(v4f32_src.w);
#else
        mute::fast_exp2(v4f32_src, v4f32_src);
#endif

        acc_qk_mn(i, j + 0) = v4f32_src.x;
        acc_qk_mn(i, j + 1) = v4f32_src.y;
        acc_qk_mn(i, j + 2) = v4f32_src.z;
        acc_qk_mn(i, j + 3) = v4f32_src.w;
      }

      // RowSum
      if constexpr (!IsFirst) {
        correction_scales(i) = is_all_masked ? Element{1.f} : exp2f((row_max_prev(i) - row_max(i)) * sm_scale_log2);
        row_sum(i)           = correction_scales(i) * row_sum(i);
      }

      float4 row_sum_cur;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        float4 vec_acc =
            make_float4(acc_qk_mn(i, j + 0), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));
        if (j == 0) {
          row_sum_cur = vec_acc;
        } else {
          mute::add(row_sum_cur, vec_acc, row_sum_cur);
        }
      }
      float2 v2_row_sum_cur;
      mute::add(v2_row_sum_cur, make_float2(row_sum_cur.x, row_sum_cur.y), make_float2(row_sum_cur.z, row_sum_cur.w));
      Element row_sum_block = v2_row_sum_cur.x + v2_row_sum_cur.y;
      row_sum(i)            = IsFirst ? row_sum_block : (row_sum(i) + row_sum_block);
    }

    return correction_scales;
  }

  template <class AccPV, class TiledMmaPV>
  MUTLASS_DEVICE void rescale_o(AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv, TensorT const& correction_scales) {
    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));
    static_assert(size<0>(acc_pv_mn) == Rows);
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_pv_mn); ++i) {
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j += 4) {
        acc_pv_mn(i, j + 0) *= correction_scales(i);
        acc_pv_mn(i, j + 1) *= correction_scales(i);
        acc_pv_mn(i, j + 2) *= correction_scales(i);
        acc_pv_mn(i, j + 3) *= correction_scales(i);
      }
    }
  }

  template <class AccPV, class TiledMmaPV, class SinkVal>
  MUTLASS_DEVICE auto tail(AccPV&            acc_pv,
                           TiledMmaPV const& tiled_mma_pv,
                           SinkVal const&    sink_vals,
                           float             final_scale = 1.f) {
    static_assert(size(SinkVal{}) == Rows);

    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    // update sum across threads
    auto reduction_target  = reduction_target_n(tiled_mma_pv);
    int constexpr red_rank = decltype(rank(reduction_target))::value;

    for_each(make_seq<red_rank>{}, [&](auto r) {
      MUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target); j *= 2) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_pv_mn); i++) {
          row_sum(i) = row_sum(i) + __shfl_xor_sync(uint32_t(-1), row_sum(i), stride<r>(reduction_target) * j);
        }
      }
    });

    Tensor lse = make_fragment_like(row_sum);

    // if (threadIdx.x == 128) {
    //   printf("row_sum:%f\n", row_sum(0));
    // }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_pv_mn); ++i) {
      Element sum = row_sum(i);

      if constexpr (HasLearnableSink) {
        sum += exp2f(sink_vals(i) * M_LOG2Ef32 - row_max(i) * sm_scale_log2 + MaxOffset);
      }

      Element inv_sum      = (sum == 0.f || sum != sum) ? 0.f : __frcp_rn(sum);
      Element output_scale = inv_sum * final_scale;

      Element sum_lse = sum;
      if constexpr (MaxOffset > 0) {
        sum_lse *= 1.f / float(1 << MaxOffset);
      }
      lse(i) = (sum == 0.f || sum != sum) ? -std::numeric_limits<float>::infinity()
                                          : row_max(i) * sm_scale + __logf(sum_lse);

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j += 4) {
        acc_pv_mn(i, j + 0) *= output_scale;
        acc_pv_mn(i, j + 1) *= output_scale;
        acc_pv_mn(i, j + 2) *= output_scale;
        acc_pv_mn(i, j + 3) *= output_scale;
      }
    }
    return lse;
  }
};

}  // namespace mate::attention::fmha
