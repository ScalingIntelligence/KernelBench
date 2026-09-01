#pragma once

#include <limits>
#include <mute/arch/simd_mp31.hpp>

#include "mate/attention/fmha/softmax.hpp"

namespace mate::attention::msa::collective {

using namespace mute;

// The M=16 SQMMA gives each consumer thread one logical attention row.
// The generic FMHA softmax assumes at least two rows per thread even though
// its actual reduction is row independent. Keep that utility unchanged and
// specialize only the online update needed by the MSA forward kernel.
template <int Rows, int MaxOffset = 8>
struct MsaFwdSoftmax : ::mate::attention::fmha::Softmax<Rows, false, MaxOffset> {
  using Base    = ::mate::attention::fmha::Softmax<Rows, false, MaxOffset>;
  using Element = typename Base::Element;
  using TensorT = typename Base::TensorT;

  static_assert(Rows > 0);

  MUTLASS_DEVICE MsaFwdSoftmax(float softmax_scale, float softmax_scale_log2)
      : Base(softmax_scale, softmax_scale_log2) {
  }

  template <bool IsFirst, bool CheckInf, class AccQK, class TiledMmaQK>
  MUTLASS_DEVICE auto online_softmax(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk) {
    auto          reduction_target_qk = ::mate::attention::fmha::reduction_target_n(tiled_mma_qk);
    constexpr int red_rank            = decltype(rank(reduction_target_qk))::value;

    Tensor acc_qk_mn =
        make_tensor(acc_qk.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    static_assert(size<0>(acc_qk_mn) == Rows);
    static_assert(size<1>(acc_qk_mn) % 4 == 0, "N must be a multiple of 4");

    TensorT correction_scales;
    Tensor  row_max_prev = make_fragment_like(this->row_max);

    if constexpr (IsFirst) {
      mute::fill(correction_scales, Element{1.f});
    } else {
      mute::copy(this->row_max, row_max_prev);
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      float4 row_max_cur;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        float4 values = make_float4(acc_qk_mn(i, j), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));
        if (j == 0) {
          row_max_cur = values;
        } else {
          mute::max(row_max_cur, values, row_max_cur);
        }
      }
      float2 row_max_pair;
      mute::max(row_max_pair, make_float2(row_max_cur.x, row_max_cur.y), make_float2(row_max_cur.z, row_max_cur.w));
      this->row_max(i) = max(row_max_pair.x, row_max_pair.y);

      for_each(make_seq<red_rank>{}, [&](auto r) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
          this->row_max(i) = max(this->row_max(i),
                                 __shfl_xor_sync(uint32_t(-1), this->row_max(i), stride<r>(reduction_target_qk) * j));
        }
      });

      if constexpr (!IsFirst) {
        this->row_max(i) = max(row_max_prev(i), this->row_max(i));
      }
      if constexpr (CheckInf) {
        this->row_max(i) =
            this->row_max(i) == -std::numeric_limits<Element>::infinity() ? Element{0} : this->row_max(i);
      }

      Element scale_max = this->row_max(i) * this->sm_scale_log2;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        float4 values = make_float4(acc_qk_mn(i, j) * this->sm_scale_log2 - scale_max + MaxOffset,
                                    acc_qk_mn(i, j + 1) * this->sm_scale_log2 - scale_max + MaxOffset,
                                    acc_qk_mn(i, j + 2) * this->sm_scale_log2 - scale_max + MaxOffset,
                                    acc_qk_mn(i, j + 3) * this->sm_scale_log2 - scale_max + MaxOffset);
        mute::fast_exp2(values, values);
        acc_qk_mn(i, j)     = values.x;
        acc_qk_mn(i, j + 1) = values.y;
        acc_qk_mn(i, j + 2) = values.z;
        acc_qk_mn(i, j + 3) = values.w;
      }

      if constexpr (!IsFirst) {
        correction_scales(i) = exp2f((row_max_prev(i) - this->row_max(i)) * this->sm_scale_log2);
        this->row_sum(i)     = correction_scales(i) * this->row_sum(i);
      }

      float4 row_sum_cur;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        float4 values = make_float4(acc_qk_mn(i, j), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));
        if (j == 0) {
          row_sum_cur = values;
        } else {
          mute::add(row_sum_cur, values, row_sum_cur);
        }
      }
      float2 row_sum_pair;
      mute::add(row_sum_pair, make_float2(row_sum_cur.x, row_sum_cur.y), make_float2(row_sum_cur.z, row_sum_cur.w));
      Element row_sum_block = row_sum_pair.x + row_sum_pair.y;
      this->row_sum(i)      = IsFirst ? row_sum_block : this->row_sum(i) + row_sum_block;
    }

    return correction_scales;
  }

  template <class AccPV, class TiledMmaPV, class SinkVal>
  MUTLASS_DEVICE auto tail(AccPV&            acc_pv,
                           TiledMmaPV const& tiled_mma_pv,
                           SinkVal const&    sink_vals,
                           float             final_scale = 1.f) {
    static_assert(size(SinkVal{}) == Rows);
    return Base::tail(acc_pv, tiled_mma_pv, sink_vals, final_scale);
  }
};

}  // namespace mate::attention::msa::collective
