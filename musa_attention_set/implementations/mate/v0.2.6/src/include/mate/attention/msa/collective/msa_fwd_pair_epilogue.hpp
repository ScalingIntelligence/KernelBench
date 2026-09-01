#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

#include "mate/attention/fmha/utils.hpp"

namespace mate::attention::msa::collective {

using namespace mute;

// Store a physical M16 tile as two adjacent TP8 query tokens:
//   rows  0..7  -> q_abs + 0
//   rows  8..15 -> q_abs + 1
template <class Element_>
struct MsaFwdPairEpilogue {
  using Element = Element_;

  static constexpr int HeadRatio      = 8;
  static constexpr int QueriesPerTile = 2;
  static constexpr int HeadDim        = 128;

  static constexpr bool IsSupportedElement = std::is_same_v<Element, mutlass::float_e4m3_t> ||
                                             std::is_same_v<Element, mutlass::half_t> ||
                                             std::is_same_v<Element, mutlass::bfloat16_t>;
  static_assert(IsSupportedElement, "MSA forward supports FP8 E4M3, FP16, and BF16 outputs.");

  struct Arguments {
    Element* ptr_o   = nullptr;
    float*   ptr_lse = nullptr;
  };

  struct Params {
    Element* ptr_o;
    float*   ptr_lse;
    int      total_q;
    int      num_qo_heads;
    int      head_ratio;
  };

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    return {args.ptr_o,
            args.ptr_lse,
            problem_size.total_q,
            problem_size.num_qo_heads,
            problem_size.num_qo_heads / problem_size.num_kv_heads};
  }

  template <class ProblemSize>
  static bool can_implement(ProblemSize const& problem_size, Arguments const& args) {
    if (problem_size.total_q == 0) {
      return true;
    }
    if (problem_size.num_kv_heads <= 0 || problem_size.num_qo_heads % problem_size.num_kv_heads != 0) {
      return false;
    }
    return problem_size.num_qo_heads / problem_size.num_kv_heads == HeadRatio && args.ptr_o != nullptr &&
           args.ptr_lse != nullptr;
  }

  template <class AccPV, class Lse, class TiledMmaPV, class WorkTile>
  MUTLASS_DEVICE void store(Params const&   params,
                            AccPV const&    acc_pv,
                            Lse const&      lse,
                            TiledMmaPV      tiled_mma_pv,
                            WorkTile const& work_tile,
                            int             consumer_thread_idx) const {
    // Each physical row is distributed over eight consumer threads.
    if (consumer_thread_idx >= QueriesPerTile * HeadRatio * 8) {
      return;
    }

    auto   thr_mma_pv = tiled_mma_pv.get_thread_slice(consumer_thread_idx);
    Tensor acc_pv_mn =
        make_tensor(acc_pv.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_pv, acc_pv.layout()));
    using PhysicalTileM = decltype(size<0>(tile_shape(TiledMmaPV{})));
    Tensor cOutput      = make_identity_tensor(make_shape(PhysicalTileM{}, Int<HeadDim>{}));
    Tensor tCcOutput    = thr_mma_pv.partition_C(cOutput);
    Tensor tCcOutput_mn =
        make_tensor(tCcOutput.data(), ::mate::attention::fmha::layout_acc_mn(tiled_mma_pv, tCcOutput.layout()));

    int head_q_begin = work_tile.head_kv * HeadRatio;
    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(acc_pv_mn); ++m) {
      int physical_row  = static_cast<int>(get<0>(tCcOutput_mn(m, 0)));
      int query_in_tile = physical_row / HeadRatio;
      int head_local    = physical_row - query_in_tile * HeadRatio;
      if (query_in_tile >= work_tile.q_count) {
        continue;
      }
      int q_abs  = work_tile.q_abs + query_in_tile;
      int head_q = head_q_begin + head_local;

      MUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<1>(acc_pv_mn); ++n) {
        int     col                 = static_cast<int>(get<1>(tCcOutput_mn(m, n)));
        int64_t output_offset       = (int64_t(q_abs) * params.num_qo_heads + int64_t(head_q)) * HeadDim + int64_t(col);
        params.ptr_o[output_offset] = Element(acc_pv_mn(m, n));
      }

      if (get<1>(tCcOutput_mn(m, 0)) == 0) {
        int64_t lse_offset         = int64_t(q_abs) * params.num_qo_heads + int64_t(head_q);
        params.ptr_lse[lse_offset] = lse(m);
      }
    }
  }
};

}  // namespace mate::attention::msa::collective
