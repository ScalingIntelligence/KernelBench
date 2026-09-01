#pragma once

#include "fmha_common.hpp"
#include "mute/arch/simd_mp31.hpp"

namespace mutlass::fmha::collective {


template <
  class Fusion,
  class Params,
  bool kIsVarlen = false
>
struct CollectiveSoftmax {
  Params const& params;
  Fusion const& fusion;
  MUTLASS_DEVICE
  CollectiveSoftmax(Params const& params, Fusion const& fusion) : params(params), fusion(fusion) {}

  using Element = float;

  template <class AccPV, class TiledMmaPV>
  MUTLASS_DEVICE
  auto
  init(AccPV const& acc_pv, TiledMmaPV const& tiled_mma_pv) {
    Tensor row_sum = make_fragment_like<Element>(size<0>(layout_acc_mn(tiled_mma_pv, acc_pv.layout())));
    Tensor row_max = make_fragment_like<Element>(row_sum);

    fill(row_max, Element{});
    fill(row_sum, Element{});

    return make_tuple(row_sum, row_max);
  }

  template <class AccQK, class TiledMmaQK, class State, class IndexQK, class ProblemSize>
  MUTLASS_DEVICE
  void
  step(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk, State& state, IndexQK& index_qk, ProblemSize const& problem_size) {
    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor index_qk_mn = make_tensor(index_qk.data(), layout_acc_mn(tiled_mma_qk, index_qk.layout()));

    fusion.template before_softmax<kIsVarlen>(acc_qk_mn, index_qk_mn, problem_size);

    auto& row_sum = get<0>(state);
    auto& row_max = get<1>(state);

    auto reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank = decltype(rank(reduction_target_qk))::value;


    static_assert(size<1>(acc_qk_mn) % 4 == 0, "N must be multiple of 4");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      float4 v4f32_max_current = make_float4(acc_qk_mn(i, 0),
                                      acc_qk_mn(i, 1),
                                      acc_qk_mn(i, 2),
                                      acc_qk_mn(i, 3));
      MUTLASS_PRAGMA_UNROLL
      for (int j = 4; j < size<1>(acc_qk_mn); j += 4) {
        float4 v4f32_max_next = make_float4(acc_qk_mn(i, j + 0),
                                        acc_qk_mn(i, j + 1),
                                        acc_qk_mn(i, j + 2),
                                        acc_qk_mn(i, j + 3));
        mute::max(v4f32_max_current, v4f32_max_current, v4f32_max_next);
      }
      float2 v2f32_max_0 = make_float2(v4f32_max_current.x, v4f32_max_current.y);
      float2 v2f32_max_1 = make_float2(v4f32_max_current.z, v4f32_max_current.w);

      mute::max(v2f32_max_0, v2f32_max_0, v2f32_max_1);
      row_max(i) = max(v2f32_max_0.x, v2f32_max_0.y);
    }

    for_each(make_seq<red_rank>{}, [&](auto r) {
      MUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
          row_max(i) = max(row_max(i), __shfl_xor_sync(uint32_t(-1), row_max(i), stride<r>(reduction_target_qk) * j));
        }
      }
    });

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      Element local_max = row_max(i) == (-std::numeric_limits<Element>::infinity()) ? Element{0} : row_max(i);

      Element scale_max = params.sm_scale_log2 * local_max;

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        acc_qk_mn(i, j + 0) = acc_qk_mn(i, j + 0) * params.sm_scale_log2 - scale_max;
        acc_qk_mn(i, j + 1) = acc_qk_mn(i, j + 1) * params.sm_scale_log2 - scale_max;
        acc_qk_mn(i, j + 2) = acc_qk_mn(i, j + 2) * params.sm_scale_log2 - scale_max;
        acc_qk_mn(i, j + 3) = acc_qk_mn(i, j + 3) * params.sm_scale_log2 - scale_max;

        float4 v4f32_src = make_float4(acc_qk_mn(i, j + 0),
                                       acc_qk_mn(i, j + 1),
                                       acc_qk_mn(i, j + 2),
                                       acc_qk_mn(i, j + 3));

        mute::fast_exp2(v4f32_src, v4f32_src);

        acc_qk_mn(i, j + 0) = v4f32_src.x;
        acc_qk_mn(i, j + 1) = v4f32_src.y;
        acc_qk_mn(i, j + 2) = v4f32_src.z;
        acc_qk_mn(i, j + 3) = v4f32_src.w;
      }
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      float4 v4f32_sum_current = make_float4(acc_qk_mn(i, 0),
                                      acc_qk_mn(i, 1),
                                      acc_qk_mn(i, 2),
                                      acc_qk_mn(i, 3));

      MUTLASS_PRAGMA_UNROLL
      for (int j = 4; j < size<1>(acc_qk_mn); j += 4) {
        float4 v4f32_sum_next = make_float4(acc_qk_mn(i, j + 0),
                                        acc_qk_mn(i, j + 1),
                                        acc_qk_mn(i, j + 2),
                                        acc_qk_mn(i, j + 3));
        mute::add(v4f32_sum_current, v4f32_sum_current, v4f32_sum_next);
      }

      float2 v2f32_sum_0 = make_float2(v4f32_sum_current.x, v4f32_sum_current.y);
      float2 v2f32_sum_1 = make_float2(v4f32_sum_current.z, v4f32_sum_current.w);
      mute::add(v2f32_sum_0, v2f32_sum_0, v2f32_sum_1);
      row_sum(i) = v2f32_sum_0.x + v2f32_sum_0.y;
    }
  }

  template <bool DoFusion, class AccQK, class TiledMmaQK, class State, class AccPV, class TiledMmaPV, class IndexQK, class ProblemSize>
  MUTLASS_DEVICE
  void
  step(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk, State& state, AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv, IndexQK& index_qk, ProblemSize const& problem_size) {
    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));
    Tensor index_qk_mn = make_tensor(index_qk.data(), layout_acc_mn(tiled_mma_qk, index_qk.layout()));

    if constexpr (DoFusion) {
      fusion.template before_softmax<false>(acc_qk_mn, index_qk_mn, problem_size);
    }

    auto& row_sum = get<0>(state);
    auto& row_max = get<1>(state);

    auto reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank = decltype(rank(reduction_target_qk))::value;
    static_assert(size<1>(acc_qk_mn) % 4 == 0, "N must be multiple of 4");

    Tensor row_max_prev = make_fragment_like(row_max);
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      row_max_prev(i) = row_max(i);

      float4 v4f32_max_current = make_float4(acc_qk_mn(i, 0),
                                      acc_qk_mn(i, 1),
                                      acc_qk_mn(i, 2),
                                      acc_qk_mn(i, 3));
      MUTLASS_PRAGMA_UNROLL
      for (int j = 4; j < size<1>(acc_qk_mn); j += 4) {
        float4 v4f32_max_next = make_float4(acc_qk_mn(i, j + 0),
                                        acc_qk_mn(i, j + 1),
                                        acc_qk_mn(i, j + 2),
                                        acc_qk_mn(i, j + 3));
        mute::max(v4f32_max_current, v4f32_max_current, v4f32_max_next);
      }
      float2 v2f32_max_0 = make_float2(v4f32_max_current.x, v4f32_max_current.y);
      float2 v2f32_max_1 = make_float2(v4f32_max_current.z, v4f32_max_current.w);

      mute::max(v2f32_max_0, v2f32_max_0, v2f32_max_1);

      row_max(i) = max(v2f32_max_0.x, v2f32_max_0.y);

      for_each(make_seq<red_rank>{}, [&](auto r) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
          row_max(i) = max(row_max(i), __shfl_xor_sync(uint32_t(-1), row_max(i), stride<r>(reduction_target_qk) * j));
        }
      });

      row_max(i) = max(row_max_prev(i), row_max(i));

      Element local_max = row_max(i) == (-std::numeric_limits<Element>::infinity()) ? Element{0} : row_max(i);
      Element scale_max = params.sm_scale_log2 * local_max;

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        acc_qk_mn(i, j + 0) = acc_qk_mn(i, j + 0) * params.sm_scale_log2 - scale_max;
        acc_qk_mn(i, j + 1) = acc_qk_mn(i, j + 1) * params.sm_scale_log2 - scale_max;
        acc_qk_mn(i, j + 2) = acc_qk_mn(i, j + 2) * params.sm_scale_log2 - scale_max;
        acc_qk_mn(i, j + 3) = acc_qk_mn(i, j + 3) * params.sm_scale_log2 - scale_max;

        //float4 v4f32_ex2 = make_float4(0, 0, 0, 0);
        float4 v4f32_src = make_float4(acc_qk_mn(i, j + 0),
                                       acc_qk_mn(i, j + 1),
                                       acc_qk_mn(i, j + 2),
                                       acc_qk_mn(i, j + 3));

        mute::fast_exp2(v4f32_src, v4f32_src);

        acc_qk_mn(i, j + 0) = v4f32_src.x;
        acc_qk_mn(i, j + 1) = v4f32_src.y;
        acc_qk_mn(i, j + 2) = v4f32_src.z;
        acc_qk_mn(i, j + 3) = v4f32_src.w;
      }

      Element correction_scale = exp2f((row_max_prev(i) - local_max) * params.sm_scale_log2);

      row_sum(i) *= correction_scale;

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j += 4) {
        acc_pv_mn(i, j + 0) *= correction_scale;
        acc_pv_mn(i, j + 1) *= correction_scale;
        acc_pv_mn(i, j + 2) *= correction_scale;
        acc_pv_mn(i, j + 3) *= correction_scale;
      }

      // update sum within thread
      {
        float4 v4f32_sum_current = make_float4(acc_qk_mn(i, 0),
                                      acc_qk_mn(i, 1),
                                      acc_qk_mn(i, 2),
                                      acc_qk_mn(i, 3));

        MUTLASS_PRAGMA_UNROLL
        for (int j = 4; j < size<1>(acc_qk_mn); j += 4) {
          float4 v4f32_sum_next = make_float4(acc_qk_mn(i, j + 0),
                                          acc_qk_mn(i, j + 1),
                                          acc_qk_mn(i, j + 2),
                                          acc_qk_mn(i, j + 3));
          mute::add(v4f32_sum_current, v4f32_sum_current, v4f32_sum_next);
        }
        float2 v2f32_sum_0 = make_float2(v4f32_sum_current.x, v4f32_sum_current.y);
        float2 v2f32_sum_1 = make_float2(v4f32_sum_current.z, v4f32_sum_current.w);
        mute::add(v2f32_sum_0, v2f32_sum_0, v2f32_sum_1);

        row_sum(i) += v2f32_sum_0.x;
        row_sum(i) += v2f32_sum_0.y;
      }
    }
  }

  template <class State, class AccPV, class TiledMmaPV>
  MUTLASS_DEVICE
  auto
  tail(State& state, AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv) {
    auto& row_sum = get<0>(state);
    auto& row_max = get<1>(state);

    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    // update sum across threads

    auto reduction_target = reduction_target_n(tiled_mma_pv);
    constexpr int red_rank = decltype(rank(reduction_target))::value;

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

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_pv_mn); ++i) {
      float sum = row_sum(i);
      float inv_sum = (sum == 0.f || sum != sum) ? 0.f : __frcp_rn(sum);

      lse(i) = (sum == 0.f || sum != sum) ? -std::numeric_limits<float>::infinity() : row_max(i) * params.sm_scale + __logf(sum);

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j += 4) {
        acc_pv_mn(i, j + 0) *= inv_sum;
        acc_pv_mn(i, j + 1) *= inv_sum;
        acc_pv_mn(i, j + 2) *= inv_sum;
        acc_pv_mn(i, j + 3) *= inv_sum;
      }
    }

    return lse;
  }
};

} // namespace mutlass::fmha::collective
