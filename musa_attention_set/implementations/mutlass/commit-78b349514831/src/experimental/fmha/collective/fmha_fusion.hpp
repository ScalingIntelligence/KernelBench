#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/fast_math.h"
#include "mute/tensor.hpp"

namespace mutlass::fmha::collective {

struct DefaultFusion {
  struct Arguments {
    // Varlen prefill only
    int prev_mask_boundary = 0;

    // PackGQA
    FastDivmod fast_divmod_hr;
  };

  using Params = Arguments;

  MUTLASS_HOST
  static Params
  to_underlying_arguments(Arguments& args) {
    return args;
  }

  Params params;

  template <class BlkCoord, class TileShape, class ProblemSize>
  MUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) const {
    return ceil_div(get<1>(problem_size), get<1>(tile_shape));
  }


  template <class BlkCoord, class TileShape, class ProblemSize>
  MUTLASS_DEVICE
  int get_masked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) const {
    return 1;
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  MUTLASS_DEVICE
  int get_unmasked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) const {
    return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
  }


  template <bool PrevMask = false, class AccQK, class IndexQK, class ProblemSize>
  MUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size
  ) const {
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); ++i) {
      auto pos = index_qk(i);
      if constexpr (PrevMask) {
        if (get<1>(pos) < params.prev_mask_boundary || get<1>(pos) >= get<1>(problem_size)) {
          acc_qk(i) = -std::numeric_limits<float>::infinity();
        }
      } else {
        if (get<1>(pos) >= get<1>(problem_size)) {
          acc_qk(i) = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }
};

template <
  bool IsUpperLeft = true, // UpperLeft or BottomRight causal mask
  bool PackGQA = false
>
struct CausalFusion : DefaultFusion {
  using Base = DefaultFusion;

  template <class BlkCoord, class TileShape, class ProblemSize>
  MUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) const {
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);

    int offset_q = IsUpperLeft ? 0 : (get<1>(problem_size) - this->params.prev_mask_boundary) - get<0>(problem_size);
    offset_q += this->params.prev_mask_boundary;

    int max_index_q = (get<0>(blk_coord) + 1) * get<0>(tile_shape);
    if constexpr (PackGQA) {
      max_index_q = this->params.fast_divmod_hr.divide(max_index_q - 1) + 1;
    }
    int max_blocks_q = ceil_div(max_index_q + offset_q, get<1>(tile_shape));

    return std::min(max_blocks_k, max_blocks_q);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  MUTLASS_DEVICE
  int get_masked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) const {
    int trip_count = get_trip_count(blk_coord, tile_shape, problem_size);
    // TODO: we may over estimate masked trip count
    return std::min(trip_count, ceil_div(get<0>(tile_shape), get<1>(tile_shape)) + 1);
  }

  template <class BlkCoord, class TileShape, class ProblemSize>
  MUTLASS_DEVICE
  int get_unmasked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) const {
    return get_trip_count(blk_coord, tile_shape, problem_size) -
           get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template <bool PrevMask = false, class AccQK, class IndexQK, class ProblemSize>
  MUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size
  ) const {
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); ++i) {
      auto pos = index_qk(i);
      auto [m_pos, n_pos] = pos;

      // oob mask
      bool need_mask = n_pos >= get<1>(problem_size);

      // TME varlen previous mask
      if constexpr (PrevMask) {
        need_mask = need_mask || (n_pos < this->params.prev_mask_boundary);
      }

      // causal mask
      int offset_q = IsUpperLeft ? 0 : (get<1>(problem_size) - this->params.prev_mask_boundary) - get<0>(problem_size);
      int row = (PackGQA ? this->params.fast_divmod_hr.divide(m_pos) : m_pos) + offset_q;

      int col = n_pos - this->params.prev_mask_boundary;
      need_mask = need_mask || row < col;


      if (need_mask) {
        acc_qk(i) = -std::numeric_limits<float>::infinity();
      }
    }
  }
};

} // namespace mutlass::fmha::collective
