#pragma once

#include <mutlass/mutlass.h>

namespace mate::flash_mla {

using namespace mute;
using namespace mutlass;
using namespace mutlass::fmha::collective;

template <bool UseTrivialScheduler_>
struct FlashMlaTileScheduler {
  static constexpr int  TileSchedulerMetaDataSize = 8;
  static constexpr bool UseTrivialScheduler       = UseTrivialScheduler_;

  struct Arguments {
    int num_mp_parts;
    int* __restrict__ metadata_ptr;
    int* __restrict__ num_splits_ptr;
  };

  struct Params {
    int num_mp_parts;
    int* __restrict__ metadata_ptr;
    int* __restrict__ num_splits_ptr;

    dim3 grid;
  };

  Params params;

  MUTLASS_DEVICE
  FlashMlaTileScheduler(Params const& params) : params(params) {
  }

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(ProblemSize const& problem_size,
                                        TileShape const&   tile_shape,
                                        Arguments          args = {}) {
    dim3 grid_dim;
    auto [Q, PageSize, D, H, PageCount, B, TotalQ, MaxQ] = problem_size;

    grid_dim.x = ceil_div(MaxQ * H, get<0>(TileShape{}));
    if constexpr (UseTrivialScheduler) {
      grid_dim.y = 1;  // since splits=1
      grid_dim.z = B;
    } else {
      grid_dim.y = 1;  // since h_k=1
      grid_dim.z = args.num_mp_parts;
    }

    return {args.num_mp_parts, args.metadata_ptr, args.num_splits_ptr, grid_dim};
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }
};

}  // namespace mate::flash_mla
