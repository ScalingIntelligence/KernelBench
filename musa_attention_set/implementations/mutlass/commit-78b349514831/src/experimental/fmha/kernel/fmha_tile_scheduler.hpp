#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/fast_math.h>
#include <mutlass/kernel_hardware_info.h>

namespace mutlass::fmha::kernel {

using namespace mute;

template <bool kIsCausal>
struct FmhaIndividualTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_h_r;
  };

  bool valid_ = true;
  Params params;

  MUTLASS_DEVICE
  FmhaIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, TileShape const& tile_shape,
      KernelHardwareInfo hw_info = {}) {

    dim3 grid_dim;

    if constexpr (kIsCausal) {
      grid_dim.x = get<4>(problem_size);
      grid_dim.y = get<6>(problem_size);
      grid_dim.z = ceil_div(get<0>(problem_size), get<0>(tile_shape));
    } else {
      grid_dim.x = ceil_div(get<0>(problem_size), get<0>(tile_shape));
      grid_dim.y = get<4>(problem_size);
      grid_dim.z = get<6>(problem_size);
    }

    int H_R = get<4>(problem_size) / get<5>(problem_size);
    FastDivmod divmod_h_r(H_R);

    return Params { grid_dim, divmod_h_r };
  }

  template <class ProblemSize, class TileShape>
  static dim3 get_grid_shape(
      ProblemSize const& problem_size, TileShape const& tile_shape,
      KernelHardwareInfo hw_info = {}) {
    auto params = to_underlying_arguments(problem_size, tile_shape, hw_info);
    return params.grid;
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  MUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  MUTLASS_DEVICE
  auto get_work_tile_coord() {
    if constexpr (kIsCausal) {
      auto [qo_head_coord, batch_coord, qo_coord] = static_cast<uint3>(blockIdx);
      auto kv_head_coord = params.divmod_h_r.divide(qo_head_coord);
      return make_tuple(qo_coord, qo_head_coord, kv_head_coord, batch_coord);
    } else {
      auto [qo_coord, qo_head_coord, batch_coord] = static_cast<uint3>(blockIdx);
      auto kv_head_coord = params.divmod_h_r.divide(qo_head_coord);
      return make_tuple(qo_coord, qo_head_coord, kv_head_coord, batch_coord);
    }
  }

  MUTLASS_DEVICE
  FmhaIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};



} // namespace mutlass::fmha::kernel
