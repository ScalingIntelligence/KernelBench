#pragma once

#include <musa.h>

#ifndef TL_DEVICE
#define TL_DEVICE __forceinline__ __device__
#endif

#ifndef TL_DEVICE_NOINLINE
#define TL_DEVICE_NOINLINE __noinline__ __device__
#endif

#ifndef TL_HOST_DEVICE
#define TL_HOST_DEVICE __forceinline__ __host__ __device__
#endif

#ifndef TL_PATCH
#define TL_PATCH
#endif

namespace tl {
namespace detail {

TL_DEVICE constexpr int default_warp_size() {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ <= 220)
  return 128;
#else
  return 32;
#endif
}

TL_DEVICE constexpr int default_warps_per_group() { return 4; }

TL_DEVICE int linear_thread_idx_in_block() {
#if defined(__MUSA_ARCH__)
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
  return 0;
#endif
}

} // namespace detail
} // namespace tl

namespace tl {

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int
get_warp_group_idx(int warp_size = detail::default_warp_size(),
                   int warps_per_group = detail::default_warps_per_group()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  warps_per_group =
      warps_per_group > 0 ? warps_per_group : detail::default_warps_per_group();
  int threads_per_group = warp_size * warps_per_group;
  threads_per_group = threads_per_group > 0 ? threads_per_group : warp_size;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}

TL_DEVICE uint32_t elect_one_sync() { return get_lane_idx() == 0; }

TL_DEVICE int canonical_warp_idx_sync() { return get_warp_idx_sync(); }

template <int y = 1, typename T>
TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  if constexpr (thread_extent == 0) {
    return canonical_warp_idx_sync() == 0 && elect_one_sync();
  } else {
    constexpr int warp_size = detail::default_warp_size();
    constexpr int warp_extent = (thread_extent + warp_size - 1) / warp_size;
    static_assert(warp_extent > 0);
    return (canonical_warp_idx_sync() % warp_extent) == 0 && elect_one_sync();
  }
}

} // namespace tl
