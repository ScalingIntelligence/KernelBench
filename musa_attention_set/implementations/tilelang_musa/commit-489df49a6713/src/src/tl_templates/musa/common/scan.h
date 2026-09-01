#pragma once

#include "intrin.h"

#include <limits>

namespace tl {

struct ScanSumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }

  template <typename T> TL_DEVICE static T identity() { return T(0); }
};

template <typename T> struct ScanMaxIdentity {
  TL_DEVICE static T value() { return std::numeric_limits<T>::lowest(); }
};

#ifdef TL_MUSA_ENABLE_FP16
template <> struct ScanMaxIdentity<half> {
  TL_DEVICE static half value() { return half(-65504.0f); }
};
#endif

#ifdef TL_MUSA_ENABLE_BF16
template <> struct ScanMaxIdentity<mt_bfloat16> {
  TL_DEVICE static mt_bfloat16 value() {
    return mt_bfloat16(-3.3895313892515355e38f);
  }
};
#endif

struct ScanMaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }

#ifdef TL_MUSA_ENABLE_FP16
  TL_DEVICE half operator()(half const &x, half const &y) {
    return __hmax(x, y);
  }
#endif

#ifdef TL_MUSA_ENABLE_BF16
  TL_DEVICE mt_bfloat16 operator()(mt_bfloat16 const &x,
                                  mt_bfloat16 const &y) {
    return __hmax(x, y);
  }
#endif

  template <typename T> TL_DEVICE static T identity() {
    return ScanMaxIdentity<T>::value();
  }
};

template <typename T>
TL_DEVICE T shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_up_sync(unsigned mask, T val, int delta) {
  return __shfl_up_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_sync(unsigned mask, T val, int src_lane) {
  return __shfl_sync(mask, val, src_lane);
}

#ifdef TL_MUSA_ENABLE_FP16
TL_DEVICE half shfl_down_sync(unsigned mask, half val, int delta) {
  float raw = static_cast<float>(val);
  return half(__shfl_down_sync(mask, raw, delta));
}

TL_DEVICE half shfl_up_sync(unsigned mask, half val, int delta) {
  float raw = static_cast<float>(val);
  return half(__shfl_up_sync(mask, raw, delta));
}

TL_DEVICE half shfl_sync(unsigned mask, half val, int src_lane) {
  float raw = static_cast<float>(val);
  return half(__shfl_sync(mask, raw, src_lane));
}
#endif

#ifdef TL_MUSA_ENABLE_BF16
TL_DEVICE mt_bfloat16 shfl_down_sync(unsigned mask, mt_bfloat16 val,
                                    int delta) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_down_sync(mask, raw, delta));
}

TL_DEVICE mt_bfloat16 shfl_up_sync(unsigned mask, mt_bfloat16 val, int delta) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_up_sync(mask, raw, delta));
}

TL_DEVICE mt_bfloat16 shfl_sync(unsigned mask, mt_bfloat16 val,
                               int src_lane) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_sync(mask, raw, src_lane));
}
#endif

template <class Reducer, bool reverse, typename T, int SEG = 32>
static TL_DEVICE void InclusiveScanLine(const T *__restrict__ src,
                                        T *__restrict__ dst, int extent,
                                        int src_stride, int dst_stride) {
  if (extent <= 0)
    return;

  constexpr unsigned MASK = 0xffffffff;
  const int lane = threadIdx.x % SEG;
  T carry = Reducer::template identity<T>();
  const int num_segments = (extent + SEG - 1) / SEG;

  if constexpr (reverse) {
    for (int seg = num_segments - 1; seg >= 0; --seg) {
      const int idx = seg * SEG + lane;
      T val = (idx < extent) ? src[idx * src_stride]
                             : Reducer::template identity<T>();

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_down_sync(MASK, val, off);
        if (lane < SEG - off)
          val = Reducer()(val, n);
      }

      val = Reducer()(val, carry);

      if (idx < extent)
        dst[idx * dst_stride] = val;

      carry = tl::shfl_sync(MASK, val, 0);
    }
  } else {
    for (int seg = 0; seg < num_segments; ++seg) {
      const int idx = seg * SEG + lane;
      T val = (idx < extent) ? src[idx * src_stride]
                             : Reducer::template identity<T>();

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_up_sync(MASK, val, off);
        if (lane >= off)
          val = Reducer()(val, n);
      }

      val = Reducer()(val, carry);

      if (idx < extent)
        dst[idx * dst_stride] = val;

      carry = tl::shfl_sync(MASK, val, SEG - 1);
    }
  }
}

template <class Reducer, int threads, bool reverse = false>
struct InclusiveScan1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if (threadIdx.x >= SEG)
      return;
    InclusiveScanLine<Reducer, reverse, T, SEG>(src, dst, N, 1, 1);
  }
};

template <class Reducer, int threads, int Axis = 0, bool reverse = false>
struct InclusiveScan2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  static_assert(Axis == 0 or Axis == 1);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
    if (H <= 0 || W <= 0)
      return;

    constexpr int TILE = threads / SEG;
    const int item = threadIdx.x / SEG;

    if constexpr (Axis == 1) {
      const int num_blocks = (H + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int row = b * TILE + item;
        if (row >= H)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(
            src + row * src_stride, dst + row * dst_stride, W, 1, 1);
      }
    } else {
      const int num_blocks = (W + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int col = b * TILE + item;
        if (col >= W)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + col, dst + col, H,
                                                    src_stride, dst_stride);
      }
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanSumOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanSumOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W, src_stride, dst_stride);
  }
};

template <int threads, bool reverse = false> struct CumMax1D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanMaxOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumMax2D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanMaxOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W, src_stride, dst_stride);
  }
};

} // namespace tl
