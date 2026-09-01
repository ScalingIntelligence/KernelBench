#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>

#include <cstdint>

namespace mega_moe::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
  int64_t  base;
  int64_t  offsets[kNumMaxRanks];
  uint32_t rank_idx;

  static_assert(kNumRanks <= kNumMaxRanks, "Too many ranks");

  SymBuffer() : base(0), offsets{}, rank_idx(0) {
  }

  template <typename ptr_t = void*>
  __device__ ptr_t get_base_ptr() const {
    return reinterpret_cast<ptr_t>(base);
  }

  template <typename ptr_t>
  __device__ ptr_t map(const ptr_t& ptr, uint32_t dst_rank_idx) const {
    int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
    return *reinterpret_cast<ptr_t*>(&mapped_ptr);
  }
};

}  // namespace mega_moe::layout

namespace mate::mega_moe {

inline void check_cpu_i64_vector(tvm::ffi::TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU) << name << " must be a CPU tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(tensor.dtype().code == kDLInt && tensor.dtype().bits == 64 && tensor.dtype().lanes == 1)
      << name << " must be int64";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 1) << name << " must be a 1D tensor";
}

template <uint32_t kNumRanks>
inline ::mega_moe::layout::SymBuffer<kNumRanks> make_sym_buffer(tvm::ffi::TensorView ptrs, int rank_idx) {
  check_cpu_i64_vector(ptrs, "sym_buffer_ptrs");
  TVM_FFI_ICHECK_EQ(ptrs.size(0), static_cast<int64_t>(kNumRanks)) << "sym_buffer_ptrs size must match JIT num_ranks";
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, static_cast<int>(kNumRanks));

  const auto*                              values = static_cast<const int64_t*>(ptrs.data_ptr());
  ::mega_moe::layout::SymBuffer<kNumRanks> result;
  result.base     = values[rank_idx];
  result.rank_idx = static_cast<uint32_t>(rank_idx);
  for (uint32_t i = 0; i < ::mega_moe::layout::kNumMaxRanks; ++i) {
    result.offsets[i] = i < kNumRanks ? (values[i] - result.base) : 0;
  }
  return result;
}

}  // namespace mate::mega_moe
