#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate_utils.hpp"
#include "tvm_ffi_utils.hpp"

namespace {

void check_cpu_tensor(ffi::TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU) << name << " must be a CPU tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

}  // namespace

int64_t mega_moe_get_ipc_handle_size() {
  return static_cast<int64_t>(sizeof(musaIpcMemHandle_t));
}

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

ffi::Tensor make_tensor_view(ffi::Tensor buffer, uint64_t byte_offset, std::vector<int64_t> shape, DLDataType dtype) {
  DLTensor prototype    = *buffer.GetDLTensorPtr();
  auto     strides      = make_contiguous_strides(shape);
  prototype.data        = static_cast<uint8_t*>(buffer.data_ptr()) + byte_offset;
  prototype.dtype       = dtype;
  prototype.ndim        = static_cast<int32_t>(shape.size());
  prototype.shape       = shape.data();
  prototype.strides     = strides.data();
  prototype.byte_offset = 0;

  TVMFFIObjectHandle out;
  auto*              obj_handle = ffi::details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(buffer);
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorCreateUnsafeView(obj_handle, &prototype, &out));
  return ffi::Tensor(ffi::details::ObjectUnsafe::ObjectPtrFromOwned<ffi::TensorObj>(static_cast<TVMFFIObject*>(out)));
}

void mega_moe_get_ipc_handle(ffi::TensorView tensor, ffi::TensorView handle_out) {
  CHECK_MUSA(tensor);
  check_cpu_tensor(handle_out, "handle_out");
  CHECK_INPUT_TYPE(handle_out, dl_uint8);
  TVM_FFI_ICHECK_GE(handle_out.numel(), static_cast<int64_t>(sizeof(musaIpcMemHandle_t)))
      << "handle_out is too small for musaIpcMemHandle_t";

  musaIpcMemHandle_t handle{};
  MATE_MUSA_RUNTIME_CHECK(musaIpcGetMemHandle(&handle, tensor.data_ptr()));
  std::memcpy(handle_out.data_ptr(), &handle, sizeof(handle));
}

void mega_moe_open_ipc_handles(ffi::TensorView handles,
                               ffi::TensorView buffer,
                               int             rank_idx,
                               ffi::TensorView ptrs_out) {
  check_cpu_tensor(handles, "handles");
  check_cpu_tensor(ptrs_out, "ptrs_out");
  CHECK_INPUT_TYPE(handles, dl_uint8);
  CHECK_INPUT_TYPE(ptrs_out, dl_int64);
  CHECK_MUSA(buffer);
  CHECK_DIM(2, handles);
  CHECK_DIM(1, ptrs_out);

  const int64_t num_ranks   = handles.size(0);
  const int64_t handle_size = handles.size(1);
  TVM_FFI_ICHECK_GE(handle_size, static_cast<int64_t>(sizeof(musaIpcMemHandle_t)))
      << "handles second dimension is too small for musaIpcMemHandle_t";
  TVM_FFI_ICHECK_EQ(ptrs_out.size(0), num_ranks) << "ptrs_out shape must match handles.shape[0]";
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, num_ranks);

  const auto* handle_bytes = static_cast<const uint8_t*>(handles.data_ptr());
  auto*       ptrs         = static_cast<int64_t*>(ptrs_out.data_ptr());
  for (int64_t i = 0; i < num_ranks; ++i) {
    if (i == rank_idx) {
      ptrs[i] = reinterpret_cast<int64_t>(buffer.data_ptr());
      continue;
    }
    musaIpcMemHandle_t handle{};
    std::memcpy(&handle, handle_bytes + i * handle_size, sizeof(handle));
    void* ptr = nullptr;
    MATE_MUSA_RUNTIME_CHECK(musaIpcOpenMemHandle(&ptr, handle, musaIpcMemLazyEnablePeerAccess));
    ptrs[i] = reinterpret_cast<int64_t>(ptr);
  }
}

void mega_moe_close_ipc_handles(ffi::TensorView ptrs, int rank_idx) {
  check_cpu_tensor(ptrs, "ptrs");
  CHECK_INPUT_TYPE(ptrs, dl_int64);
  CHECK_DIM(1, ptrs);

  const auto*   values    = static_cast<const int64_t*>(ptrs.data_ptr());
  const int64_t num_ranks = ptrs.size(0);
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, num_ranks);
  for (int64_t i = 0; i < num_ranks; ++i) {
    if (i == rank_idx || values[i] == 0) {
      continue;
    }
    MATE_MUSA_RUNTIME_CHECK(musaIpcCloseMemHandle(reinterpret_cast<void*>(values[i])));
  }
}

std::tuple<int64_t, ffi::Function> mega_moe_get_buffer_layout(
    int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden) {
  TVM_FFI_ICHECK_GT(num_ranks, 0);
  TVM_FFI_ICHECK_GT(num_experts, 0);
  TVM_FFI_ICHECK_EQ(num_experts % num_ranks, 0);
  TVM_FFI_ICHECK_GT(num_max_tokens_per_rank, 0);
  TVM_FFI_ICHECK_GT(num_topk, 0);
  TVM_FFI_ICHECK_GT(hidden, 0);
  TVM_FFI_ICHECK_GT(intermediate_hidden, 0);
  TVM_FFI_ICHECK_EQ(hidden % 128, 0);
  TVM_FFI_ICHECK_EQ(intermediate_hidden % 128, 0);

  constexpr uint32_t block_m = 32;
  const auto         layout  = ::mega_moe::layout::get_symm_buffer_layout_info(static_cast<uint32_t>(num_ranks),
                                                                      static_cast<uint32_t>(num_experts),
                                                                      static_cast<uint32_t>(num_max_tokens_per_rank),
                                                                      static_cast<uint32_t>(num_topk),
                                                                      static_cast<uint32_t>(hidden),
                                                                      static_cast<uint32_t>(intermediate_hidden),
                                                                      block_m);

  auto slice_buffer = ffi::Function::FromTyped([=](ffi::Tensor buffer) {
    CHECK_INPUT_TYPE(buffer, dl_int8);
    CHECK_MUSA(buffer);
    TVM_FFI_ICHECK_GE(buffer.numel() * get_element_size(buffer), static_cast<int64_t>(layout.num_bytes))
        << "MegaMoE symmetric buffer is smaller than required layout size";

    return std::make_tuple(
        make_tensor_view(buffer, layout.input_token_offset, {num_max_tokens_per_rank, hidden}, dl_int8),
        make_tensor_view(buffer, layout.input_sf_offset, {num_max_tokens_per_rank, hidden / 128}, dl_float32),
        make_tensor_view(buffer, layout.input_topk_idx_offset, {num_max_tokens_per_rank, num_topk}, dl_int64),
        make_tensor_view(buffer, layout.input_topk_weights_offset, {num_max_tokens_per_rank, num_topk}, dl_float32),
        make_tensor_view(
            buffer, layout.l1_token_offset, {static_cast<int64_t>(layout.num_max_pool_tokens), hidden}, dl_int8),
        make_tensor_view(buffer,
                         layout.l1_sf_offset,
                         {static_cast<int64_t>(layout.num_padded_sf_pool_tokens), hidden / 128},
                         dl_float32),
        make_tensor_view(
            buffer, layout.l1_topk_weights_offset, {static_cast<int64_t>(layout.num_max_pool_tokens)}, dl_float32),
        make_tensor_view(buffer,
                         layout.l2_token_offset,
                         {static_cast<int64_t>(layout.num_max_pool_tokens), intermediate_hidden},
                         dl_int8),
        make_tensor_view(buffer,
                         layout.l2_sf_offset,
                         {static_cast<int64_t>(layout.num_padded_sf_pool_tokens), intermediate_hidden / 128},
                         dl_float32),
        make_tensor_view(
            buffer, layout.combine_token_offset, {num_topk, num_max_tokens_per_rank, hidden}, dl_bfloat16));
  });

  return {static_cast<int64_t>(layout.num_bytes), slice_buffer};
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_get_ipc_handle_size, mega_moe_get_ipc_handle_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_get_ipc_handle, mega_moe_get_ipc_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_open_ipc_handles, mega_moe_open_ipc_handles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_close_ipc_handles, mega_moe_close_ipc_handles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_get_buffer_layout, mega_moe_get_buffer_layout);
