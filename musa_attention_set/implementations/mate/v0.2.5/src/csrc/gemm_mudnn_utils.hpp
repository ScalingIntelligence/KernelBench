#pragma once

#include <string>

#include "gemm_common_utils.hpp"

namespace mate::gemm::mudnn {

inline void init_mudnn_handle(musa::dnn::Handle& handle, musaStream_t stream) {
  MATE_MUDNN_STATUS_CHECK(handle.SetStream(stream));
}

inline void validate_mudnn_backend(const std::string& backend, const char* func_name) {
  if (backend == "mudnn" || backend == "auto") {
    return;
  }
  TVM_FFI_THROW(ValueError) << func_name << " got unsupported backend: " << backend;
}

inline void run_mudnn_lt_matmul(musa::dnn::Handle&              handle,
                                musa::dnn::Tensor&              output,
                                const musa::dnn::Tensor&        a,
                                const musa::dnn::Tensor&        b,
                                const musa::dnn::Tensor&        c,
                                bool                            use_c,
                                TensorMajor                     major_a,
                                TensorMajor                     major_b,
                                const musa::dnn::MatMulLtParam& lt_param,
                                bool                            deterministic = false) {
  musa::dnn::BatchMatMul bmm;
  MATE_MUDNN_STATUS_CHECK(bmm.SetComputeMode(musa::dnn::BatchMatMul::ComputeMode::TENSOR));
  MATE_MUDNN_STATUS_CHECK(bmm.SetTranspose(major_a != TensorMajor::K, major_b == TensorMajor::K));
  MATE_MUDNN_STATUS_CHECK(bmm.SetDeterministic(deterministic));
  if (use_c) {
    MATE_MUDNN_STATUS_CHECK(bmm.SetBeta(1.0));
  }
  MATE_MUDNN_STATUS_CHECK(bmm.RunLt(handle, output, a, b, c, musa::dnn::Tensor{}, lt_param, nullptr));
}

inline void run_mudnn_lt_matmul(musa::dnn::Handle&              handle,
                                musa::dnn::Tensor&              output,
                                const musa::dnn::Tensor&        a,
                                const musa::dnn::Tensor&        b,
                                TensorMajor                     major_a,
                                TensorMajor                     major_b,
                                const musa::dnn::MatMulLtParam& lt_param,
                                bool                            deterministic = false) {
  run_mudnn_lt_matmul(handle, output, a, b, musa::dnn::Tensor{}, false, major_a, major_b, lt_param, deterministic);
}

}  // namespace mate::gemm::mudnn
