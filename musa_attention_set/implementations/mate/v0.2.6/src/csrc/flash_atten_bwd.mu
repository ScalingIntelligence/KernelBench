#include <mudnn_xmma.h>
#include <musa.h>
#include <musa_runtime.h>

#include <initializer_list>

#include "mudnn_utils.hpp"
#include "op_utils.hpp"

void dnn_mha_varlen_bwd(ffi::TensorView                dout,
                        ffi::TensorView                q,
                        ffi::TensorView                k,
                        ffi::TensorView                v,
                        ffi::TensorView                out,
                        ffi::TensorView                softmax_lse,
                        ffi::Optional<ffi::TensorView> dq_opt,
                        ffi::Optional<ffi::TensorView> dk_opt,
                        ffi::Optional<ffi::TensorView> dv_opt,
                        ffi::TensorView                cu_seqlens_q,
                        ffi::TensorView                cu_seqlens_k,
                        ffi::Optional<ffi::TensorView> alibi_slopes,
                        int64_t                        max_seqlen_q,
                        int64_t                        max_seqlen_k,
                        double                         p_dropout,
                        double                         softmax_scale,
                        bool                           zero_tensors,
                        bool                           is_causal,
                        int64_t                        window_size_left,
                        int64_t                        window_size_right,
                        double                         softcap,
                        bool                           deterministic,
                        ffi::Optional<ffi::TensorView> gen,
                        ffi::Optional<ffi::TensorView> rng_state) {
  (void)alibi_slopes;
  (void)deterministic;
  (void)gen;
  (void)rng_state;

  if (is_causal) {
    window_size_right = 0;
  }

  check_mp31(q.device(), "dnn_mha_varlen_bwd");

  CHECK_MUSA(dout);
  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(v);
  CHECK_MUSA(out);
  CHECK_MUSA(softmax_lse);
  CHECK_MUSA(cu_seqlens_q);
  CHECK_MUSA(cu_seqlens_k);
  CHECK_DEVICE(q, dout);
  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, v);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, softmax_lse);
  CHECK_DEVICE(q, cu_seqlens_q);
  CHECK_DEVICE(q, cu_seqlens_k);
  TVM_FFI_ICHECK_EQ(q.stride(-1), 1) << "q must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(k.stride(-1), 1) << "k must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(v.stride(-1), 1) << "v must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(out.stride(-1), 1) << "out must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(dout.stride(-1), 1) << "dout must have contiguous last dimension";
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  CHECK_HAS_VALUE(dq_opt);
  CHECK_HAS_VALUE(dk_opt);
  CHECK_HAS_VALUE(dv_opt);
  ffi::TensorView dq = dq_opt.value();
  ffi::TensorView dk = dk_opt.value();
  ffi::TensorView dv = dv_opt.value();

  CHECK_MUSA(dq);
  CHECK_MUSA(dk);
  CHECK_MUSA(dv);
  CHECK_DEVICE(q, dq);
  CHECK_DEVICE(q, dk);
  CHECK_DEVICE(q, dv);
  TVM_FFI_ICHECK_EQ(dq.stride(-1), 1) << "dq must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(dk.stride(-1), 1) << "dk must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(dv.stride(-1), 1) << "dv must have contiguous last dimension";

  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(q.dtype())) << "FlashAttention only supports fp16 and bf16";
  TVM_FFI_ICHECK(dtype_equal(k.dtype(), q.dtype())) << "q and k must have the same dtype";
  TVM_FFI_ICHECK(dtype_equal(v.dtype(), q.dtype())) << "q and v must have the same dtype";
  TVM_FFI_ICHECK(dtype_equal(out.dtype(), q.dtype())) << "q and out must have the same dtype";
  TVM_FFI_ICHECK(dtype_equal(dout.dtype(), q.dtype())) << "q and dout must have the same dtype";
  TVM_FFI_ICHECK(dtype_equal(dq.dtype(), q.dtype())) << "dq must have the same dtype as q";
  TVM_FFI_ICHECK(dtype_equal(dk.dtype(), q.dtype())) << "dk must have the same dtype as q";
  TVM_FFI_ICHECK(dtype_equal(dv.dtype(), q.dtype())) << "dv must have the same dtype as q";
  TVM_FFI_ICHECK(dtype_equal(cu_seqlens_q.dtype(), dl_int32)) << "cu_seqlens_q must be int32";
  TVM_FFI_ICHECK(dtype_equal(cu_seqlens_k.dtype(), dl_int32)) << "cu_seqlens_k must be int32";
  TVM_FFI_ICHECK(dtype_equal(softmax_lse.dtype(), dl_float32)) << "softmax_lse must be float32";

  const int total_q     = static_cast<int>(q.size(0));
  const int batch_size  = static_cast<int>(cu_seqlens_q.numel()) - 1;
  const int num_heads   = static_cast<int>(q.size(1));
  const int head_size   = static_cast<int>(q.size(2));
  const int head_size_v = static_cast<int>(v.size(2));
  const int total_k     = static_cast<int>(k.size(0));
  const int num_heads_k = static_cast<int>(k.size(1));

  TVM_FFI_ICHECK(batch_size > 0) << "batch size must be positive";
  TVM_FFI_ICHECK(head_size % 8 == 0) << "head_size should be a multiple of 8";
  TVM_FFI_ICHECK(head_size <= 256) << "FlashAttention backward only supports head dimension at most 256";
  TVM_FFI_ICHECK(num_heads % num_heads_k == 0) << "num_heads_k must divide num_heads";
  TVM_FFI_ICHECK(softcap == 0.0) << "Softcapping does not support dropout for now";

  if (window_size_left >= max_seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k) {
    window_size_right = -1;
  }
  (void)window_size_left;
  (void)window_size_right;
  (void)p_dropout;

  expect_shape(q, {total_q, num_heads, head_size}, "q");
  expect_shape(k, {total_k, num_heads_k, head_size}, "k");
  expect_shape(v, {total_k, num_heads_k, head_size_v}, "v");
  expect_shape(out, {total_q, num_heads, head_size_v}, "out");
  expect_shape(dout, {total_q, num_heads, head_size_v}, "dout");
  expect_shape(dq, {total_q, num_heads, head_size}, "dq");
  expect_shape(dk, {total_k, num_heads_k, head_size}, "dk");
  expect_shape(dv, {total_k, num_heads_k, head_size_v}, "dv");
  expect_shape(softmax_lse, {batch_size, num_heads, max_seqlen_q}, "softmax_lse");
  expect_shape(cu_seqlens_q, {batch_size + 1}, "cu_seqlens_q");
  expect_shape(cu_seqlens_k, {batch_size + 1}, "cu_seqlens_k");

  if (zero_tensors) {
    fill_mudnn_tensor(dq.device(), get_stream(dq.device()), dq.data_ptr(), dq.dtype(), dq.shape(), dq.strides(), 0.0);
    fill_mudnn_tensor(dk.device(), get_stream(dk.device()), dk.data_ptr(), dk.dtype(), dk.shape(), dk.strides(), 0.0);
    fill_mudnn_tensor(dv.device(), get_stream(dv.device()), dv.data_ptr(), dv.dtype(), dv.shape(), dv.strides(), 0.0);
  }

  ffi::MUSADeviceGuard device_guard(q.device().device_id);
  musa::dnn::Handle    handle(q.device().device_id);
  MATE_MUDNN_STATUS_CHECK(handle.SetStream(get_stream(q.device())));

  musa::dnn::ScaledDotProductAttention op;
  op.SetEmbedDim(num_heads * head_size_v);
  op.SetHeadsNum(num_heads);
  op.SetMaskMode(false);
  op.SetCausal(is_causal);
  op.SetBatchSize(batch_size);
  op.SetMaxSeqlenQ(static_cast<int>(max_seqlen_q));
  op.SetMaxSeqlenK(static_cast<int>(max_seqlen_k));
  op.SetScale(softmax_scale);
  op.SetComputeMode(static_cast<musa::dnn::ScaledDotProductAttention::ComputeMode>(0));

  musa::dnn::Tensor q_dnn = make_mudnn_tensor(
      q.data_ptr(), q.dtype(), {total_q, num_heads, head_size}, {q.stride(0), q.stride(1), q.stride(2)});
  musa::dnn::Tensor k_dnn = make_mudnn_tensor(
      k.data_ptr(), k.dtype(), {total_k, num_heads_k, head_size}, {k.stride(0), k.stride(1), k.stride(2)});
  musa::dnn::Tensor v_dnn = make_mudnn_tensor(
      v.data_ptr(), v.dtype(), {total_k, num_heads_k, head_size_v}, {v.stride(0), v.stride(1), v.stride(2)});
  musa::dnn::Tensor out_dnn = make_mudnn_tensor(
      out.data_ptr(), out.dtype(), {total_q, num_heads, head_size_v}, {out.stride(0), out.stride(1), out.stride(2)});
  musa::dnn::Tensor dout_dnn = make_mudnn_tensor(dout.data_ptr(),
                                                 dout.dtype(),
                                                 {total_q, num_heads, head_size_v},
                                                 {dout.stride(0), dout.stride(1), dout.stride(2)});
  musa::dnn::Tensor lse_dnn =
      make_mudnn_tensor(softmax_lse.data_ptr(),
                        dl_float32,
                        {batch_size, num_heads, max_seqlen_q, 1},
                        {softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2), 1});
  musa::dnn::Tensor dq_dnn = make_mudnn_tensor(
      dq.data_ptr(), dq.dtype(), {total_q, num_heads, head_size}, {dq.stride(0), dq.stride(1), dq.stride(2)});
  musa::dnn::Tensor dk_dnn = make_mudnn_tensor(
      dk.data_ptr(), dk.dtype(), {total_k, num_heads_k, head_size}, {dk.stride(0), dk.stride(1), dk.stride(2)});
  musa::dnn::Tensor dv_dnn = make_mudnn_tensor(
      dv.data_ptr(), dv.dtype(), {total_k, num_heads_k, head_size_v}, {dv.stride(0), dv.stride(1), dv.stride(2)});
  musa::dnn::Tensor cu_seqlens_q_dnn =
      make_mudnn_tensor(cu_seqlens_q.data_ptr(), dl_int32, {cu_seqlens_q.size(0)}, {1});
  musa::dnn::Tensor cu_seqlens_k_dnn =
      make_mudnn_tensor(cu_seqlens_k.data_ptr(), dl_int32, {cu_seqlens_k.size(0)}, {1});
  musa::dnn::Tensor empty_dropout_mask;
  musa::dnn::Tensor empty_mask;

  MATE_MUDNN_STATUS_CHECK(op.RunFlashVarlenBwd(handle,
                                               dq_dnn,
                                               dk_dnn,
                                               dv_dnn,
                                               dout_dnn,
                                               q_dnn,
                                               k_dnn,
                                               v_dnn,
                                               empty_mask,
                                               out_dnn,
                                               lse_dnn,
                                               empty_dropout_mask,
                                               cu_seqlens_q_dnn,
                                               cu_seqlens_k_dnn,
                                               mudnn_internal_mem_alloc));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dnn_mha_varlen_bwd, dnn_mha_varlen_bwd);
