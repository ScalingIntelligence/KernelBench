#include <mutlass/device_kernel.h>

#include <algorithm>
#include <optional>

#include "attention_combine.hpp"
#include "attention_scheduler.hpp"
#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_mla_collective_tme_warpspecialized.hpp"
#include "kernel/mla_kernel_tme_warpspecialzed.hpp"
#include "mate/attention/flash_mla/fmha_mla_collective_tme_warpspecialized.hpp"
#include "mate/attention/flash_mla/mla_collective_epilogue.hpp"
#include "mate/attention/flash_mla/mla_kernel_tme_warpspecialized.hpp"
#include "mate/attention/flash_mla/mla_tile_scheduler.hpp"
#include "mate/attention/flash_mla/mpxx_mla_combine.hpp"
#include "mate/attention/flash_mla/mpxx_params.hpp"
#include "op_utils.hpp"

using namespace mute;

struct DecodingAttnImplMeta {
  int num_mp_parts;
  int fixed_overhead_num_blocks;
  int k_block_size;
};

DecodingAttnImplMeta get_attn_impl_meta(Arch                   arch,
                                        int                    mp_count,
                                        int                    num_q_tokens_per_head_k,
                                        int                    h_k,
                                        ffi::Optional<int64_t> h_q,
                                        bool                   is_fp8_kvcache,
                                        bool                   is_sparse_attn,
                                        int                    tile_m = 128) {
  (void)h_q;
  if (arch.is_mp31()) {
    if (is_sparse_attn) {
      if (is_fp8_kvcache) {
        return {std::max(mp_count / h_k / mutlass::ceil_div(num_q_tokens_per_head_k, tile_m), 1), 5, 64};
      } else {
        TVM_FFI_ICHECK(false) << "Sparse BF16 MLA is not supported on MP31";
      }
    } else {
      if (is_fp8_kvcache) {
        TVM_FFI_ICHECK(false) << "Dense FP8 MLA is not supported on MP31";
      } else {
        return {std::max(mp_count / h_k / mutlass::ceil_div(num_q_tokens_per_head_k, tile_m), 1), 5, 64};
      }
    }
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported GPU architecture";
  }
  return {0, 0, 0};
}

namespace {

struct MetadataResult {
  ffi::Tensor     tile_scheduler_metadata_storage;
  ffi::Tensor     num_splits_storage;
  ffi::TensorView tile_scheduler_metadata;
  ffi::TensorView num_splits;
};

}  // namespace

template <typename Element, typename ElementO, bool IsCausal>
void dispatch_mla_kernel(ffi::TensorView q_nope,
                         ffi::TensorView q_pe,
                         ffi::TensorView ckv,
                         ffi::TensorView kpe,
                         ffi::TensorView out,
                         ffi::TensorView softmax_lse,
                         ffi::TensorView page_table,
                         ffi::TensorView kv_len,
                         float           sm_scale,
                         int             batch,
                         int             page_count,
                         int             page_size,
                         int             head_num_q,
                         int             head_dim_ckv,
                         int             head_dim_kpe,
                         int             seqlen_q,
                         int64_t         q_nope_stride_batch,
                         int64_t         q_nope_stride_num_heads,
                         int64_t         q_pe_stride_batch,
                         int64_t         q_pe_stride_num_heads,
                         int64_t         ckv_stride_page_num,
                         int64_t         ckv_stride_num_heads,
                         int64_t         kpe_stride_page_num,
                         int64_t         kpe_stride_num_heads,
                         int32_t         pt_stride_batch,
                         int64_t         o_stride_batch,
                         int64_t         o_stride_num_heads,
                         int32_t         lse_stride_batch,
                         int32_t         lse_stride_num_heads) {
  auto stride_q_nope = make_stride(q_nope_stride_num_heads, _1{}, make_stride(_0{}, q_nope_stride_batch));
  auto stride_q_pe   = make_stride(q_pe_stride_num_heads, _1{}, make_stride(_0{}, q_pe_stride_batch));
  auto stride_ckv    = make_stride(ckv_stride_num_heads, _1{}, make_stride(_0{}, ckv_stride_page_num));
  auto stride_kpe    = make_stride(kpe_stride_num_heads, _1{}, make_stride(_0{}, kpe_stride_page_num));
  auto stride_pt     = make_stride(pt_stride_batch, _1{});
  auto stride_o      = make_stride(o_stride_num_heads, _1{}, make_stride(_0{}, o_stride_batch));
  auto stride_lse    = make_stride(_1{}, make_stride(lse_stride_num_heads, lse_stride_batch));

  using StrideQ   = decltype(stride_q_nope);
  using StrideC   = decltype(stride_ckv);
  using StrideO   = decltype(stride_o);
  using StrideLse = decltype(stride_lse);
  using TileShape = Shape<_32, _64, Shape<_512, _64>>;

  using Fusion = conditional_t<IsCausal,
                               mutlass::fmha::collective::CausalFusion<false, true>,
                               mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::
      FmhaMlaMainloopTmeWarpSpecializedV2<Element, float, TileShape, StrideQ, StrideC, Fusion>;
  using CollectiveEpilogue =
      mutlass::fmha::collective::FmhaFwdEpilogue<Element, float, Shape<_32, _512>, StrideO, StrideLse>;
  using MlaKernel = mutlass::fmha::kernel::MlaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue>;

  using ProblemShapeType = typename MlaKernel::ProblemShape;
  ProblemShapeType problem_shape =
      make_shape(seqlen_q, page_size, make_shape(head_dim_ckv, head_dim_kpe), head_num_q, page_count, batch);
  typename MlaKernel::CollectiveMainloop::Fusion::Arguments fusion;
  int                                                       head_num_kv = 1;
  mutlass::FastDivmod                                       divmod_hr(head_num_q / head_num_kv);
  fusion.fast_divmod_hr = divmod_hr;

  typename MlaKernel::Arguments arguments{
      problem_shape,
      {static_cast<Element*>(q_nope.data_ptr()),
       stride_q_nope,
       static_cast<Element*>(q_pe.data_ptr()),
       stride_q_pe,
       static_cast<Element*>(ckv.data_ptr()),
       stride_ckv,
       static_cast<Element*>(kpe.data_ptr()),
       stride_kpe,
       static_cast<int32_t*>(page_table.data_ptr()),
       stride_pt,
       static_cast<int32_t*>(kv_len.data_ptr()),
       sm_scale,
       fusion},
      {
          static_cast<ElementO*>(out.data_ptr()),
          stride_o,
          static_cast<float*>(softmax_lse.data_ptr()),
          stride_lse,
      },
  };

  musaStream_t               stream     = get_stream(q_nope.device());
  typename MlaKernel::Params params     = MlaKernel::to_underlying_arguments(arguments);
  int                        num_splits = 1;

  dim3 grid_dim{static_cast<uint32_t>(ceil_div(head_num_q * seqlen_q, get<0>(TileShape{}))),
                static_cast<uint32_t>(num_splits),
                static_cast<uint32_t>(get<5>(problem_shape))};
  mutlass::device_kernel<MlaKernel>
      <<<grid_dim, MlaKernel::MaxThreadsPerBlock, MlaKernel::SharedStorageSize, stream>>>(params);
}

void mla(ffi::TensorView q_nope,
         ffi::TensorView q_pe,
         ffi::TensorView ckv,
         ffi::TensorView kpe,
         ffi::TensorView page_table,
         ffi::TensorView kv_len,
         ffi::TensorView out,
         ffi::TensorView lse,
         double          sm_scale,
         bool            is_causal) {
  check_mp31(q_nope.device(), "mla");

  CHECK_MUSA(q_nope);
  CHECK_MUSA(q_pe);
  CHECK_MUSA(ckv);
  CHECK_MUSA(kpe);
  CHECK_MUSA(page_table);
  CHECK_MUSA(kv_len);
  CHECK_MUSA(out);
  CHECK_MUSA(lse);
  CHECK_DEVICE(q_nope, q_pe);
  CHECK_DEVICE(q_nope, ckv);
  CHECK_DEVICE(q_nope, kpe);
  CHECK_DEVICE(q_nope, page_table);
  CHECK_DEVICE(q_nope, kv_len);
  CHECK_DEVICE(q_nope, out);
  CHECK_DEVICE(q_nope, lse);
  TVM_FFI_ICHECK_EQ(q_nope.stride(-1), 1) << "q_nope must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(q_pe.stride(-1), 1) << "q_pe must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(ckv.stride(-1), 1) << "ckv must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(kpe.stride(-1), 1) << "kpe must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(page_table.stride(-1), 1) << "page_table must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(kv_len.stride(-1), 1) << "kv_len must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(lse.stride(-1), 1) << "lse must have contiguous last dimension";

  const DLDataType q_type = q_nope.dtype();
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(q_type)) << "mla only supports fp16 and bf16";
  TVM_FFI_ICHECK(dtype_equal(q_pe.dtype(), q_type) && dtype_equal(ckv.dtype(), q_type) &&
                 dtype_equal(kpe.dtype(), q_type))
      << "all input tensors must have the same dtype";
  CHECK_INPUT_TYPE(page_table, dl_int32);
  CHECK_INPUT_TYPE(kv_len, dl_int32);

  const int batch        = static_cast<int>(kv_len.size(0));
  const int seqlen_q     = static_cast<int>(q_nope.size(1));
  const int head_num_q   = static_cast<int>(q_nope.size(-2));
  const int head_dim_ckv = static_cast<int>(q_nope.size(-1));
  const int head_dim_kpe = static_cast<int>(q_pe.size(-1));
  const int page_size    = static_cast<int>(ckv.size(1));
  const int page_count   = static_cast<int>(ckv.size(0));

  expect_shape(q_nope, {batch, seqlen_q, head_num_q, head_dim_ckv}, "q_nope");
  expect_shape(q_pe, {batch, seqlen_q, head_num_q, head_dim_kpe}, "q_pe");
  if (ckv.dim() == 3) {
    expect_shape(ckv, {page_count, page_size, head_dim_ckv}, "ckv");
    expect_shape(kpe, {page_count, page_size, head_dim_kpe}, "kpe");
  } else {
    expect_shape(ckv, {page_count, page_size, 1, head_dim_ckv}, "ckv");
    expect_shape(kpe, {page_count, page_size, 1, head_dim_kpe}, "kpe");
  }
  expect_shape(kv_len, {batch}, "kv_len");
  TVM_FFI_ICHECK_EQ(page_table.size(0), batch) << "page_table size(0) must equal batch size";
  TVM_FFI_ICHECK_EQ(page_size, 64) << "page size only supports 64";
  TVM_FFI_ICHECK_EQ(head_dim_ckv, 512) << "head dim ckv only supports 512";
  TVM_FFI_ICHECK_EQ(head_dim_kpe, 64) << "head dim kpe only supports 64";

  TVM_FFI_ICHECK_EQ(out.stride(-1), 1) << "out must have contiguous last dimension";
  TVM_FFI_ICHECK(dtype_equal(out.dtype(), q_type)) << "out must have the same dtype as inputs";
  TVM_FFI_ICHECK(dtype_equal(lse.dtype(), dl_float32)) << "lse must have dtype float32";
  expect_shape(out, {batch, seqlen_q, head_num_q, head_dim_ckv}, "out");
  expect_shape(lse, {batch, seqlen_q, head_num_q}, "lse");
  TVM_FFI_ICHECK_EQ(out.stride(1), head_num_q * out.stride(2))
      << "non-contiguous out layout is not representable for mla; materialize out first";
  TVM_FFI_ICHECK_EQ(lse.stride(1), head_num_q * lse.stride(2))
      << "non-contiguous lse layout is not representable for mla; materialize lse first";

  ffi::MUSADeviceGuard device_guard(q_nope.device().device_id);

#define LAUNCH_KERNEL(QKV_TYPE, OUT_TYPE, IS_CAUSAL)                                               \
  [&] {                                                                                            \
    dispatch_mla_kernel<QKV_TYPE, OUT_TYPE, IS_CAUSAL>(q_nope,                                     \
                                                       q_pe,                                       \
                                                       ckv,                                        \
                                                       kpe,                                        \
                                                       out,                                        \
                                                       lse,                                        \
                                                       page_table,                                 \
                                                       kv_len,                                     \
                                                       static_cast<float>(sm_scale),               \
                                                       batch,                                      \
                                                       page_count,                                 \
                                                       page_size,                                  \
                                                       head_num_q,                                 \
                                                       head_dim_ckv,                               \
                                                       head_dim_kpe,                               \
                                                       seqlen_q,                                   \
                                                       q_nope.stride(0),                           \
                                                       q_nope.stride(2),                           \
                                                       q_pe.stride(0),                             \
                                                       q_pe.stride(2),                             \
                                                       ckv.stride(0),                              \
                                                       ckv.stride(1),                              \
                                                       kpe.stride(0),                              \
                                                       kpe.stride(1),                              \
                                                       static_cast<int32_t>(page_table.stride(0)), \
                                                       out.stride(0),                              \
                                                       out.stride(2),                              \
                                                       static_cast<int32_t>(lse.stride(0)),        \
                                                       static_cast<int32_t>(lse.stride(2)));       \
  }()

  if (dtype_equal(q_type, dl_float16)) {
    if (is_causal) {
      LAUNCH_KERNEL(mutlass::half_t, mutlass::half_t, true);
    } else {
      LAUNCH_KERNEL(mutlass::half_t, mutlass::half_t, false);
    }
  } else if (dtype_equal(q_type, dl_bfloat16)) {
    if (is_causal) {
      LAUNCH_KERNEL(mutlass::bfloat16_t, mutlass::bfloat16_t, true);
    } else {
      LAUNCH_KERNEL(mutlass::bfloat16_t, mutlass::bfloat16_t, false);
    }
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported MLA input dtype";
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

void mla_with_kvcache(ffi::TensorView                q_latent,
                      ffi::TensorView                q_rope,
                      ffi::TensorView                ckv,
                      ffi::TensorView                kpe,
                      ffi::TensorView                seqlens_k,
                      ffi::Optional<ffi::TensorView> cu_seqlens_q,
                      ffi::Optional<int64_t>         max_seqlen_q,
                      ffi::TensorView                block_table,
                      ffi::TensorView                tile_scheduler_metadata,
                      ffi::TensorView                num_splits,
                      ffi::TensorView                out,
                      ffi::TensorView                softmax_lse,
                      double                         softmax_scale,
                      bool                           is_causal) {
  check_mp31(q_latent.device(), "mla_with_kvcache");

  CHECK_MUSA(q_latent);
  CHECK_MUSA(q_rope);
  CHECK_MUSA(ckv);
  CHECK_MUSA(kpe);
  CHECK_MUSA(seqlens_k);
  CHECK_MUSA(block_table);
  CHECK_MUSA(tile_scheduler_metadata);
  CHECK_MUSA(num_splits);
  CHECK_DEVICE(q_latent, q_rope);
  CHECK_DEVICE(q_latent, ckv);
  CHECK_DEVICE(q_latent, kpe);
  CHECK_DEVICE(q_latent, seqlens_k);
  CHECK_DEVICE(q_latent, block_table);
  CHECK_DEVICE(q_latent, tile_scheduler_metadata);
  CHECK_DEVICE(q_latent, num_splits);
  TVM_FFI_ICHECK_EQ(q_latent.stride(-1), 1) << "q_latent must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(q_rope.stride(-1), 1) << "q_rope must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(ckv.stride(-1), 1) << "ckv must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(kpe.stride(-1), 1) << "kpe must have contiguous last dimension";

  const DLDataType q_type = q_latent.dtype();
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(q_type)) << "mla only supports fp16 and bf16";
  TVM_FFI_ICHECK(dtype_equal(q_rope.dtype(), q_type) && dtype_equal(ckv.dtype(), q_type) &&
                 dtype_equal(kpe.dtype(), q_type))
      << "input tensor dtypes must match";

  CHECK_CONTIGUOUS(seqlens_k);
  CHECK_CONTIGUOUS(tile_scheduler_metadata);
  CHECK_CONTIGUOUS(num_splits);
  CHECK_INPUT_TYPE(seqlens_k, dl_int32);
  CHECK_INPUT_TYPE(block_table, dl_int32);
  CHECK_INPUT_TYPE(tile_scheduler_metadata, dl_int32);
  CHECK_INPUT_TYPE(num_splits, dl_int32);
  TVM_FFI_ICHECK_EQ(block_table.stride(-1), 1) << "block_table must have contiguous last dimension";

  const bool is_varlen_q = cu_seqlens_q.has_value();
  if (is_varlen_q) {
    CHECK_HAS_VALUE_WITH_MSG(max_seqlen_q, "max_seqlen_q must be provided when cu_seqlens_q is provided");
    CHECK_MUSA(cu_seqlens_q.value());
    CHECK_CONTIGUOUS(cu_seqlens_q.value());
    CHECK_INPUT_TYPE(cu_seqlens_q.value(), dl_int32);
    CHECK_DEVICE(q_latent, cu_seqlens_q.value());
  }

  const int batch_size =
      !is_varlen_q ? static_cast<int>(q_latent.size(0)) : static_cast<int>(cu_seqlens_q.value().size(0) - 1);
  const int seqlen_q = !is_varlen_q ? static_cast<int>(q_latent.size(1)) : static_cast<int>(max_seqlen_q.value());
  const int total_q =
      !is_varlen_q ? static_cast<int>(q_latent.size(0) * q_latent.size(1)) : static_cast<int>(q_latent.size(0));
  const int     num_heads_q     = static_cast<int>(q_latent.size(-2));
  const int     head_dim_latent = static_cast<int>(q_latent.size(-1));
  const int     head_dim_rope   = static_cast<int>(q_rope.size(-1));
  const int     num_blocks      = static_cast<int>(ckv.size(0));
  const int     page_size       = static_cast<int>(ckv.size(1));
  constexpr int num_heads_k     = 1;

  if (!is_varlen_q) {
    expect_shape(q_latent, {batch_size, seqlen_q, num_heads_q, head_dim_latent}, "q_latent");
    expect_shape(q_rope, {batch_size, seqlen_q, num_heads_q, head_dim_rope}, "q_rope");
  } else {
    expect_shape(q_latent, {total_q, num_heads_q, head_dim_latent}, "q_latent");
    expect_shape(q_rope, {total_q, num_heads_q, head_dim_rope}, "q_rope");
    expect_shape(cu_seqlens_q.value(), {batch_size + 1}, "cu_seqlens_q");
  }

  if (ckv.dim() == 3) {
    expect_shape(ckv, {num_blocks, page_size, head_dim_latent}, "ckv");
    expect_shape(kpe, {num_blocks, page_size, head_dim_rope}, "kpe");
  } else {
    expect_shape(ckv, {num_blocks, page_size, num_heads_k, head_dim_latent}, "ckv");
    expect_shape(kpe, {num_blocks, page_size, num_heads_k, head_dim_rope}, "kpe");
  }
  expect_shape(seqlens_k, {batch_size}, "seqlens_k");
  expect_shape(block_table, {batch_size, block_table.size(1)}, "block_table");
  TVM_FFI_ICHECK_EQ(tile_scheduler_metadata.ndim(), 2) << "tile_scheduler_metadata must be 2D";
  TVM_FFI_ICHECK_EQ(tile_scheduler_metadata.size(1), mate::flash_mla::TileSchedulerMetaDataSize)
      << "tile_scheduler_metadata has unexpected second dimension";
  expect_shape(num_splits, {batch_size + 1}, "num_splits");

  TVM_FFI_ICHECK_EQ(page_size, 64) << "page size only supports 64";
  TVM_FFI_ICHECK_EQ(head_dim_latent, 512) << "head dim latent only supports 512";
  TVM_FFI_ICHECK_EQ(head_dim_rope, 64) << "head dim rope only supports 64";

  ffi::MUSADeviceGuard device_guard(q_latent.device().device_id);
  musaDeviceProp       dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, q_latent.device().device_id));

  const int heads_ratio  = num_heads_q / num_heads_k;
  const int q_seq_per_hk = seqlen_q * heads_ratio;
  TVM_FFI_ICHECK_EQ(num_heads_q % num_heads_k, 0) << "num_heads_q must be divisible by num_heads_k";

  std::optional<StridedTensorView> q_latent_flattened;
  std::optional<StridedTensorView> q_rope_flattened;
  ffi::TensorView                  q_latent_work = q_latent;
  ffi::TensorView                  q_rope_work   = q_rope;
  if (!is_varlen_q) {
    TVM_FFI_ICHECK_EQ(q_latent.stride(1), num_heads_q * q_latent.stride(2))
        << "non-contiguous q_latent layout is not representable for mla_with_kvcache; materialize q first";
    TVM_FFI_ICHECK_EQ(q_rope.stride(1), num_heads_q * q_rope.stride(2))
        << "non-contiguous q_rope layout is not representable for mla_with_kvcache; materialize q first";
    q_latent_flattened.emplace(
        q_latent,
        ffi::Shape{batch_size, q_seq_per_hk, num_heads_k, head_dim_latent},
        ffi::Shape{q_latent.stride(0), q_latent.stride(2), q_latent.stride(2), q_latent.stride(3)});
    q_rope_flattened.emplace(q_rope,
                             ffi::Shape{batch_size, q_seq_per_hk, num_heads_k, head_dim_rope},
                             ffi::Shape{q_rope.stride(0), q_rope.stride(2), q_rope.stride(2), q_rope.stride(3)});
    q_latent_work = q_latent_flattened->view;
    q_rope_work   = q_rope_flattened->view;
  }

  CHECK_MUSA(out);
  CHECK_MUSA(softmax_lse);
  CHECK_DEVICE(out, q_latent);
  CHECK_DEVICE(softmax_lse, q_latent);
  TVM_FFI_ICHECK_EQ(out.stride(-1), 1) << "out must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(softmax_lse.stride(-1), 1) << "softmax_lse must have contiguous last dimension";
  TVM_FFI_ICHECK(dtype_equal(out.dtype(), q_type)) << "out must have the same dtype as q_latent";
  TVM_FFI_ICHECK(dtype_equal(softmax_lse.dtype(), dl_float32)) << "softmax_lse must have dtype float32";

  std::optional<StridedTensorView> out_view;
  std::optional<StridedTensorView> softmax_lse_view;
  ffi::TensorView                  out_work         = out;
  ffi::TensorView                  softmax_lse_work = softmax_lse;
  if (!is_varlen_q) {
    expect_shape(out, {batch_size, seqlen_q, num_heads_q, head_dim_latent}, "out");
    expect_shape(softmax_lse, {batch_size, seqlen_q, num_heads_q}, "softmax_lse");
    TVM_FFI_ICHECK_EQ(out.stride(1), num_heads_q * out.stride(2))
        << "non-contiguous out layout is not representable for mla_with_kvcache; materialize out first";
    TVM_FFI_ICHECK_EQ(softmax_lse.stride(1), num_heads_q * softmax_lse.stride(2))
        << "non-contiguous softmax_lse layout is not representable for mla_with_kvcache; materialize softmax_lse first";
    out_view.emplace(out,
                     ffi::Shape{batch_size, q_seq_per_hk, num_heads_k, head_dim_latent},
                     ffi::Shape{out.stride(0), out.stride(2), out.stride(2), out.stride(3)});
    softmax_lse_view.emplace(softmax_lse,
                             ffi::Shape{batch_size, num_heads_k, q_seq_per_hk},
                             ffi::Shape{softmax_lse.stride(0), softmax_lse.stride(0), 1});
    out_work         = out_view->view;
    softmax_lse_work = softmax_lse_view->view;
  } else {
    expect_shape(out, {total_q, num_heads_q, head_dim_latent}, "out");
    expect_shape(softmax_lse, {num_heads_q, total_q}, "softmax_lse");
    out_view.emplace(out,
                     ffi::Shape{total_q, heads_ratio, num_heads_k, head_dim_latent},
                     ffi::Shape{out.stride(0), out.stride(1), out.stride(1), out.stride(2)});
    softmax_lse_view.emplace(
        softmax_lse,
        ffi::Shape{num_heads_k, heads_ratio, total_q},
        ffi::Shape{softmax_lse.size(0) * softmax_lse.stride(0), softmax_lse.stride(0), softmax_lse.stride(1)});
    out_work         = out_view->view;
    softmax_lse_work = softmax_lse_view->view;
  }

  ffi::Tensor softmax_lse_accum =
      alloc_tensor(ffi::Shape{batch_size + tile_scheduler_metadata.size(0), num_heads_k, q_seq_per_hk},
                   dl_float32,
                   q_latent.device());
  ffi::Tensor out_accum =
      alloc_tensor(ffi::Shape{batch_size + tile_scheduler_metadata.size(0), num_heads_k, q_seq_per_hk, head_dim_latent},
                   dl_float32,
                   q_latent.device());

  auto stride_q_latent =
      !is_varlen_q
          ? make_stride(q_latent_work.stride(1), _1{}, make_stride(_0{}, static_cast<int>(q_latent_work.stride(0))))
          : make_stride(q_latent_work.stride(1), _1{}, make_stride(_0{}, 0));
  auto stride_q_rope =
      !is_varlen_q
          ? make_stride(q_rope_work.stride(1), _1{}, make_stride(_0{}, static_cast<int>(q_rope_work.stride(0))))
          : make_stride(q_rope_work.stride(1), _1{}, make_stride(_0{}, 0));
  auto stride_ckv = make_stride(ckv.stride(1), _1{}, make_stride(_0{}, ckv.stride(0)));
  auto stride_kpe = make_stride(kpe.stride(1), _1{}, make_stride(_0{}, kpe.stride(0)));
  auto stride_pt  = make_stride(static_cast<int>(block_table.stride(0)), _1{});

  auto stride_out = !is_varlen_q
                        ? make_stride(out_work.stride(2), _1{}, make_stride(_0{}, static_cast<int>(out_work.stride(0))))
                        : make_stride(out_work.stride(2), _1{}, make_stride(_0{}, 0));
  auto stride_lse = !is_varlen_q ? make_stride(_1{},
                                               make_stride(static_cast<int>(softmax_lse_work.stride(1)),
                                                           static_cast<int>(softmax_lse_work.stride(0))))
                                 : make_stride(_1{}, make_stride(static_cast<int>(softmax_lse_work.stride(0)), 0));

  auto run_mla = [&](auto element_type, auto use_trival_scheduler, auto causal, auto varlen_q) {
    static constexpr bool UseTrivalScheduler = decltype(use_trival_scheduler)::value;
    static constexpr bool Causal             = decltype(causal)::value;
    static constexpr bool VarlenQ            = decltype(varlen_q)::value;

    using StrideQ   = decltype(stride_q_latent);
    using StrideC   = decltype(stride_ckv);
    using StrideO   = decltype(stride_out);
    using StrideLse = decltype(stride_lse);
    using Element   = decltype(element_type);
    using TileShape = Shape<_128, _64, Shape<_512, _64>>;

    using Fusion             = conditional_t<Causal,
                                             mutlass::fmha::collective::CausalFusion<false, true>,
                                             mutlass::fmha::collective::DefaultFusion>;
    using CollectiveMainloop = mate::flash_mla::
        FmhaMlaMainloopTmeWarpSpecialized<Element, float, TileShape, StrideQ, StrideC, Fusion, VarlenQ>;
    using CollectiveEpilogue =
        mate::flash_mla::MlaFwdEpilogue<Element, float, Shape<_32, _512>, StrideO, StrideLse, VarlenQ>;
    using TileScheduler = mate::flash_mla::FlashMlaTileScheduler<UseTrivalScheduler>;
    using MlaKernel =
        mate::flash_mla::MlaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

    typename MlaKernel::ProblemShape problem_shape;
    auto                             problem_size = make_shape(seqlen_q,
                                   page_size,
                                   make_shape(head_dim_latent, head_dim_rope),
                                   num_heads_q,
                                   num_blocks,
                                   batch_size,
                                   total_q,
                                   seqlen_q);

    get<0>(problem_shape) = mutlass::fmha::collective::VariableLength{
        !is_varlen_q ? static_cast<int>(get<0>(problem_size)) : total_q,
        static_cast<int>(seqlen_q),
        !is_varlen_q ? nullptr : static_cast<int32_t*>(cu_seqlens_q.value().data_ptr())};
    get<1>(problem_shape) = get<1>(problem_size);
    get<2>(problem_shape) = get<2>(problem_size);
    get<3>(problem_shape) = get<3>(problem_size);
    get<4>(problem_shape) = get<4>(problem_size);
    get<5>(problem_shape) = get<5>(problem_size);
    get<6>(problem_shape) = get<6>(problem_size);
    get<7>(problem_shape) = get<7>(problem_size);

    typename Fusion::Arguments fusion;
    mutlass::FastDivmod        divmod_hr(heads_ratio);
    fusion.fast_divmod_hr = divmod_hr;

    typename MlaKernel::Arguments arguments{
        problem_shape,
        {static_cast<Element*>(q_latent_work.data_ptr()),
         stride_q_latent,
         static_cast<Element*>(q_rope_work.data_ptr()),
         stride_q_rope,
         static_cast<Element*>(ckv.data_ptr()),
         stride_ckv,
         static_cast<Element*>(kpe.data_ptr()),
         stride_kpe,
         static_cast<int32_t*>(block_table.data_ptr()),
         stride_pt,
         static_cast<int32_t*>(seqlens_k.data_ptr()),
         static_cast<float>(softmax_scale),
         fusion,
         VarlenQ ? static_cast<int64_t*>(cu_seqlens_q.value().data_ptr()) : nullptr},
        {static_cast<Element*>(out_work.data_ptr()),
         stride_out,
         static_cast<float*>(softmax_lse_work.data_ptr()),
         stride_lse,
         static_cast<float*>(out_accum.data_ptr()),
         static_cast<float*>(softmax_lse_accum.data_ptr()),
         q_seq_per_hk},
        {static_cast<int>(tile_scheduler_metadata.size(0)),
         static_cast<int32_t*>(tile_scheduler_metadata.data_ptr()),
         static_cast<int32_t*>(num_splits.data_ptr())}};

    typename MlaKernel::Params params   = MlaKernel::to_underlying_arguments(arguments);
    dim3                       grid_dim = TileScheduler::get_grid_shape(params.tile_scheduler);
    musaStream_t               stream   = get_stream(q_latent.device());

    mutlass::device_kernel<MlaKernel>
        <<<grid_dim, MlaKernel::MaxThreadsPerBlock, MlaKernel::SharedStorageSize, stream>>>(params);

    if constexpr (!UseTrivalScheduler) {
      mate::flash_mla::MlaCombineParams combine_params{};
      combine_params.batch_size           = batch_size;
      combine_params.num_mp_parts         = static_cast<int>(tile_scheduler_metadata.size(0));
      combine_params.o_ptr                = out_work.data_ptr();
      combine_params.softmax_lse_ptr      = softmax_lse_work.data_ptr();
      combine_params.oaccum_ptr           = out_accum.data_ptr();
      combine_params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
      combine_params.num_splits_ptr       = static_cast<int32_t*>(num_splits.data_ptr());
      combine_params.seqlens_q_ptr        = !VarlenQ ? nullptr : static_cast<int32_t*>(cu_seqlens_q.value().data_ptr());
      combine_params.max_q_seq_per_hk     = q_seq_per_hk;
      combine_params.total_q              = total_q;
      combine_params.h_r                  = heads_ratio;
      combine_params.h_k                  = 1;
      combine_params.o_batch_stride       = static_cast<int>(out_work.stride(0));
      combine_params.o_row_stride         = static_cast<int>(out_work.stride(1));
      combine_params.o_head_stride        = static_cast<int>(out_work.stride(2));

      run_mla_combine_kernel<Element, VarlenQ>(combine_params, stream);
    }
  };

  constexpr bool need_load_balance = true;
  if (need_load_balance) {
    if (is_causal) {
      if (is_varlen_q) {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::true_type{}, mute::true_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::true_type{}, mute::true_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      } else {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::true_type{}, mute::false_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::true_type{}, mute::false_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      }
    } else {
      if (is_varlen_q) {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::false_type{}, mute::true_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::false_type{}, mute::true_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      } else {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::false_type{}, mute::false_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::false_type{}, mute::false_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      }
    }
  } else {
    if (is_causal) {
      if (is_varlen_q) {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::true_type{}, mute::true_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::true_type{}, mute::true_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      } else {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::true_type{}, mute::false_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::true_type{}, mute::false_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      }
    } else {
      if (is_varlen_q) {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::false_type{}, mute::true_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::false_type{}, mute::true_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      } else {
        if (dtype_equal(q_type, dl_float16)) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::false_type{}, mute::false_type{});
        } else if (dtype_equal(q_type, dl_bfloat16)) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::false_type{}, mute::false_type{});
        } else {
          TVM_FFI_ICHECK(false) << "Unsupported element dtype";
        }
      }
    }
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

MetadataResult get_mla_decoding_metadata_impl(ffi::Optional<ffi::TensorView> seqlens_k,
                                              int64_t                        num_q_tokens_per_head_k,
                                              int64_t                        h_k,
                                              ffi::Optional<int64_t>         h_q,
                                              bool                           is_fp8_kvcache,
                                              ffi::Optional<int64_t>         topk,
                                              ffi::Optional<ffi::TensorView> tile_scheduler_metadata_opt,
                                              ffi::Optional<ffi::TensorView> num_splits_opt,
                                              ffi::Optional<ffi::TensorView> q,
                                              ffi::Optional<int64_t>         bs,
                                              ffi::Optional<ffi::TensorView> topk_length,
                                              ffi::Optional<ffi::TensorView> extra_topk_length,
                                              ffi::Optional<int64_t>         extra_topk) {
  FFI_CHECK(seqlens_k.has_value() || q.has_value(), "seqlens_k or q must be provided");

  auto                 tensor_device = seqlens_k.has_value() ? seqlens_k.value().device() : q.value().device();
  auto                 device_id     = tensor_device.device_id;
  ffi::MUSADeviceGuard device_guard(device_id);

  musaDeviceProp dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, device_id));
  const int mp_count = dprops.multiProcessorCount;
  Arch      arch     = {dprops.major, dprops.minor};
  TVM_FFI_ICHECK(arch.is_mp31()) << "Only MP31 is supported";

  const int batch_size = static_cast<int>(
      bs.has_value() ? bs.value() : (seqlens_k.has_value() ? seqlens_k.value().size(0) : q.value().size(0)));
  const bool      is_sparse_attn    = topk.has_value();
  ffi::TensorView device_cmp_tensor = seqlens_k.has_value() ? seqlens_k.value() : q.value();
  if (extra_topk.has_value()) {
    FFI_CHECK(is_sparse_attn, "extra_topk may only be provided when topk is provided");
    FFI_CHECK(extra_topk.value() >= 0, "extra_topk must be non-negative");
  }

  if (is_sparse_attn) {
    CHECK_HAS_VALUE_WITH_MSG(h_q, "num_heads_q must be provided when topk is provided");
  } else {
    CHECK_HAS_VALUE_WITH_MSG(seqlens_k, "seqlens_k must be provided when topk is not provided");
    CHECK_MUSA(seqlens_k.value());
    CHECK_CONTIGUOUS(seqlens_k.value());
    CHECK_INPUT_TYPE(seqlens_k.value(), dl_int32);
  }
  if (topk_length.has_value()) {
    CHECK_MUSA(topk_length.value());
    CHECK_CONTIGUOUS(topk_length.value());
    CHECK_INPUT_TYPE(topk_length.value(), dl_int32);
    CHECK_DEVICE(topk_length.value(), device_cmp_tensor);
    expect_shape(topk_length.value(), {batch_size}, "topk_length");
  }
  if (extra_topk_length.has_value()) {
    CHECK_MUSA(extra_topk_length.value());
    CHECK_CONTIGUOUS(extra_topk_length.value());
    CHECK_INPUT_TYPE(extra_topk_length.value(), dl_int32);
    CHECK_DEVICE(extra_topk_length.value(), device_cmp_tensor);
    expect_shape(extra_topk_length.value(), {batch_size}, "extra_topk_length");
  }

  const int            tile_m         = is_sparse_attn ? 64 : 128;
  DecodingAttnImplMeta attn_impl_meta = get_attn_impl_meta(arch,
                                                           mp_count,
                                                           static_cast<int>(num_q_tokens_per_head_k),
                                                           static_cast<int>(h_k),
                                                           h_q,
                                                           is_fp8_kvcache,
                                                           is_sparse_attn,
                                                           tile_m);

  ffi::Tensor     tile_scheduler_metadata_storage;
  ffi::Tensor     num_splits_storage;
  ffi::TensorView tile_scheduler_metadata =
      tile_scheduler_metadata_opt.has_value()
          ? tile_scheduler_metadata_opt.value()
          : ffi::TensorView(tile_scheduler_metadata_storage = alloc_tensor(
                                ffi::Shape{attn_impl_meta.num_mp_parts, mate::flash_mla::TileSchedulerMetaDataSize},
                                dl_int32,
                                tensor_device));
  ffi::TensorView num_splits =
      num_splits_opt.has_value()
          ? num_splits_opt.value()
          : ffi::TensorView(num_splits_storage = alloc_tensor(ffi::Shape{batch_size + 1}, dl_int32, tensor_device));

  CHECK_DEVICE(tile_scheduler_metadata, device_cmp_tensor);
  CHECK_DEVICE(num_splits, device_cmp_tensor);
  CHECK_CONTIGUOUS(tile_scheduler_metadata);
  CHECK_CONTIGUOUS(num_splits);
  CHECK_INPUT_TYPE(tile_scheduler_metadata, dl_int32);
  CHECK_INPUT_TYPE(num_splits, dl_int32);
  expect_shape(tile_scheduler_metadata,
               {attn_impl_meta.num_mp_parts, mate::flash_mla::TileSchedulerMetaDataSize},
               "tile_scheduler_metadata");
  expect_shape(num_splits, {batch_size + 1}, "num_splits");

  mate::flash_mla::GetDecodingMetadataParams params = {};
  params.seqlens_k_ptr = seqlens_k.has_value() ? static_cast<int32_t*>(seqlens_k.value().data_ptr()) : nullptr;
  params.tile_scheduler_metadata_ptr = static_cast<int32_t*>(tile_scheduler_metadata.data_ptr());
  params.num_splits_ptr              = static_cast<int32_t*>(num_splits.data_ptr());
  params.topk_length_ptr = topk_length.has_value() ? static_cast<int32_t*>(topk_length.value().data_ptr()) : nullptr;
  params.extra_topk_length_ptr =
      extra_topk_length.has_value() ? static_cast<int32_t*>(extra_topk_length.value().data_ptr()) : nullptr;
  params.batch_size                = batch_size;
  params.block_size_n              = attn_impl_meta.k_block_size;
  params.fixed_overhead_num_blocks = attn_impl_meta.fixed_overhead_num_blocks;
  params.num_mp_parts              = attn_impl_meta.num_mp_parts;
  params.topk                      = is_sparse_attn ? static_cast<int>(topk.value()) : -1;
  params.extra_topk                = extra_topk.has_value() ? static_cast<int>(extra_topk.value()) : -1;
  run_get_mla_metadata_kernel(params,
                              get_stream(seqlens_k.has_value() ? seqlens_k.value().device() : q.value().device()));

  return {tile_scheduler_metadata_storage, num_splits_storage, tile_scheduler_metadata, num_splits};
}

ffi::Array<ffi::Any> get_mla_decoding_metadata(ffi::Optional<ffi::TensorView> seqlens_k,
                                               int64_t                        num_q_tokens_per_head_k,
                                               int64_t                        h_k,
                                               ffi::Optional<int64_t>         h_q,
                                               bool                           is_fp8_kvcache,
                                               ffi::Optional<int64_t>         topk,
                                               ffi::Optional<ffi::Tensor>     tile_scheduler_metadata,
                                               ffi::Optional<ffi::Tensor>     num_splits,
                                               ffi::Optional<ffi::TensorView> q,
                                               ffi::Optional<int64_t>         bs,
                                               ffi::Optional<ffi::TensorView> topk_length,
                                               ffi::Optional<ffi::TensorView> extra_topk_length,
                                               ffi::Optional<int64_t>         extra_topk) {
  TVM_FFI_ICHECK(tile_scheduler_metadata.has_value() == num_splits.has_value())
      << "tile_scheduler_metadata and num_splits must be both provided or both omitted";

  ffi::Optional<ffi::TensorView> tile_scheduler_metadata_view;
  ffi::Optional<ffi::TensorView> num_splits_view;

  if (tile_scheduler_metadata.has_value()) {
    const ffi::Tensor& tile_scheduler_metadata_tensor = tile_scheduler_metadata.value();
    const ffi::Tensor& num_splits_tensor              = num_splits.value();
    tile_scheduler_metadata_view                      = ffi::TensorView(tile_scheduler_metadata_tensor);
    num_splits_view                                   = ffi::TensorView(num_splits_tensor);
  }

  auto result = get_mla_decoding_metadata_impl(seqlens_k,
                                               num_q_tokens_per_head_k,
                                               h_k,
                                               h_q,
                                               is_fp8_kvcache,
                                               topk,
                                               tile_scheduler_metadata_view,
                                               num_splits_view,
                                               q,
                                               bs,
                                               topk_length,
                                               extra_topk_length,
                                               extra_topk);

  ffi::Tensor tile_scheduler_metadata_result =
      tile_scheduler_metadata.has_value() ? tile_scheduler_metadata.value() : result.tile_scheduler_metadata_storage;
  ffi::Tensor num_splits_result = num_splits.has_value() ? num_splits.value() : result.num_splits_storage;

  return ffi::Array<ffi::Any>{ffi::Any(tile_scheduler_metadata_result), ffi::Any(num_splits_result)};
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla, mla);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_with_kvcache, mla_with_kvcache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mla_decoding_metadata, get_mla_decoding_metadata);
