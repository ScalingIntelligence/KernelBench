#include <vector>
#include <cstdint>
#include <stdexcept>

#include <musa.h>
#include <mutlass/device_kernel.h>

#include "glog/logging.h"  // For VLOG()
#include "paddle/common/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/extension.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/include/tensor_utils.h"

#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_collective_tme_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_paged_collective_tme_warpspecialized.hpp"
#include "fmha_options.hpp"
#include "kernel/fmha_kernel_tme_warpspecialzed.hpp"
#include "kernel/fmha_paged_kernel_tme_warpspecialzed.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "mutlass/bfloat16.h"
#include "mutlass/half.h"

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

struct TileConfig {
  int kCTA_Q;
  int kCTA_K;
  int kHeadSizeQK;
  int kHeadSizeV;
};

template <int HEADDIM_QK, int HEADDIM_V, bool PAGED_KV>
constexpr TileConfig get_tile_config() {
  if constexpr (PAGED_KV) {
    if constexpr (HEADDIM_QK == 128 && HEADDIM_V == 128) {
      return {256, 64, 128, 128};
    }
  } else {
    if constexpr (HEADDIM_QK == 128 && HEADDIM_V == 128) {
      return {256, 128, 128, 128};
    } else if constexpr (HEADDIM_QK == 192 && HEADDIM_V == 128) {
      return {256, 64, 192, 128};
    }
  }
  return {0, 0, 0, 0};
}

#define HEADDIM_SWITCH(HEADDIM_QK, HEADDIM_V, ...)      \
  [&] {                                                 \
    if (HEADDIM_QK == 128 && HEADDIM_V == 128) {        \
      constexpr static int kHeadSizeQK = 128;           \
      constexpr static int kHeadSizeV  = 128;           \
      return __VA_ARGS__();                             \
    } else if (HEADDIM_QK == 192 && HEADDIM_V == 128) { \
      constexpr static int kHeadSizeQK = 192;           \
      constexpr static int kHeadSizeV  = 128;           \
      return __VA_ARGS__();                             \
    }                                                   \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = mutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = mutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()


struct Qkv_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;

  // The number of heads.
  int h, h_k;
};

struct Flash_fwd_params : public Qkv_params {
  using index_t = int64_t;

  // The O matrix (output).
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  // For FP8 scaling
  float* __restrict__ q_descale_ptr;
  float* __restrict__ k_descale_ptr;
  float* __restrict__ v_descale_ptr;
  index_t q_descale_batch_stride;
  index_t q_descale_head_stride;
  index_t k_descale_batch_stride;
  index_t k_descale_head_stride;
  index_t v_descale_batch_stride;
  index_t v_descale_head_stride;

  // The dimensions.
  int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
  int total_q, total_k, total_knew;
  int b_k;             // When having KV cache and with cache_batch_idx, K & V might have larger batch size
                       // than Q
  int dv, dv_rounded;  // For the case where V headdim is different from Q/K headdim

  // The scaling factors for the kernel.
  float scale_softmax;
  float softcap;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;
  int* __restrict__ cu_seqlens_knew;
  int* __restrict__ leftpad_k;

  // If provided, the actual length of each q/k sequence.
  int* __restrict__ seqused_q;
  int* __restrict__ seqused_k;

  // The stride between rows of Oaccum.
  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;

  // The stride between rows of LSEaccum.
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;

  index_t lse_batch_stride;
  index_t lse_head_stride;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;

  void* __restrict__ qv_ptr;
  index_t qv_batch_stride;
  index_t qv_row_stride;
  index_t qv_head_stride;

  // The cos and sin matrices for rotary embedding.
  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;
  int* __restrict__ seqlens_rotary;

  // The indices to index into the KV cache.
  int* __restrict__ kv_batch_idx;

  // Paged KV cache
  int* __restrict__ page_table;
  index_t page_table_batch_stride;
  int     page_size;
  int     num_pages;
  bool    pagedkv_tma;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint32_t p_dropout_in_uint;
  // uint16_t p_dropout_in_uint16_t;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;

  // Local window size
  int window_size_left, window_size_right;
  int attention_chunk;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;

  bool is_rotary_interleaved;

  int  num_splits;  // For split-KV version
  bool pack_gqa;

  int* __restrict__ tile_count_semaphore;
  int* __restrict__ num_m_blocks_ptr;
  // int * __restrict__ num_n_blocks_ptr;
  int* __restrict__ num_splits_dynamic_ptr;
  int* __restrict__ varlen_batch_idx_ptr;  // virtual -> actual
  int* __restrict__ num_nheads_in_l2_ptr;
  bool skip_scheduler_metadata_computation;
  bool varlen_sort_batches;
  int  tile_count_semaphore_offset;
  bool head_swizzle;
  bool prepare_varlen_pdl;

  int  arch;
  int  num_sm;
  bool is_varlen_q;
  bool is_varlen_k;
};

template <typename ShapeLike>
inline int64_t getEle(const ShapeLike& x, int idx) {
  const int64_t n = static_cast<int64_t>(x.size());
  if (idx < 0) {
    idx += n;
  }
  if (idx < 0 || idx >= n) {
    throw std::out_of_range("Python-style index out of range");
  }
  return x[idx];
}

namespace mute {

template <typename KernelT>
static void PrintLaunchDebug(const phi::CustomContext* dev_ctx,
                             const dim3& grid,
                             int block_threads,
                             int dyn_shmem,
                             const typename KernelT::Arguments& arguments,
                             const typename KernelT::Params& kernel_params,
                             const Flash_fwd_params& p) {
  int dev = 0;
  musaGetDevice(&dev);

  int max_threads = 0;
  musaDeviceGetAttribute(&max_threads, musaDevAttrMaxThreadsPerBlock, dev);

  int max_shmem = 0;
  musaDeviceGetAttribute(&max_shmem, musaDevAttrMaxSharedMemoryPerBlock, dev);

  int optin_shmem = 0;
  musaDeviceGetAttribute(&optin_shmem, musaDevAttrMaxSharedMemoryPerBlockOptin, dev);

  musaFuncAttributes fa{};
  auto fa_err = musaFuncGetAttributes(&fa, (const void*)mutlass::device_kernel<KernelT>);

  VLOG(1) << "[fmha][launch] dev=" << dev
          << " grid=(" << grid.x << "," << grid.y << "," << grid.z << ")"
          << " block_threads=" << block_threads
          << " dyn_shmem=" << dyn_shmem
          << " stream=" << (const void*)reinterpret_cast<musaStream_t>(dev_ctx->stream());

  VLOG(1) << "[fmha][device] maxThreadsPerBlock=" << max_threads
          << " maxShmemPerBlock=" << max_shmem
          << " optinShmem=" << optin_shmem;

  VLOG(1) << "[fmha][kernel] Kernel::MaxThreadsPerBlock=" << KernelT::MaxThreadsPerBlock
          << " Kernel::SharedStorageSize=" << KernelT::SharedStorageSize;

  VLOG(1) << "[fmha][funcattr] err=" << (int)fa_err
          << " (" << musaGetErrorString(fa_err) << ")"
          << " static_shared=" << fa.sharedSizeBytes
          << " maxThreads=" << fa.maxThreadsPerBlock
          << " numRegs=" << fa.numRegs
          << " localSizeBytes=" << fa.localSizeBytes;

  VLOG(1) << "[fmha][params.host] b=" << p.b
          << " h=" << p.h
          << " h_k=" << p.h_k
          << " seqlen_q=" << p.seqlen_q
          << " seqlen_k=" << p.seqlen_k
          << " total_q=" << p.total_q
          << " total_k=" << p.total_k
          << " is_varlen_q=" << (int)p.is_varlen_q
          << " is_varlen_k=" << (int)p.is_varlen_k
          << " scale_softmax=" << p.scale_softmax
          << " num_splits=" << p.num_splits;

  VLOG(1) << "[fmha][ptrs] q=" << p.q_ptr
          << " k=" << p.k_ptr
          << " v=" << p.v_ptr
          << " o=" << p.o_ptr
          << " lse=" << p.softmax_lse_ptr
          << " cu_q=" << p.cu_seqlens_q
          << " cu_k=" << p.cu_seqlens_k;

  VLOG(1) << "[fmha][strides] q_row=" << p.q_row_stride
          << " q_head=" << p.q_head_stride
          << " q_batch=" << p.q_batch_stride
          << " | k_row=" << p.k_row_stride
          << " k_head=" << p.k_head_stride
          << " k_batch=" << p.k_batch_stride
          << " | v_row=" << p.v_row_stride
          << " v_head=" << p.v_head_stride
          << " v_batch=" << p.v_batch_stride
          << " | o_row=" << p.o_row_stride
          << " o_head=" << p.o_head_stride
          << " o_batch=" << p.o_batch_stride
          << " | lse_head=" << p.lse_head_stride
          << " lse_batch=" << p.lse_batch_stride;

  VLOG(1) << "[fmha][types] sizeof(Arguments)=" << sizeof(typename KernelT::Arguments)
          << " alignof(Arguments)=" << alignof(typename KernelT::Arguments)
          << " sizeof(Params)=" << sizeof(typename KernelT::Params)
          << " alignof(Params)=" << alignof(typename KernelT::Params);

  VLOG(1) << "[fmha][traits] Params_trivial="
          << (int)std::is_trivially_copyable<typename KernelT::Params>::value
          << " Args_trivial="
          << (int)std::is_trivially_copyable<typename KernelT::Arguments>::value;
}

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split>
void fmha_kernel_launcher(Flash_fwd_params params, const phi::CustomContext* dev_ctx) {
  int  seqlen_q   = params.is_varlen_q ? params.total_q : params.seqlen_q;
  int  seqlen_k   = params.is_varlen_k ? params.total_k : params.seqlen_k;
  auto stride_q   = make_stride(params.q_row_stride, _1{}, make_stride(params.q_head_stride, params.q_batch_stride));
  auto stride_k   = make_stride(params.k_row_stride, _1{}, make_stride(params.k_head_stride, params.k_batch_stride));
  auto stride_v   = make_stride(params.v_row_stride, _1{}, make_stride(params.v_head_stride, params.v_batch_stride));
  auto stride_o   = make_stride(params.o_row_stride, _1{}, make_stride(params.o_head_stride, params.o_batch_stride));
  auto stride_lse = make_stride(_1{}, make_stride(params.lse_head_stride, params.lse_batch_stride));

  using StrideQ   = decltype(stride_q);
  using StrideK   = decltype(stride_k);
  using StrideV   = decltype(stride_v);
  using StrideO   = decltype(stride_o);
  using StrideLse = decltype(stride_lse);

  constexpr int Consumers = CTA_Q / 64;

  // CTA_Q, CTA_KV, D_QK, D_VO

  using TileShape = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEADDIM_QK>, Int<HEADDIM_V>>;

  static constexpr bool UseFopMask = false;
  static constexpr bool UpperLeft  = false;

  using Fusion = std::conditional_t<Causal,
                                    mutlass::fmha::collective::CausalFusion<UpperLeft, false>,
                                    mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaMainloopTmeWarpSpecialized<
      Element,
      float,
      TileShape,
      StrideQ,
      StrideK,
      StrideV,
      Fusion,
      mutlass::fmha::Option<mutlass::fmha::Tag::NumMmaWarpSquads, Int<Consumers>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::Varlen, conditional_t<Varlen, true_type, false_type>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::VStage, Int<1>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::KStage, Int<1>>>;

  using EpilogueTileShape = Shape<Int<tuple_element_t<0, TileShape>{} / Consumers>, tuple_element_t<3, TileShape>>;

  using CollectiveEpilogue =
      mutlass::fmha::collective::FmhaFwdEpilogue<ElementO, float, EpilogueTileShape, StrideO, StrideLse>;

  using TileScheduler = mutlass::fmha::kernel::FmhaIndividualTileScheduler<Causal>;
  using Kernel =
      mutlass::fmha::kernel::FmhaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  auto problem_size = make_shape(seqlen_q, seqlen_k, HEADDIM_QK, HEADDIM_V, params.h, params.h_k, params.b);

  typename Kernel::ProblemShape problem_shape;  // run kernel

  get<0>(problem_shape) = mutlass::fmha::collective::VariableLength{static_cast<int>(get<0>(problem_size)),
                                                                    static_cast<int>(params.seqlen_q),
                                                                    static_cast<int*>(params.cu_seqlens_q)};
  get<1>(problem_shape) = mutlass::fmha::collective::VariableLength{static_cast<int>(get<1>(problem_size)),
                                                                    static_cast<int>(params.seqlen_k),
                                                                    static_cast<int*>(params.cu_seqlens_k)};
  get<2>(problem_shape) = get<2>(problem_size);
  get<3>(problem_shape) = get<3>(problem_size);
  get<4>(problem_shape) = get<4>(problem_size);
  get<5>(problem_shape) = get<5>(problem_size);
  get<6>(problem_shape) = get<6>(problem_size);

  typename Kernel::Arguments arguments{
      problem_shape,
      {
          static_cast<Element*>(params.q_ptr),
          stride_q,
          static_cast<Element*>(params.k_ptr),
          stride_k,
          static_cast<Element*>(params.v_ptr),
          stride_v,
          params.scale_softmax,
      },
      {
          static_cast<ElementO*>(params.o_ptr),
          stride_o,
          static_cast<float*>(params.softmax_lse_ptr),
          stride_lse,
      },
  };

  musaStream_t            stream        = reinterpret_cast<musaStream_t>(dev_ctx->stream());
  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(arguments);

  auto grid_dim = TileScheduler::get_grid_shape(kernel_params.scheduler);

  mutlass::device_kernel<Kernel>
    <<<grid_dim, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize, stream>>>(kernel_params);

  auto launch_err = musaGetLastError();
  if (launch_err != musaSuccess) {
    VLOG(1) << "[fmha] launch_err=" << static_cast<int>(launch_err) << " (" << musaGetErrorString(launch_err) << ")"
            << " stream=" << stream
            << " grid=(" << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z << ")"
            << " block=" << Kernel::MaxThreadsPerBlock
            << " shmem=" << Kernel::SharedStorageSize;
    PADDLE_THROW(common::errors::InvalidArgument(
        "fmha launch failed, musaGetLastError=%d", static_cast<int>(launch_err)));
  }
}

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split>
void fmha_paged_kernel_launcher(Flash_fwd_params params, const phi::CustomContext* dev_ctx) {
  auto stride_q   = make_stride(params.q_row_stride, _1{}, make_stride(params.q_head_stride, params.q_batch_stride));
  auto stride_k   = make_stride(params.k_row_stride, _1{}, make_stride(params.k_head_stride, params.k_batch_stride));
  auto stride_v   = make_stride(params.v_row_stride, _1{}, make_stride(params.v_head_stride, params.v_batch_stride));
  auto stride_o   = make_stride(params.o_row_stride, _1{}, make_stride(params.o_head_stride, params.o_batch_stride));
  auto stride_lse = make_stride(_1{}, make_stride(params.lse_head_stride, params.lse_batch_stride));
  auto stride_pt  = make_stride(int(params.page_table_batch_stride), _1{});  // TODO:

  using StrideQ   = decltype(stride_q);
  using StrideK   = decltype(stride_k);
  using StrideV   = decltype(stride_v);
  using StrideO   = decltype(stride_o);
  using StrideLse = decltype(stride_lse);

  constexpr int Consumers = CTA_Q / 64;

  // CTA_Q, CTA_KV, D_QK, D_VO

  static constexpr bool UseFopMask = false;
  static constexpr bool UpperLeft  = false;
  static constexpr bool PackGQA    = false;

  using Fusion = std::conditional_t<Causal,
                                    mutlass::fmha::collective::CausalFusion<UpperLeft, PackGQA>,
                                    mutlass::fmha::collective::DefaultFusion>;

  using TileShape = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEADDIM_QK>, Int<HEADDIM_V>>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaPagedMainloopTmeWarpSpecialized<
      Element,
      float,
      TileShape,
      StrideQ,
      StrideK,
      StrideV,
      Fusion,
      mutlass::fmha::Option<mutlass::fmha::Tag::NumMmaWarpSquads, Int<Consumers>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::KStage, Int<2>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::VStage, Int<2>>>;

  using EpilogueTileShape = Shape<Int<tuple_element_t<0, TileShape>{} / Consumers>, tuple_element_t<3, TileShape>>;

  using CollectiveEpilogue =
      mutlass::fmha::collective::FmhaFwdEpilogue<ElementO, float, EpilogueTileShape, StrideO, StrideLse>;

  using TileScheduler = mutlass::fmha::kernel::FmhaIndividualTileScheduler<Causal>;
  using Kernel =
      mutlass::fmha::kernel::PagedFmhaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  typename Kernel::ProblemShape problem_shape =
      make_shape(params.seqlen_q, params.seqlen_k, HEADDIM_QK, HEADDIM_V, params.h, params.h_k, params.b);

  typename Kernel::Arguments arguments{
      problem_shape,
      {
          static_cast<Element*>(params.q_ptr),
          stride_q,
          static_cast<Element*>(params.k_ptr),
          stride_k,
          static_cast<Element*>(params.v_ptr),
          stride_v,
          static_cast<int*>(params.page_table),
          stride_pt,
          static_cast<int*>(params.seqused_k),
          params.scale_softmax,
          params.page_size,
          params.num_pages,
      },
      {
          static_cast<ElementO*>(params.o_ptr),
          stride_o,
          static_cast<float*>(params.softmax_lse_ptr),
          stride_lse,
      },
  };

  musaStream_t            stream        = reinterpret_cast<musaStream_t>(dev_ctx->stream());
  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(arguments);

  auto grid_dim = TileScheduler::get_grid_shape(kernel_params.scheduler);
  mutlass::device_kernel<Kernel>
      <<<grid_dim, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize, stream>>>(kernel_params);
  PADDLE_ENFORCE_EQ(musaGetLastError(), musaSuccess, 
            common::errors::InvalidArgument("Error in fmha_kernel_launcher"));
}

}

namespace phi {

#define CHECK_SHAPE(x, ...)                  \
  PADDLE_ENFORCE_EQ(x.dims(), phi::make_ddim({__VA_ARGS__}), \
            common::errors::InvalidArgument("Tensor " #x " must have shape (" #__VA_ARGS__ ")"))

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split,
          bool PAGED_KV>
void dispatch_fmha_kernel(Flash_fwd_params params, const phi::CustomContext* dev_ctx) {
  if constexpr (PAGED_KV) {
    mute::fmha_paged_kernel_launcher<Arch, Element, ElementO, Causal, Varlen, CTA_Q, CTA_KV, HEADDIM_QK, HEADDIM_V, Split>(
        params, dev_ctx);
  } else {
    mute::fmha_kernel_launcher<Arch, Element, ElementO, Causal, Varlen, CTA_Q, CTA_KV, HEADDIM_QK, HEADDIM_V, Split>(
        params, dev_ctx);
  }
}


template<typename T>
void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      DenseTensor& q,
                      DenseTensor& k,
                      DenseTensor& v,
                      DenseTensor& out,
                      void*            cu_seqlens_q_d,
                      void*            cu_seqlens_k_d,
                      void*            seqused_q,
                      void*            seqused_k,
                      void*            softmax_lse_d,
                      float            p_dropout,
                      float            softmax_scale,
                      int              window_size_left,
                      int              window_size_right,
                      int              attention_chunk,
                      const float      softcap   = 0.f,
                      const int        sm_margin = 0) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == phi::DataType::BFLOAT16;
  params.is_e4m3 = false;

  // Set the pointers and strides.
  params.q_ptr = q.data<T>();
  params.k_ptr = k.data<T>();
  params.v_ptr = v.data<T>();
  // All stride are in elements, not bytes.
  params.q_row_stride  = getEle(q.strides(), -3);
  params.k_row_stride  = getEle(k.strides(), -3);
  params.v_row_stride  = getEle(v.strides(), -3);
  params.q_head_stride = getEle(q.strides(), -2);
  params.k_head_stride = getEle(k.strides(), -2);
  params.v_head_stride = getEle(v.strides(), -2);
  params.v_dim_stride  = getEle(v.strides(), -1);
  params.o_ptr         = out.data<T>();
  params.o_row_stride  = getEle(out.strides(), -3);
  params.o_head_stride = getEle(out.strides(), -2);

  // if (cu_seqlens_q_d == nullptr) {
  params.q_batch_stride = q.strides()[0];
  params.o_batch_stride = out.strides()[0];
  // }
  // if (cu_seqlens_k_d == nullptr) {
  params.k_batch_stride = k.strides()[0];
  params.v_batch_stride = v.strides()[0];
  // }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_q    = static_cast<int*>(seqused_q);
  params.seqused_k    = static_cast<int*>(seqused_k);

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b                = b;
  params.h                = h;
  params.h_k              = h_k;
  params.seqlen_q         = seqlen_q;
  params.seqlen_k         = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d                = d;
  params.d_rounded        = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.softcap       = softcap;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout           = 1.f / params.p_dropout;
  PADDLE_ENFORCE_EQ(p_dropout < 1.f, true, 
    common::errors::InvalidArgument("p_dropout must be less than 1.0"));

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
  params.is_local  = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  if (attention_chunk > 0) {
    window_size_left  = std::min(window_size_left, attention_chunk - 1);
    window_size_right = std::min(window_size_right, attention_chunk - 1);
  }
  params.window_size_left  = window_size_left;
  params.window_size_right = window_size_right;
  params.attention_chunk   = attention_chunk;

  PADDLE_ENFORCE_EQ(params.is_local, false, 
    common::errors::InvalidArgument("This flash attention build does not support local attention."));
}

const phi::CustomContext* getcontext(const paddle::Tensor& tensor) {
  return static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(tensor.place()));
}

phi::DenseTensor paddletensor2densortensor(const paddle::Tensor& paddletensor) {
  return *(static_cast<const phi::DenseTensor*>(paddletensor.impl().get()));
}

phi::DenseTensor* opt_paddletensor2densortensor_ptr(
    const paddle::optional<paddle::Tensor>& opt_paddletensor) {
  if (opt_paddletensor) {
    auto ptr = *(opt_paddletensor.get_ptr());
    return static_cast<phi::DenseTensor*>(ptr.impl().get());
  } else {
    return nullptr;
  }
}

template<typename T>
std::vector<paddle::Tensor> FlashAttnKernelKVCacheMateImpl(
    const paddle::Tensor& q_,
    const paddle::Tensor& k_,
    const paddle::Tensor& v_,
    const paddle::optional<paddle::Tensor>& k_new_,
    const paddle::optional<paddle::Tensor>& v_new_,
    const paddle::optional<paddle::Tensor>& q_v_,
    const paddle::optional<paddle::Tensor>& out_,
    const paddle::optional<paddle::Tensor>& cu_seqlens_q_,
    const paddle::optional<paddle::Tensor>& cu_seqlens_k_,
    const paddle::optional<paddle::Tensor>& cu_seqlens_k_new_,
    paddle::optional<paddle::Tensor>& seqused_q_,
    paddle::optional<paddle::Tensor>& seqused_k_,
    int max_seqlen_q_,
    int max_seqlen_k_,
    const paddle::optional<paddle::Tensor>& page_table_,
    const paddle::optional<paddle::Tensor>& kv_batch_idx_,
    const paddle::optional<paddle::Tensor>& leftpad_k_,
    const paddle::optional<paddle::Tensor>& rotary_cos_,
    const paddle::optional<paddle::Tensor>& rotary_sin_,
    const paddle::optional<paddle::Tensor>& seqlens_rotary_,
    const paddle::optional<paddle::Tensor>& q_descale_,
    const paddle::optional<paddle::Tensor>& k_descale_,
    const paddle::optional<paddle::Tensor>& v_descale_,
    double softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,
    const paddle::optional<paddle::Tensor>& scheduler_metadata_,
    int64_t num_splits,
    int pack_gqa_,                     // -1 表示 std::nullopt；0/1 表示 false/true
    int64_t sm_margin
) {
    auto dev_ctx = getcontext(q_);
    auto q = paddletensor2densortensor(q_);
    auto k = paddletensor2densortensor(k_);
    auto v = paddletensor2densortensor(v_);

    Flash_fwd_params params;
    auto             q_type = q.dtype();
    PADDLE_ENFORCE_EQ(q_type == phi::DataType::FLOAT16 || q_type == phi::DataType::BFLOAT16, true, 
                common::errors::InvalidArgument("Fmha fwd only supports fp16 and bf16 data type"));
    PADDLE_ENFORCE_EQ(k.dtype() == q_type, true, 
                common::errors::InvalidArgument("query and key must have the same dtype"));
    PADDLE_ENFORCE_EQ(v.dtype() == q_type, true, 
                common::errors::InvalidArgument("query and value must have the same dtype"));

    PADDLE_ENFORCE_EQ(getEle(q.strides(), -1) == 1, true, 
                common::errors::InvalidArgument("Input tensor must have contiguous last dimension"));
    PADDLE_ENFORCE_EQ(getEle(k.strides(), -1) == 1, true, 
                common::errors::InvalidArgument("Input tensor must have contiguous last dimension"));
    PADDLE_ENFORCE_EQ(getEle(v.strides(), -1) == 1, true, 
                common::errors::InvalidArgument("Input tensor must have contiguous last dimension"));

    DenseTensor* page_table = opt_paddletensor2densortensor_ptr(page_table_);
    const bool paged_KV = page_table_.is_initialized();
    if (paged_KV) {
        PADDLE_ENFORCE_EQ(page_table->dtype(), phi::DataType::INT32, 
                common::errors::InvalidArgument("page_table must have dtype int32"));
        PADDLE_ENFORCE_EQ(getEle(page_table->strides(), -1), 1, 
                common::errors::InvalidArgument("Input tensor must have contiguous last dimension"));
    }

    DenseTensor* cu_seqlens_q = opt_paddletensor2densortensor_ptr(cu_seqlens_q_);
    bool const is_varlen_q = cu_seqlens_q_.is_initialized();
    if (is_varlen_q) {
        PADDLE_ENFORCE_EQ(cu_seqlens_q->dtype(), phi::DataType::INT32, 
                common::errors::InvalidArgument("cu_seqlens_q must have dtype int32"));
        PADDLE_ENFORCE_EQ(max_seqlen_q_ > 0, true, 
                common::errors::InvalidArgument("max_seqlen_q must be provided if cu_seqlens_q is provided"));
    }

    DenseTensor* cu_seqlens_k = opt_paddletensor2densortensor_ptr(cu_seqlens_k_);
    bool const is_varlen_k = cu_seqlens_k_.is_initialized();
    if (is_varlen_k) {
        PADDLE_ENFORCE_EQ(cu_seqlens_k->dtype(), phi::DataType::INT32, 
                common::errors::InvalidArgument("cu_seqlens_k must have dtype int32"));
        PADDLE_ENFORCE_EQ(max_seqlen_k_ > 0, true, 
                common::errors::InvalidArgument("max_seqlen_k must be provided if cu_seqlens_k is provided"));
        PADDLE_ENFORCE_EQ(paged_KV, false, 
                common::errors::InvalidArgument("If cu_seqlens_k is passed in, then page table is not supported"));
        PADDLE_ENFORCE_EQ(kv_batch_idx_.is_initialized(), false, 
                common::errors::InvalidArgument("If cu_seqlens_k is passed in, then page table is not supported"));
    }

    auto const sizes                 = q.dims();
    const int  batch_size            = !is_varlen_q ? sizes[0] : cu_seqlens_q->dims()[0] - 1;
    int        seqlen_q              = !is_varlen_q ? sizes[1] : max_seqlen_q_;
    int        total_q               = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
    int        num_heads             = getEle(q.dims(), -2);
    int const  head_size             = getEle(q.dims(), -1);
    int const  head_size_v           = getEle(v.dims(), -1);
    int const  max_num_pages_per_seq = !paged_KV ? 0 : page_table->dims()[1];
    int const  num_pages             = !paged_KV ? 0 : k.dims()[0];
    int const  page_size             = !paged_KV ? 1 : k.dims()[1];
    int const  seqlen_k =
        !is_varlen_k ? (!paged_KV ? k.dims()[1] : max_num_pages_per_seq * page_size) : max_seqlen_k_;
    int const total_k      = !is_varlen_k ? batch_size * k.dims()[1] : k.dims()[0];
    int const num_heads_k  = getEle(k.dims(), -2);
    int const batch_size_k = !paged_KV ? (!is_varlen_k ? k.dims()[0] : cu_seqlens_k->dims()[0] - 1) : page_table->dims()[0];

    if (paged_KV) {
        PADDLE_ENFORCE_EQ(page_size, 64, 
                common::errors::InvalidArgument("page_size must be 64 for paged KV now"));
        PADDLE_ENFORCE_EQ(seqused_k_.is_initialized(), true, 
                common::errors::InvalidArgument("seqused_k_ must be given for paged KV"));
    }

    double softmax_scale = 1.0 / sqrt(double(head_size));
    if (softmax_scale_ > 0) {
        softmax_scale = softmax_scale_;
    }

    if (!kv_batch_idx_.is_initialized()) {
        PADDLE_ENFORCE_EQ(batch_size, batch_size_k, 
                common::errors::InvalidArgument("batch_size must be equal to batch_size_k"));
    }

    PADDLE_ENFORCE_EQ(window_size_left == -1 && window_size_right == -1, true, 
                common::errors::InvalidArgument("mha not supported Sliding-window attention yet"));
    if (window_size_left >= seqlen_k - 1) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_q - 1) {
        window_size_right = -1;
    }
    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
        // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
        if ((head_size <= 64 || head_size > 128) || !paged_KV) {
            is_causal = false;
        }
    }


    if (head_size == 128 && head_size == 128) {
        PADDLE_ENFORCE_EQ(getEle(q.strides(), -2), getEle(q.dims(), -1), 
                common::errors::InvalidArgument("q must be contiguous in HD dim"));
    }

    if (is_causal) {
        window_size_right = 0;
    }
    if (!is_varlen_q) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    } else {
        CHECK_SHAPE(q, total_q, num_heads, head_size);
        CHECK_SHAPE((*cu_seqlens_q), batch_size + 1);
    }
    if (!paged_KV) {
        if (!is_varlen_k) {
            CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
            CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
        } else {
            CHECK_SHAPE(k, total_k, num_heads_k, head_size);
            CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
            CHECK_SHAPE((*cu_seqlens_k), batch_size + 1);
        }
    } else {
        CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
        CHECK_SHAPE((*page_table), batch_size_k, max_num_pages_per_seq);
    }

    PADDLE_ENFORCE_EQ(seqused_q_.is_initialized(), false, 
                common::errors::InvalidArgument("mha not supported Specified seqused_q_ yet"));
    DenseTensor* seqused_k = opt_paddletensor2densortensor_ptr(seqused_k_);
    if (seqused_k_.is_initialized()) {
        PADDLE_ENFORCE_EQ(seqused_k->dtype(), phi::DataType::INT32, 
                common::errors::InvalidArgument("seqused_k must have dtype int32"));
        CHECK_SHAPE((*seqused_k), batch_size);
    }

    PADDLE_ENFORCE_EQ(leftpad_k_.is_initialized(), false, 
                common::errors::InvalidArgument("mha not supported leftpad_k_ yet"));
    
    bool const is_varlen =
        is_varlen_q || is_varlen_k || seqused_q_.is_initialized() || 
        seqused_k_.is_initialized() || leftpad_k_.is_initialized();
    auto       out_type = q_type;
    phi::DenseTensor* out_raw = opt_paddletensor2densortensor_ptr(out_);
    std::shared_ptr<phi::DenseTensor> out;
    if (out_.is_initialized()) {
        out = std::shared_ptr<phi::DenseTensor>(std::shared_ptr<void>{}, out_raw);
        PADDLE_ENFORCE_EQ(getEle(out->strides(), -1), 1, 
                common::errors::InvalidArgument("Output tensor must have contiguous last dimension"));
        if (!is_varlen_q) {
            CHECK_SHAPE((*out), batch_size, seqlen_q, num_heads, head_size_v);
        } else {
            CHECK_SHAPE((*out), total_q, num_heads, head_size_v);
        }
    } else {
        out = std::make_shared<phi::DenseTensor>();
        if (!is_varlen_q) {
            out->Resize(phi::make_ddim({batch_size, seqlen_q, num_heads, head_size_v}));
        } else {
            out->Resize(phi::make_ddim({total_q, num_heads, head_size_v}));
        }
        dev_ctx->Alloc(out.get(), out_type);
    }

    // DenseTensor softmax_lse;
    std::shared_ptr<phi::DenseTensor> softmax_lse =
        std::make_shared<phi::DenseTensor>();
    if (!is_varlen_q) {
        phi::DenseTensorMeta meta(
                phi::DataType::FLOAT32,
                phi::make_ddim({batch_size, num_heads, seqlen_q}),
                phi::DataLayout::NCHW
            );
        softmax_lse.get()->set_meta(meta);
    } else {
        phi::DenseTensorMeta meta(
                phi::DataType::FLOAT32,
                phi::make_ddim({num_heads, total_q}),
                phi::DataLayout::NCHW
            );
        softmax_lse.get()->set_meta(meta);
    }
    dev_ctx->Alloc(softmax_lse.get(), phi::DataType::FLOAT32);


    auto      round_multiple      = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded   = head_size;
    int const head_size_v_rounded = head_size_v;
    int const seqlen_q_rounded    = round_multiple(seqlen_q, 128);
    int const seqlen_k_rounded    = round_multiple(seqlen_k, 128);

    set_params_fprop<T>(params,
                    batch_size,
                    seqlen_q,
                    seqlen_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    num_heads,
                    num_heads_k,
                    head_size,
                    head_size_rounded,
                    q,
                    k,
                    v,
                    *(out.get()),
                    !is_varlen_q ? nullptr : cu_seqlens_q->data(),
                    !is_varlen_k ? nullptr : cu_seqlens_k->data(),
                    seqused_q_.is_initialized() ? seqused_q_.get().data() : nullptr,
                    seqused_k_.is_initialized() ? seqused_k_.get().data() : nullptr,
                    softmax_lse.get()->data<float>(),
                    /*p_dropout=*/0.f,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    attention_chunk,
                    softcap,
                    sm_margin);

    params.lse_batch_stride = is_varlen_q ? 0 : softmax_lse.get()->strides()[0];
    params.lse_head_stride  = is_varlen_q ? softmax_lse.get()->strides()[0] : softmax_lse.get()->strides()[1];
    params.is_varlen_q      = is_varlen_q;
    params.is_varlen_k      = is_varlen_k;
    params.total_q          = total_q;
    params.total_k          = total_k;
    params.b_k              = batch_size_k;
    params.dv               = head_size_v;
    params.dv_rounded       = head_size_v_rounded;

    if (paged_KV) {
        params.page_table              = page_table->data<int>();
        params.page_table_batch_stride = page_table->strides()[0];
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    PADDLE_ENFORCE_EQ(k_new_.is_initialized(), false, 
                common::errors::InvalidArgument("mha not supported k_new_ yet"));
    PADDLE_ENFORCE_EQ(rotary_cos_.is_initialized(), false, 
                common::errors::InvalidArgument("mha not supported rotary yet"));
    params.rotary_dim = 0;

    PADDLE_ENFORCE_EQ(kv_batch_idx_.is_initialized(), false, 
                common::errors::InvalidArgument("mha not supported kv_batch_idx_ yet"));

    DenseTensor out_accum, softmax_lse_accum;
    auto       outaccum_type = phi::DataType::FLOAT32;
    if (params.num_splits > 1) {
        PADDLE_ENFORCE_EQ(params.num_splits <= 256, true, 
                common::errors::InvalidArgument("num_splits > 256 not supported"));
        if (!is_varlen_q) {
            phi::DenseTensorMeta meta_out_accum(
                outaccum_type,
                phi::make_ddim({params.num_splits, batch_size, num_heads, seqlen_q, head_size_v}),
                phi::DataLayout::NCHW
            );
            out_accum.set_meta(meta_out_accum);
            dev_ctx->Alloc(&out_accum, outaccum_type);

            phi::DenseTensorMeta meta_softmax_lse_accum(
                outaccum_type,
                phi::make_ddim({params.num_splits, batch_size, num_heads, seqlen_q}),
                phi::DataLayout::NCHW
            );
            softmax_lse_accum.set_meta(meta_softmax_lse_accum);
            dev_ctx->Alloc(&softmax_lse_accum, outaccum_type);

            params.oaccum_batch_stride   = out_accum.strides()[1];
            params.lseaccum_batch_stride = softmax_lse_accum.strides()[1];
        } else {
            phi::DenseTensorMeta meta_out_accum(
                outaccum_type,
                phi::make_ddim({params.num_splits, num_heads, total_q, head_size_v}),
                phi::DataLayout::NCHW
            );
            out_accum.set_meta(meta_out_accum);
            dev_ctx->Alloc(&out_accum, outaccum_type);

            phi::DenseTensorMeta meta_softmax_lse_accum(
                outaccum_type,
                phi::make_ddim({params.num_splits, num_heads, total_q}),
                phi::DataLayout::NCHW
            );
            softmax_lse_accum.set_meta(meta_softmax_lse_accum);
            dev_ctx->Alloc(&softmax_lse_accum, outaccum_type);
        }
        params.is_fp32               = false;
        params.oaccum_ptr            = out_accum.data<float>();
        params.softmax_lseaccum_ptr  = softmax_lse_accum.data<float>();
        params.oaccum_split_stride   = out_accum.strides()[0];
        params.oaccum_row_stride     = getEle(out_accum.strides(), -2);
        params.oaccum_head_stride    = getEle(out_accum.strides(), -3);
        params.lseaccum_split_stride = softmax_lse_accum.strides()[0];
        params.lseaccum_head_stride  = getEle(softmax_lse_accum.strides(), -2);
    }

    PADDLE_ENFORCE_EQ(num_heads % num_heads_k, 0, 
                common::errors::InvalidArgument("Number of heads in key/value must divide number of heads in query"));
    PADDLE_ENFORCE_EQ(getEle(k.strides(), -2), getEle(k.dims(), -1), 
                common::errors::InvalidArgument("k (num_heads, head_size) dimention must be contiguous"));
    PADDLE_ENFORCE_EQ(getEle(out->strides(), -1), 1, 
                common::errors::InvalidArgument("out tensor must have contiguous last dimension"));
    
    #define LAUNCH_KERNEL(QKV_TYPE, OUT_TYPE, IS_CAUSAL, IS_VARLEN, CTA_Q, CTA_KV, HEADDIM_QK, HEADDIM_V, kPagedKV) \
    [&] {                                                                                                         \
        dispatch_fmha_kernel<31,                                                                                    \
                            QKV_TYPE,                                                                              \
                            OUT_TYPE,                                                                              \
                            IS_CAUSAL,                                                                             \
                            IS_VARLEN,                                                                             \
                            CTA_Q,                                                                                 \
                            CTA_KV,                                                                                \
                            HEADDIM_QK,                                                                            \
                            HEADDIM_V,                                                                             \
                            false,                                                                                 \
                            kPagedKV>(params, dev_ctx);                                                                     \
    }()

    FP16_SWITCH(q_type == phi::DataType::FLOAT16, [&] {
        HEADDIM_SWITCH(head_size, head_size_v, [&] {
            BOOL_SWITCH(is_causal, kIsCausal, [&] {
                BOOL_SWITCH(is_varlen, kIsVarlen, [&] {
                    BOOL_SWITCH(paged_KV, kPagedKV, [&] {
                        constexpr auto cfg         = get_tile_config<kHeadSizeQK, kHeadSizeV, kPagedKV>();
                        constexpr int  kCTA_Q      = cfg.kCTA_Q;
                        constexpr int  kCTA_K      = cfg.kCTA_K;
                        constexpr int  kHeadSizeQK = cfg.kHeadSizeQK;
                        constexpr int  kHeadSizeV  = cfg.kHeadSizeV;

                        if constexpr (kCTA_Q != 0) {
                            LAUNCH_KERNEL(
                                elem_type, elem_type, kIsCausal, kIsVarlen, kCTA_Q, kCTA_K, kHeadSizeQK, kHeadSizeV, kPagedKV);
                        } else {
                            PADDLE_THROW(common::errors::InvalidArgument(
                                "mutlass fmha unsupported head dim combination: "
                                "head_size_qk=%d head_size_v=%d paged_kv=%d",
                                (int)kHeadSizeQK, (int)kHeadSizeV, (int)kPagedKV));
                        }
                    });
                });
            });
        });
    });

    return {paddle::Tensor(out), paddle::Tensor(softmax_lse)};
    // return {out, softmax_lse, out_accum, softmax_lse_accum};
}

std::vector<paddle::Tensor> FlashAttnKernelKVCacheMate(
    const paddle::Tensor& q_,
    const paddle::Tensor& k_,
    const paddle::Tensor& v_,
    const paddle::optional<paddle::Tensor>& k_new_,
    const paddle::optional<paddle::Tensor>& v_new_,
    const paddle::optional<paddle::Tensor>& q_v_,
    const paddle::optional<paddle::Tensor>& out_,
    const paddle::optional<paddle::Tensor>& cu_seqlens_q_,
    const paddle::optional<paddle::Tensor>& cu_seqlens_k_,
    const paddle::optional<paddle::Tensor>& cu_seqlens_k_new_,
    paddle::optional<paddle::Tensor>& seqused_q_,
    paddle::optional<paddle::Tensor>& seqused_k_,
    int max_seqlen_q_,
    int max_seqlen_k_,
    const paddle::optional<paddle::Tensor>& page_table_,
    const paddle::optional<paddle::Tensor>& kv_batch_idx_,
    const paddle::optional<paddle::Tensor>& leftpad_k_,
    const paddle::optional<paddle::Tensor>& rotary_cos_,
    const paddle::optional<paddle::Tensor>& rotary_sin_,
    const paddle::optional<paddle::Tensor>& seqlens_rotary_,
    const paddle::optional<paddle::Tensor>& q_descale_,
    const paddle::optional<paddle::Tensor>& k_descale_,
    const paddle::optional<paddle::Tensor>& v_descale_,
    double softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,
    const paddle::optional<paddle::Tensor>& scheduler_metadata_,
    int64_t num_splits,
    int pack_gqa_,
    int64_t sm_margin) {
  const auto dtype = q_.dtype();

  switch (dtype) {
    case phi::DataType::FLOAT16:
      return FlashAttnKernelKVCacheMateImpl<phi::dtype::float16>(
          q_, k_, v_, k_new_, v_new_, q_v_, out_,
          cu_seqlens_q_, cu_seqlens_k_, cu_seqlens_k_new_,
          seqused_q_, seqused_k_,
          max_seqlen_q_, max_seqlen_k_,
          page_table_, kv_batch_idx_, leftpad_k_,
          rotary_cos_, rotary_sin_, seqlens_rotary_,
          q_descale_, k_descale_, v_descale_,
          softmax_scale_, is_causal,
          window_size_left, window_size_right, attention_chunk,
          softcap, is_rotary_interleaved,
          scheduler_metadata_, num_splits, pack_gqa_, sm_margin);

    case phi::DataType::BFLOAT16:
      return FlashAttnKernelKVCacheMateImpl<phi::dtype::bfloat16>(
          q_, k_, v_, k_new_, v_new_, q_v_, out_,
          cu_seqlens_q_, cu_seqlens_k_, cu_seqlens_k_new_,
          seqused_q_, seqused_k_,
          max_seqlen_q_, max_seqlen_k_,
          page_table_, kv_batch_idx_, leftpad_k_,
          rotary_cos_, rotary_sin_, seqlens_rotary_,
          q_descale_, k_descale_, v_descale_,
          softmax_scale_, is_causal,
          window_size_left, window_size_right, attention_chunk,
          softcap, is_rotary_interleaved,
          scheduler_metadata_, num_splits, pack_gqa_, sm_margin);

    case phi::DataType::FLOAT32:
      return FlashAttnKernelKVCacheMateImpl<float>(
          q_, k_, v_, k_new_, v_new_, q_v_, out_,
          cu_seqlens_q_, cu_seqlens_k_, cu_seqlens_k_new_,
          seqused_q_, seqused_k_,
          max_seqlen_q_, max_seqlen_k_,
          page_table_, kv_batch_idx_, leftpad_k_,
          rotary_cos_, rotary_sin_, seqlens_rotary_,
          q_descale_, k_descale_, v_descale_,
          softmax_scale_, is_causal,
          window_size_left, window_size_right, attention_chunk,
          softcap, is_rotary_interleaved,
          scheduler_metadata_, num_splits, pack_gqa_, sm_margin);

    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "FlashAttnKernelKVCacheMate: unsupported dtype %d",
          static_cast<int>(dtype)));
  }
}

std::vector<std::vector<int64_t>> fusedattentionInferShapeMate(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape) {
  return {q_shape, k_shape, v_shape};
}

}

PD_BUILD_OP(flash_attn_kvcache_mate)
    .Inputs({
        "q_",
        "k_",
        "v_",
        paddle::Optional("k_new_"),
        paddle::Optional("v_new_"),
        paddle::Optional("q_v_"),
        paddle::Optional("out_"),
        paddle::Optional("cu_seqlens_q_"),
        paddle::Optional("cu_seqlens_k_"),
        paddle::Optional("cu_seqlens_k_new_"),
        paddle::Optional("seqused_q_"),
        paddle::Optional("seqused_k_"),
        paddle::Optional("page_table_"),
        paddle::Optional("kv_batch_idx_"),
        paddle::Optional("leftpad_k_"),
        paddle::Optional("rotary_cos_"),
        paddle::Optional("rotary_sin_"),
        paddle::Optional("seqlens_rotary_"),
        paddle::Optional("q_descale_"),
        paddle::Optional("k_descale_"),
        paddle::Optional("v_descale_"),
        paddle::Optional("scheduler_metadata_"),
    })
    .Outputs({
        "out_",
        "softmax_lse_"
    })
    .Attrs({
        // use -1 to represent None
        "max_seqlen_q_:int",
        "max_seqlen_k_:int",
        "softmax_scale_:double",
        "is_causal:bool",
        "window_size_left:int64_t",
        "window_size_right:int64_t",
        "attention_chunk:int64_t",
        "softcap:double",
        "is_rotary_interleaved:bool",
        "num_splits:int64_t",
        "pack_gqa_:int",
        "sm_margin:int64_t",
    })
    .SetKernelFn(PD_KERNEL(phi::FlashAttnKernelKVCacheMate))
    .SetInferShapeFn(PD_INFER_SHAPE(phi::fusedattentionInferShapeMate));