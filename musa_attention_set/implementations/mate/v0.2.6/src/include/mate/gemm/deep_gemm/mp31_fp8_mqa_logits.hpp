#pragma once

#include <mutlass/mutlass.h>

#include <mute/arch/simd_mp31.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "../../common/mma_mp31_sqmma.hpp"
#include "mate/attention/fmha/utils.hpp"

namespace mate::deep_gemm {

using namespace mute;

using f4 = float __attribute__((ext_vector_type(4)));

MUTLASS_DEVICE f4 bst4_relu(f4 v) {
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  return __musa_max_f_bst4_sv(0.0f, v);
#else
  return f4{
      v[0] > 0.0f ? v[0] : 0.0f,
      v[1] > 0.0f ? v[1] : 0.0f,
      v[2] > 0.0f ? v[2] : 0.0f,
      v[3] > 0.0f ? v[3] : 0.0f,
  };
#endif
}

MUTLASS_DEVICE f4 bst4_mul_vv(f4 a, f4 b) {
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  return __musa_mul_f_bst4_vv(a, b);
#else
  return f4{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
#endif
}

MUTLASS_DEVICE f4 bst4_mul_sv(float s, f4 v) {
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  return __musa_mul_f_bst4_sv(s, v);
#else
  return f4{s * v[0], s * v[1], s * v[2], s * v[3]};
#endif
}

MUTLASS_DEVICE f4 bst4_fma_vvv(f4 a, f4 b, f4 c) {
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  return __musa_fma_f_bst4_vvv(a, b, c);
#else
  return f4{
      a[0] * b[0] + c[0],
      a[1] * b[1] + c[1],
      a[2] * b[2] + c[2],
      a[3] * b[3] + c[3],
  };
#endif
}

MUTLASS_DEVICE f4 bst4_fma_svv(float w, f4 v, f4 acc) {
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  return __musa_fma_f_bst4_svv(w, v, acc);
#else
  return f4{
      w * v[0] + acc[0],
      w * v[1] + acc[1],
      w * v[2] + acc[2],
      w * v[3] + acc[3],
  };
#endif
}

MUTLASS_DEVICE float hsum4(f4 v) {
  return v[0] + v[1] + v[2] + v[3];
}

MUTLASS_DEVICE f4 bst4_add_vv(f4 a, f4 b) {
#if defined(MUTE_ARCH_SIMD_MATH_ENABLED)
  return __musa_fma_f_bst4_svv(1.0f, b, a);
#else
  return f4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
#endif
}

MUTLASS_DEVICE f4 shfl_xor_f4(f4 v, int lane_mask) {
#if defined(__MUSA_ARCH__)
  return f4{
      __shfl_xor_sync(0xffffffffu, v[0], lane_mask),
      __shfl_xor_sync(0xffffffffu, v[1], lane_mask),
      __shfl_xor_sync(0xffffffffu, v[2], lane_mask),
      __shfl_xor_sync(0xffffffffu, v[3], lane_mask),
  };
#else
  return v;
#endif
}

template <int kRT>
MUTLASS_DEVICE f4 warp_group_reduce_sum_f4(f4 v) {
  MUTE_UNROLL
  for (int off = 1; off < kRT; off <<= 1) {
    f4 t = shfl_xor_f4(v, off);
    v    = bst4_add_vv(v, t);
  }
  return v;
}

struct MqaLogitsTask {
  uint32_t q_group_idx;
  uint32_t kv_start_block;
  uint32_t num_kv_blocks;
};

template <uint32_t kBlockKV, uint32_t kBlockQ>
__device__ inline bool next_q_block(uint32_t       seq_q_groups,
                                    uint32_t       total_seq_q,
                                    uint32_t       seq_kv,
                                    uint32_t       max_seq_kv,
                                    int32_t const* ks,
                                    int32_t const* ke,
                                    uint32_t       mp_idx,
                                    uint32_t       num_mps,
                                    uint32_t&      q_iter_idx,
                                    MqaLogitsTask& task) {
  while (true) {
    uint32_t q_group = mp_idx + q_iter_idx * num_mps;
    if (q_group >= seq_q_groups) {
      return false;
    }

    uint32_t start    = seq_kv;
    uint32_t end      = 0;
    uint32_t base_row = q_group * kBlockQ;

    MUTE_UNROLL
    for (uint32_t i = 0; i < kBlockQ; ++i) {
      uint32_t q_row = base_row + i;
      if (q_row >= total_seq_q) break;

      int32_t ks_i = __ldg(ks + q_row);
      int32_t ke_i = __ldg(ke + q_row);

      ks_i = max(int32_t(0), min(ks_i, static_cast<int32_t>(seq_kv)));
      ke_i = max(int32_t(0), min(ke_i, static_cast<int32_t>(seq_kv)));

      start = min(start, static_cast<uint32_t>(ks_i));
      end   = max(end, static_cast<uint32_t>(ke_i));
    }

    if (end <= start) {
      ++q_iter_idx;
      continue;
    }

    uint32_t end_cap = start + max_seq_kv;
    end              = min(end, seq_kv);
    end              = min(end, end_cap);

    if (end <= start) {
      ++q_iter_idx;
      continue;
    }

    uint32_t start_block = start / kBlockKV;
    uint32_t end_block   = (end + kBlockKV - 1u) / kBlockKV;

    task.q_group_idx    = q_group;
    task.kv_start_block = start_block;
    task.num_kv_blocks  = end_block - start_block;

    ++q_iter_idx;
    return true;
  }
}

template <bool     kIsCompressedLogits,
          uint32_t kBlockQ_,
          uint32_t kNumHeads_,
          uint32_t kHeadDim_,
          uint32_t kBlockKv_,
          class StrideQ_,
          class StrideK_,
          class StrideSFK_,
          class StrideW_,
          class StrideLogits_>
struct Mp31Fp8NonPagedMqaLogits {
  using Element            = float_e4m3_t;
  using ElementAccumulator = float;

  using StrideQ      = StrideQ_;
  using StrideK      = StrideK_;
  using StrideSFK    = StrideSFK_;
  using StrideW      = StrideW_;
  using StrideLogits = StrideLogits_;

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumMmaWarpSquads  = 4;

  static constexpr int kAlignment = 32 / sizeof_bits_v<Element>;

  static constexpr int kStagesQ = 2;
  static constexpr int kStagesK = 2;

  static constexpr int kBlockQ   = int(kBlockQ_);
  static constexpr int kNumHeads = int(kNumHeads_);
  static constexpr int kHeadDim  = int(kHeadDim_);
  static constexpr int kBlockKV  = int(kBlockKv_);

  using TileShape = Shape<Int<kBlockKV>, Int<kBlockQ * kNumHeads>, Int<kHeadDim>>;

  using CollectiveMma = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                              mutlass::arch::OpClassTensorOp,
                                                                              Element,
                                                                              StrideK,
                                                                              kAlignment,
                                                                              Element,
                                                                              StrideQ,
                                                                              kAlignment,
                                                                              ElementAccumulator,
                                                                              TileShape,
                                                                              Shape<_1, _1, _1>,
                                                                              Int<kStagesQ>,
                                                                              mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMma    = typename CollectiveMma::TiledMma;
  using SmemLayoutQ = typename CollectiveMma::SmemLayoutB;

  using SmemLayoutK = decltype(tile_to_shape(typename CollectiveMma::SmemLayoutA{},
                                             Shape<Int<kBlockKV * NumMmaWarpSquads>, Int<kHeadDim>, Int<kStagesK>>{}));

  using SmemLayoutSFK = Layout<Shape<Int<kBlockKV>, _1, Int<kStagesK>, Int<NumMmaWarpSquads>>>;

  using SmemLayoutW = Layout<Shape<Int<kBlockQ * kNumHeads>, _1, Int<kStagesQ>>>;

  using SmemLayoutWView =
      decltype(make_ordered_layout(make_shape(Int<kBlockQ>{}, Int<kNumHeads>{}, Int<kStagesQ>{}), Step<_1, _0, _2>{}));

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, 256> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, 256> smem_k;
    mute::array_aligned<float, cosize_v<SmemLayoutW>, 256>   smem_weights;
    mute::array_aligned<float, cosize_v<SmemLayoutSFK>, 256> smem_sfk;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  using TME_Q = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
      take<0, 2>(typename CollectiveMma::SmemLayoutB{})));

  using TME_K = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideK{}, int32_t(0)), StrideK{}),
      take<0, 2>(typename CollectiveMma::SmemLayoutA{})));

  using TME_SFK = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), repeat_like(StrideSFK{}, int32_t(0)), StrideSFK{}),
      take<0, 2>(SmemLayoutSFK{})));

  using TME_W = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), repeat_like(StrideW{}, int32_t(0)), StrideW{}),
      take<0, 2>(SmemLayoutW{})));

  using PipelineQ       = mutlass::Mp31PipelineTmeAsync<kStagesQ>;
  using PipelineK       = mutlass::Mp31PipelineTmeAsync<kStagesK>;
  using PipelineQParams = typename PipelineQ::Params;
  using PipelineQState  = typename PipelineQ::PipelineState;
  using PipelineKParams = typename PipelineK::Params;
  using PipelineKState  = typename PipelineK::PipelineState;

  static constexpr int TmeTransactionBytesQAndWeight =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutQ{})) * sizeof_bits_v<Element>) +
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutW{})) * sizeof_bits_v<float>);

  static constexpr int TmeTransactionBytesKAndSFK =
      mutlass::bits_to_bytes(size(take<0, 2>(typename CollectiveMma::SmemLayoutA{})) * sizeof_bits_v<Element>) +
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutSFK{})) * sizeof_bits_v<float>);

  static constexpr int      SmemAlignmentBytes         = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad;

  static constexpr uint32_t NumMmaThreads = NumMmaWarpSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t NumMmaWarps   = NumMmaThreads / mutlass::NumThreadsPerWarp;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineQ[PipelineQ::NumBarriers];
    uint8_t PipelineK[PipelineK::NumBarriers * NumMmaWarpSquads];
  };

  struct Arguments {
    int32_t q_group;
    int32_t seq_kv;
    int32_t num_heads;
    int32_t head_dim;
    int32_t total_seq_q;

    Element const* ptr_q;
    StrideQ        stride_q;

    Element const* ptr_k;
    StrideK        stride_k;

    float const* ptr_sfk;
    StrideSFK    stride_sfk;

    float const* ptr_weights;
    StrideW      stride_weights;

    int32_t const* ptr_ks;
    int32_t const* ptr_ke;

    float*       ptr_logits;
    StrideLogits stride_logits;

    int32_t max_seq_kv;
  };

  struct Params {
    int32_t q_group;
    int32_t seq_kv;
    int32_t num_heads;
    int32_t head_dim;

    int32_t total_seq_q;
    int32_t num_kv_blocks;

    TME_Q   tme_q;
    TME_K   tme_k;
    TME_SFK tme_sfk;
    TME_W   tme_w;

    int32_t const* ptr_ks;
    int32_t const* ptr_ke;

    float*       ptr_logits;
    StrideLogits stride_logits;

    int32_t max_seq_kv;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    int32_t const q_group     = args.q_group;
    int32_t const seq_kv      = args.seq_kv;
    int32_t const num_heads   = args.num_heads;
    int32_t const head_dim    = args.head_dim;
    int32_t       total_seq_q = args.total_seq_q;

    int32_t const block_kv_i32  = int32_t(kBlockKV);
    int32_t       num_kv_blocks = (seq_kv + block_kv_i32 - 1) / block_kv_i32;

    int32_t max_seq_kv = args.max_seq_kv > 0 ? args.max_seq_kv : seq_kv;

    TME_Q tme_q = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_q), make_shape(total_seq_q * num_heads, head_dim), args.stride_q),
        take<0, 2>(typename CollectiveMma::SmemLayoutB{}));

    TME_K tme_k = make_tme_copy(MP31_TME_LOAD{},
                                make_tensor(make_gmem_ptr(args.ptr_k), make_shape(seq_kv, head_dim), args.stride_k),
                                take<0, 2>(typename CollectiveMma::SmemLayoutA{}));

    TME_SFK tme_sfk =
        make_tme_copy(MP31_TME_LOAD{},
                      make_tensor(make_gmem_ptr(args.ptr_sfk), make_shape(seq_kv, int32_t(1)), args.stride_sfk),
                      take<0, 2>(SmemLayoutSFK{}));

    TME_W tme_w = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(
            make_gmem_ptr(args.ptr_weights), make_shape(total_seq_q * num_heads, int32_t(1)), args.stride_weights),
        take<0, 2>(SmemLayoutW{}));

    return Params{
        .q_group       = q_group,
        .seq_kv        = seq_kv,
        .num_heads     = num_heads,
        .head_dim      = head_dim,
        .total_seq_q   = total_seq_q,
        .num_kv_blocks = num_kv_blocks,
        .tme_q         = tme_q,
        .tme_k         = tme_k,
        .tme_sfk       = tme_sfk,
        .tme_w         = tme_w,
        .ptr_ks        = args.ptr_ks,
        .ptr_ke        = args.ptr_ke,
        .ptr_logits    = args.ptr_logits,
        .stride_logits = args.stride_logits,
        .max_seq_kv    = max_seq_kv,
    };
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    enum class WarpSquadRole {
      Producer  = 0,
      Consumer0 = 1,
      Consumer1 = 2,
      Consumer2 = 3,
      Consumer3 = 4,
    };

    const int thread_idx               = threadIdx.x;
    const int warp_idx                 = mutlass::canonical_warp_idx_sync();
    const int warp_squad_idx           = mutlass::canonical_warp_squad_idx();
    const int warp_idx_in_warp_squad   = warp_idx % mutlass::NumWarpsPerWarpSquad;
    const int consumer_warp_squad_idx  = warp_squad_idx - int(WarpSquadRole::Consumer0);
    const int thread_idx_in_warp_squad = thread_idx % mutlass::NumThreadsPerWarpSquad;

    const auto warp_squad_role = WarpSquadRole(warp_squad_idx);

    const int kv_group_idx =
        (warp_squad_role == WarpSquadRole::Producer) ? warp_idx_in_warp_squad : consumer_warp_squad_idx;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    // --------------------- Pipeline Q ---------------------
    PipelineQParams pipeline_params_q;
    pipeline_params_q.transaction_bytes = TmeTransactionBytesQAndWeight;
    pipeline_params_q.num_consumers     = NumMmaWarps;
    pipeline_params_q.num_producers     = 1;

    PipelineQ      pipeline_q(pipeline_params_q, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    PipelineQState pipeline_q_producer_state = mutlass::make_producer_start_state<PipelineQ>();
    PipelineQState pipeline_q_consumer_state;

    // --------------------- Pipeline K ---------------------
    PipelineKParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = TmeTransactionBytesKAndSFK;
    pipeline_params_k.num_consumers     = mutlass::NumWarpsPerWarpSquad;
    pipeline_params_k.num_producers     = 1;

    PipelineK pipeline_k(pipeline_params_k,
                         static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->PipelineK)) +
                             kv_group_idx * PipelineK::NumBarriers,
                         mutlass::NumWarpsPerWarpSquad);

    PipelineKState pipeline_k_producer_state = mutlass::make_producer_start_state<PipelineK>();
    PipelineKState pipeline_k_consumer_state;

    __syncthreads();

    if (warp_squad_role == WarpSquadRole::Producer) {
      PipelineQState& pipe_q_write = pipeline_q_producer_state;
      PipelineKState& pipe_k_write = pipeline_k_producer_state;

      const int lane_idx = thread_idx % mutlass::NumThreadsPerWarp;

      auto cta_tme_q   = params.tme_q.get_slice(0);
      auto cta_tme_k   = params.tme_k.get_slice(0);
      auto cta_tme_sfk = params.tme_sfk.get_slice(0);
      auto cta_tme_w   = params.tme_w.get_slice(0);

      // ---------------- Q ----------------
      Tensor mQ      = params.tme_q.get_tme_tensor(make_shape(params.total_seq_q * params.num_heads, params.head_dim));
      Tensor gQ_full = local_tile(mQ, select<1, 2>(TileShape{}), make_coord(_, _));
      Tensor sQ      = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
      Tensor tQgQ    = cta_tme_q.partition_S(gQ_full(_, _, _, _0{}));
      Tensor tQsQ    = cta_tme_q.partition_D(sQ);

      // ---------------- K ----------------
      Tensor mK      = params.tme_k.get_tme_tensor(make_shape(params.seq_kv, params.head_dim));
      Tensor gK_full = local_tile(mK, select<0, 2>(TileShape{}), make_coord(_, _));
      Tensor sK      = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
      Tensor tKgK    = cta_tme_k.partition_S(gK_full(_, _, _, _0{}));
      Tensor tKsK    = cta_tme_k.partition_D(sK(make_coord(_, kv_group_idx), _, _));

      // ---------------- SFK ----------------
      Tensor mSFK      = params.tme_sfk.get_tme_tensor(make_shape(params.seq_kv, Int<1>{}));
      Tensor gSFK_full = local_tile(mSFK, Shape<Int<kBlockKV>, _1>{}, make_coord(_, _));
      Tensor sSFK      = make_tensor(make_smem_ptr(shared_storage.smem_sfk.data()), SmemLayoutSFK{});
      Tensor tSFKgSFK  = cta_tme_sfk.partition_S(gSFK_full(_, _, _, _0{}));
      Tensor tSFKsSFK  = cta_tme_sfk.partition_D(sSFK(_, _, _, kv_group_idx));

      // ---------------- Weights ----------------
      Tensor mW      = params.tme_w.get_tme_tensor(make_shape(params.total_seq_q * params.num_heads, Int<1>{}));
      Tensor gW_full = local_tile(mW, Shape<Int<kBlockQ * kNumHeads>, _1>{}, make_coord(_, _));
      Tensor sW      = make_tensor(make_smem_ptr(shared_storage.smem_weights.data()), SmemLayoutW{});
      Tensor tWgW    = cta_tme_w.partition_S(gW_full(_, _, _, _0{}));
      Tensor tWsW    = cta_tme_w.partition_D(sW);

      auto issue_tme_q = [&](uint32_t issued_q_group) {
        if (kv_group_idx == 0 && lane_idx == 0) {
          pipeline_q.producer_acquire(pipe_q_write);
          const uint32_t bar_id = pipeline_q.producer_get_barrier_id(pipe_q_write);

          const int stage = pipe_q_write.index();

          copy(params.tme_q.with(bar_id), tQgQ(_, _, _, issued_q_group), tQsQ(_, _, _, stage));
          copy(params.tme_w.with(bar_id), tWgW(_, _, _, issued_q_group), tWsW(_, _, _, stage));

          ++pipe_q_write;
        }
      };

      const uint32_t mp_idx       = (uint32_t)blockIdx.x;
      const uint32_t num_mps      = (uint32_t)gridDim.x;
      const uint32_t seq_q_groups = (uint32_t)params.q_group;

      uint32_t q_iter_idx = 0;

      MqaLogitsTask curr_task{};
      MqaLogitsTask next_task{};

      bool has_curr = next_q_block<kBlockKV, kBlockQ>(seq_q_groups,
                                                      (uint32_t)params.total_seq_q,
                                                      (uint32_t)params.seq_kv,
                                                      (uint32_t)params.max_seq_kv,
                                                      params.ptr_ks,
                                                      params.ptr_ke,
                                                      mp_idx,
                                                      num_mps,
                                                      q_iter_idx,
                                                      curr_task);

      if (!has_curr) {
        return;
      }

      issue_tme_q(curr_task.q_group_idx);

      bool has_next = next_q_block<kBlockKV, kBlockQ>(seq_q_groups,
                                                      (uint32_t)params.total_seq_q,
                                                      (uint32_t)params.seq_kv,
                                                      (uint32_t)params.max_seq_kv,
                                                      params.ptr_ks,
                                                      params.ptr_ke,
                                                      mp_idx,
                                                      num_mps,
                                                      q_iter_idx,
                                                      next_task);

      if (has_next) {
        issue_tme_q(next_task.q_group_idx);
      }

      while (has_curr) {
        for (uint32_t kv_block_in_group = (uint32_t)kv_group_idx; kv_block_in_group < curr_task.num_kv_blocks;
             kv_block_in_group += (uint32_t)NumMmaWarpSquads) {
          const uint32_t kv_block_idx = curr_task.kv_start_block + kv_block_in_group;

          if (lane_idx == 0) {
            pipeline_k.producer_acquire(pipe_k_write);
            const uint32_t bar_id = pipeline_k.producer_get_barrier_id(pipe_k_write);

            const int kstage = pipe_k_write.index();
            copy(params.tme_k.with(bar_id), tKgK(_, _, _, kv_block_idx), tKsK(_, _, _, kstage));
            copy(params.tme_sfk.with(bar_id), tSFKgSFK(_, _, _, kv_block_idx), tSFKsSFK(_, _, _, kstage));

            ++pipe_k_write;
          }
        }

        if (!has_next) {
          break;
        }

        curr_task = next_task;

        has_next = next_q_block<kBlockKV, kBlockQ>(seq_q_groups,
                                                   (uint32_t)params.total_seq_q,
                                                   (uint32_t)params.seq_kv,
                                                   (uint32_t)params.max_seq_kv,
                                                   params.ptr_ks,
                                                   params.ptr_ke,
                                                   mp_idx,
                                                   num_mps,
                                                   q_iter_idx,
                                                   next_task);

        if (has_next) {
          issue_tme_q(next_task.q_group_idx);
        }
      }

      return;
    }

    {
      PipelineQState& pipe_q_read = pipeline_q_consumer_state;
      PipelineKState& pipe_k_read = pipeline_k_consumer_state;

      const int lane_idx = thread_idx_in_warp_squad % mutlass::NumThreadsPerWarp;

      TiledMma tiled_mma;
      auto     thr_mma = tiled_mma.get_thread_slice(thread_idx_in_warp_squad);

      constexpr int reduction_target = size(mate::attention::fmha::reduction_target_n(tiled_mma));
      Tensor        weights          = make_tensor<float>(Shape<Int<kBlockQ>, Int<kNumHeads / reduction_target>>{});

      Tensor sW   = make_tensor(make_smem_ptr(shared_storage.smem_weights.data()), SmemLayoutWView{});
      Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_sfk.data()), SmemLayoutSFK{});
      Tensor sQ   = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
      Tensor sK   = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});

      Tensor tQsQ = thr_mma.partition_B(sQ);
      Tensor tQrQ = thr_mma.make_fragment_B(tQsQ);

      Tensor tKsK = thr_mma.partition_A(sK(make_coord(_, kv_group_idx), _, _));
      Tensor tKrK = thr_mma.make_fragment_A(tKsK);

      const int64_t stride_row = get<0>(params.stride_logits);

      const uint32_t mp_idx       = (uint32_t)blockIdx.x;
      const uint32_t num_mps      = (uint32_t)gridDim.x;
      const uint32_t seq_q_groups = (uint32_t)params.q_group;

      uint32_t      q_iter_idx = 0;
      MqaLogitsTask task{};

      while (next_q_block<kBlockKV, kBlockQ>(seq_q_groups,
                                             (uint32_t)params.total_seq_q,
                                             (uint32_t)params.seq_kv,
                                             (uint32_t)params.max_seq_kv,
                                             params.ptr_ks,
                                             params.ptr_ke,
                                             mp_idx,
                                             num_mps,
                                             q_iter_idx,
                                             task)) {
        pipeline_q.consumer_wait(pipe_q_read);
        const int q_stage = pipe_q_read.index();

        const int J = int(kNumHeads / reduction_target);

        MUTE_UNROLL
        for (int i = 0; i < kBlockQ; ++i) {
          MUTE_UNROLL
          for (int j = 0; j < kNumHeads / reduction_target; ++j) {
            weights(i, j) = sW(i, j * reduction_target + lane_idx % reduction_target, q_stage);
          }
        }

        for (uint32_t kv_block_in_group = (uint32_t)kv_group_idx; kv_block_in_group < task.num_kv_blocks;
             kv_block_in_group += (uint32_t)NumMmaWarpSquads) {
          const uint32_t kv_block_idx = task.kv_start_block + kv_block_in_group;

          pipeline_k.consumer_wait(pipe_k_read);
          const uint32_t kv_stage_idx = pipe_k_read.index();

          Tensor accum    = partition_fragment_C(TiledMma{}, take<0, 2>(TileShape{}));
          Tensor accum_mn = make_tensor(accum.data(), mate::attention::fmha::layout_acc_mn(tiled_mma, accum.layout()));
          clear(accum);

          mute::gemm(tiled_mma, tKrK(_, _, _, kv_stage_idx), tQrQ(_, _, _, q_stage), accum);
          mate::warpsquad_commit_batch();

          using LayoutC_TV = typename TiledMma::LayoutC_TV;

          constexpr auto separated =
              mate::attention::fmha::layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
                                                     mute::make_layout(mute::shape<0>(LayoutC_TV{})),
                                                     mute::stride<0>(LayoutC_TV{}));
          constexpr int lanes_per_pos = mute::size(get<1>(separated));

          constexpr int kSubWarpStride = mutlass::NumThreadsPerWarp / lanes_per_pos;

          constexpr int kKVPack = mute::size(get<0>(separated));
          constexpr int kRows   = int(kBlockKV) / kKVPack;

          constexpr int kBurst = 4;
          static_assert(kRows % kBurst == 0);

          const int lane_idx        = thread_idx_in_warp_squad % mutlass::NumThreadsPerWarp;
          const int sub_warp_offset = warp_idx_in_warp_squad * kSubWarpStride;
          const int base_v_offset   = lane_idx / lanes_per_pos;
          const int sfk_lane        = thread_idx_in_warp_squad / lanes_per_pos;

          float scales_kv[kRows];
          MUTE_UNROLL
          for (int i = 0; i < kRows; ++i) {
            scales_kv[i] = sSFK(i * kKVPack + sfk_lane, _0{}, kv_stage_idx, kv_group_idx);
          }

          mate::warpsquad_wait();

          pipeline_k.consumer_release(pipe_k_read);
          ++pipe_k_read;

          const int64_t q_block_base  = static_cast<int64_t>(task.q_group_idx) * kBlockQ;
          const int64_t kv_block_base = static_cast<int64_t>(kv_block_idx) * kBlockKV + sub_warp_offset;

          MUTE_UNROLL
          for (int row_base = 0; row_base + (kBurst - 1) < kRows; row_base += kBurst) {
            MUTE_UNROLL
            for (int ni = 0; ni < kBlockQ; ++ni) {
              const int64_t q_row = q_block_base + ni;

              f4        accv     = f4{0.f, 0.f, 0.f, 0.f};
              const int col_base = ni * J;

              MUTE_UNROLL
              for (int j = 0; j < J; ++j) {
                const float w   = weights(ni, j);
                const int   col = col_base + j;
                f4          v   = f4{
                    accum_mn(row_base + 0, col),
                    accum_mn(row_base + 1, col),
                    accum_mn(row_base + 2, col),
                    accum_mn(row_base + 3, col),
                };
                v    = bst4_relu(v);
                accv = bst4_fma_svv(w, v, accv);
              }

              const f4 sc = f4{
                  scales_kv[row_base + 0],
                  scales_kv[row_base + 1],
                  scales_kv[row_base + 2],
                  scales_kv[row_base + 3],
              };

              accv    = bst4_mul_vv(sc, accv);
              f4 sum4 = warp_group_reduce_sum_f4<reduction_target>(accv);

              float* out_row = params.ptr_logits + q_row * stride_row;

              const int64_t kv_base = kv_block_base + base_v_offset + static_cast<int64_t>(row_base) * kKVPack;

              const float s0 = sum4[0];
              const float s1 = sum4[1];
              const float s2 = sum4[2];
              const float s3 = sum4[3];

              if constexpr (!kIsCompressedLogits) {
                float* p       = out_row + kv_base;
                p[0]           = s0;
                p[1 * kKVPack] = s1;
                p[2 * kKVPack] = s2;
                p[3 * kKVPack] = s3;
              } else {
                const int32_t ks_row  = __ldg(params.ptr_ks + q_row);
                const int64_t lc_base = kv_base - static_cast<int64_t>(ks_row);

                const int64_t lc0 = lc_base + 0;
                const int64_t lc1 = lc_base + kKVPack;
                const int64_t lc2 = lc_base + 2 * kKVPack;
                const int64_t lc3 = lc_base + 3 * kKVPack;

                if (lc0 >= 0 && lc3 < params.max_seq_kv) {
                  float* p       = out_row + lc0;
                  p[0]           = s0;
                  p[1 * kKVPack] = s1;
                  p[2 * kKVPack] = s2;
                  p[3 * kKVPack] = s3;
                } else {
                  if (lc0 >= 0 && lc0 < params.max_seq_kv) out_row[lc0] = s0;
                  if (lc1 >= 0 && lc1 < params.max_seq_kv) out_row[lc1] = s1;
                  if (lc2 >= 0 && lc2 < params.max_seq_kv) out_row[lc2] = s2;
                  if (lc3 >= 0 && lc3 < params.max_seq_kv) out_row[lc3] = s3;
                }
              }
            }
          }
        }

        pipeline_q.consumer_release(pipe_q_read);
        ++pipe_q_read;
      }
    }
  }
};

}  // namespace mate::deep_gemm
