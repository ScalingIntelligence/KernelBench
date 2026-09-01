// Adapted from https://github.com/deepseek-ai/FlashMLA
#pragma once

#include <mutlass/mutlass.h>

#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "../../common/mma_mp31_sqmma.hpp"
#include "mate/attention/fmha/utils.hpp"
#include "mate/gemm/deep_gemm/mp31_fp8_mqa_logits.hpp"

namespace mate::deep_gemm {

using namespace mute;

template <uint32_t kAlignedBatchSize, bool kIsContextLens2D, uint32_t kNextN>
__global__ __launch_bounds__(32, 1) void mpxx_paged_mqa_logits_metadata(const uint32_t  batch_size,
                                                                        const uint32_t* context_lens,
                                                                        uint32_t*       schedule_metadata,
                                                                        const uint32_t  SPLIT_KV,
                                                                        const uint32_t  kNumMPs) {
  static_assert(kAlignedBatchSize % 32 == 0, "Invalid aligned batch size");
  const uint32_t lane_idx = mutlass::canonical_lane_idx();

  uint32_t num_segs[kAlignedBatchSize / 32];
#pragma unroll
  for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++k) {
    const uint32_t q_idx       = k * 32 + lane_idx;
    const uint32_t lens_idx    = kIsContextLens2D ? (q_idx * kNextN + (kNextN - 1)) : q_idx;
    const uint32_t context_len = (q_idx < batch_size ? __ldg(context_lens + lens_idx) : 0);
    num_segs[k]                = ceil_div(context_len, SPLIT_KV);
  }

  __shared__ uint32_t prefix_sum[kAlignedBatchSize];
  uint32_t            sum = 0;
#pragma unroll
  for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++k) {
    uint32_t x = num_segs[k];
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
      const uint32_t& y = __shfl_up_sync(0xffffffff, x, offset);
      x += (lane_idx >= offset ? y : 0);
    }
    x += sum;
    prefix_sum[k * 32 + lane_idx] = x;
    sum                           = __shfl_sync(0xffffffff, x, 31);
  }

  const uint32_t &q = sum / kNumMPs, r = sum % kNumMPs;
  for (uint32_t mp_idx = lane_idx; mp_idx <= kNumMPs; mp_idx += 32) {
    uint32_t seg_starts = mp_idx * q + min(mp_idx, r);
    uint32_t q_idx      = 0;
    while (q_idx < batch_size and prefix_sum[q_idx] <= seg_starts) ++q_idx;
    const uint32_t& kv_split_idx = (q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1]);
    __syncwarp();

    schedule_metadata[mp_idx * 2]     = q_idx;
    schedule_metadata[mp_idx * 2 + 1] = kv_split_idx;
  }
}

template <uint32_t BLOCK_KV, uint32_t kNumMathWarpSquads, uint32_t kNextN, bool kIsContextLens2D>
struct PagedMQALogitsScheduler {
  uint32_t        batch_size;
  const uint32_t* context_lens;

  uint32_t current_q_idx, current_kv_idx;
  uint32_t end_q_idx, end_kv_idx;
  uint32_t current_num_kv;

  __device__ __forceinline__ uint32_t get_num_kv(uint32_t q_idx) const {
    if (q_idx >= batch_size) return 0;
    const uint32_t lens_idx = kIsContextLens2D ? (q_idx * kNextN + (kNextN - 1)) : q_idx;
    return ceil_div(__ldg(context_lens + lens_idx), BLOCK_KV);
  }

  __device__ __forceinline__ explicit PagedMQALogitsScheduler(const uint32_t& batch_size,
                                                              const uint32_t& mp_idx,
                                                              const uint32_t* context_lens,
                                                              const uint32_t* schedule_meta) {
    this->batch_size   = batch_size;
    this->context_lens = context_lens;

    const auto& current_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + mp_idx);
    const auto& end_pack     = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + mp_idx + 1);
    current_q_idx            = current_pack.x;
    current_kv_idx           = current_pack.y * kNumMathWarpSquads;
    end_q_idx                = end_pack.x;
    end_kv_idx               = end_pack.y * kNumMathWarpSquads;

    current_num_kv = get_num_kv(current_q_idx);
  }

  __device__ __forceinline__ bool fetch_next_task(uint32_t& q_idx, uint32_t& kv_idx, uint32_t& num_kv) {
    q_idx  = current_q_idx;
    kv_idx = current_kv_idx;
    num_kv = current_num_kv;

    if (q_idx == end_q_idx and kv_idx == end_kv_idx) return false;

    current_kv_idx += kNumMathWarpSquads;
    if (current_kv_idx >= current_num_kv) {
      ++current_q_idx;
      current_kv_idx = 0;
      current_num_kv = get_num_kv(current_q_idx);
    }
    return true;
  }

  __device__ __forceinline__ bool exist_q_idx(const uint32_t& q_idx) const {
    return q_idx < end_q_idx or (q_idx == end_q_idx and 0 < end_kv_idx);
  }
};

template <uint32_t kNextN_,
          uint32_t kNumHeads_,
          uint32_t kHeadDim_,
          uint32_t BLOCK_KV_,
          bool     kIsContextLens2D_,
          class StrideQ_,
          class StrideK_,
          class StrideSFK_,
          class StrideW_,
          class StrideBlockTable_,
          class StrideLogits_>
struct Mp31Fp8PagedMqaLogits {
  using Element            = float_e4m3_t;
  using ElementAccumulator = float;

  using StrideQ          = StrideQ_;
  using StrideK          = StrideK_;
  using StrideSFK        = StrideSFK_;
  using StrideW          = StrideW_;
  using StrideBlockTable = StrideBlockTable_;
  using StrideLogits     = StrideLogits_;

  static constexpr int  NumLoadWarpSquads = 1;
  static constexpr int  NumMmaWarpSquads  = 4;
  static constexpr int  Alignment         = 32 / sizeof_bits_v<Element>;
  static constexpr int  StagesQ           = 3;
  static constexpr int  StagesK           = 3;
  static constexpr int  NextN             = kNextN_;
  static constexpr int  NumHeads          = kNumHeads_;
  static constexpr int  HeadDim           = kHeadDim_;
  static constexpr int  BLOCK_KV          = BLOCK_KV_;
  static constexpr int  SPLIT_KV          = BLOCK_KV * NumMmaWarpSquads;
  static constexpr bool IsContextLens2D   = kIsContextLens2D_;  // <<< exposed as alias

  using TileShape = Shape<Int<BLOCK_KV>, Int<NextN * NumHeads>, Int<HeadDim>>;

  using CollectiveMma = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                              mutlass::arch::OpClassTensorOp,
                                                                              Element,
                                                                              StrideK,
                                                                              Alignment,
                                                                              Element,
                                                                              StrideQ,
                                                                              Alignment,
                                                                              ElementAccumulator,
                                                                              TileShape,
                                                                              Shape<_1, _1, _1>,
                                                                              Int<StagesQ>,
                                                                              mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMma      = typename CollectiveMma::TiledMma;
  using SmemLayoutQ   = typename CollectiveMma::SmemLayoutB;
  using SmemLayoutK   = decltype(tile_to_shape(typename CollectiveMma::SmemLayoutA{},
                                             Shape<Int<BLOCK_KV * NumMmaWarpSquads>, Int<HeadDim>, Int<StagesK>>{}));
  using SmemLayoutSFK = Layout<Shape<Int<BLOCK_KV>, _1, Int<StagesK>, Int<NumMmaWarpSquads>>>;
  using SmemLayoutW   = Layout<Shape<Int<NextN * NumHeads>, _1, Int<StagesQ>>>;
  using SmemLayoutWView =
      decltype(make_ordered_layout(make_shape(Int<NextN>{}, Int<NumHeads>{}, Int<StagesQ>{}), Step<_1, _0, _2>{}));

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, 256> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, 256> smem_k;
    mute::array_aligned<float, cosize_v<SmemLayoutW>, 256>   smem_weights;
    mute::array_aligned<float, cosize_v<SmemLayoutSFK>, 256> smem_sfk;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  using TME_Q   = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
      take<0, 2>(typename CollectiveMma::SmemLayoutB{})));
  using TME_K   = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideK{}, int32_t(0)), StrideK{}),
      take<0, 2>(typename CollectiveMma::SmemLayoutA{})));
  using TME_SFK = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), repeat_like(StrideSFK{}, int32_t(0)), StrideSFK{}),
      take<0, 2>(SmemLayoutSFK{})));
  using TME_W   = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), repeat_like(StrideW{}, int32_t(0)), StrideW{}),
      take<0, 2>(SmemLayoutW{})));

  using PipelineQ       = mutlass::Mp31PipelineTmeAsync<StagesQ>;
  using PipelineK       = mutlass::Mp31PipelineTmeAsync<StagesK>;
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
    Element const* ptr_q;
    StrideQ        stride_q;
    Element const* ptr_k;
    StrideK        stride_k;
    float const*   ptr_sfk;
    StrideSFK      stride_sfk;
    float const*   ptr_weights;
    StrideW        stride_weights;

    uint32_t const* ptr_context_lens;
    uint32_t const* ptr_schedule_meta;

    int32_t const*   ptr_block_table;
    StrideBlockTable stride_block_table;

    float*       ptr_logits;
    StrideLogits stride_logits;

    int32_t batch_size;
    int32_t num_kv_blocks;
  };

  struct Params {
    TME_Q   tme_q;
    TME_K   tme_k;
    TME_SFK tme_sfk;
    TME_W   tme_w;

    uint32_t const* ptr_context_lens;
    uint32_t const* ptr_schedule_meta;

    int32_t const*   ptr_block_table;
    StrideBlockTable stride_block_table;

    float*       ptr_logits;
    StrideLogits stride_logits;

    int32_t batch_size;
    int32_t num_kv_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    TME_Q tme_q = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_q), make_shape(args.batch_size * NextN * NumHeads, HeadDim), args.stride_q),
        take<0, 2>(typename CollectiveMma::SmemLayoutB{}));

    TME_K tme_k = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_k), make_shape(BLOCK_KV, HeadDim, args.num_kv_blocks), args.stride_k),
        take<0, 2>(typename CollectiveMma::SmemLayoutA{}));

    TME_SFK tme_sfk = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_sfk), make_shape(BLOCK_KV, args.num_kv_blocks), args.stride_sfk),
        take<0, 2>(SmemLayoutSFK{}));

    TME_W tme_w = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(
            make_gmem_ptr(args.ptr_weights), make_shape(NextN * NumHeads, args.batch_size), args.stride_weights),
        take<0, 2>(SmemLayoutW{}));

    return {tme_q,
            tme_k,
            tme_sfk,
            tme_w,
            args.ptr_context_lens,
            args.ptr_schedule_meta,
            args.ptr_block_table,
            args.stride_block_table,
            args.ptr_logits,
            args.stride_logits,
            args.batch_size,
            args.num_kv_blocks};
  }

  MUTLASS_DEVICE
  void operator()(const Params& params, char* smem) {
    enum class WarpSquadRole {
      Producer  = 0,
      Consumer0 = 1,
      Consumer1 = 2,
      Consumer2 = 3,
      Consumer3 = 4,
    };

    int thread_idx               = threadIdx.x;
    int warp_idx                 = mutlass::canonical_warp_idx_sync();
    int lane_idx                 = thread_idx % mutlass::NumThreadsPerWarp;
    int warp_squad_idx           = mutlass::canonical_warp_squad_idx();
    int warp_idx_in_warp_squad   = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int consumer_warp_squad_idx  = warp_squad_idx - static_cast<int>(WarpSquadRole::Consumer0);
    int thread_idx_in_warp_squad = thread_idx % mutlass::NumThreadsPerWarpSquad;

    auto warp_squad_role = WarpSquadRole(warp_squad_idx);

    int const kv_group_idx =
        warp_squad_role == WarpSquadRole::Producer ? warp_idx_in_warp_squad : consumer_warp_squad_idx;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineQParams pipeline_params_q;
    pipeline_params_q.transaction_bytes = TmeTransactionBytesQAndWeight;
    pipeline_params_q.num_consumers     = NumMmaWarps;
    pipeline_params_q.num_producers     = 1;

    PipelineQ      pipeline_q(pipeline_params_q, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    PipelineQState pipeline_q_producer_state = mutlass::make_producer_start_state<PipelineQ>();
    PipelineQState pipeline_q_consumer_state;

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

    using Scheduler = PagedMQALogitsScheduler<BLOCK_KV, NumMmaWarpSquads, NextN, IsContextLens2D>;
    auto scheduler  = Scheduler(params.batch_size, blockIdx.x, params.ptr_context_lens, params.ptr_schedule_meta);

    uint32_t q_iter_idx  = 0;
    uint32_t kv_iter_idx = 0;

    if (warp_squad_role == WarpSquadRole::Producer) {
      auto cta_tme_q   = params.tme_q.get_slice(0);
      auto cta_tme_k   = params.tme_k.get_slice(0);
      auto cta_tme_sfk = params.tme_sfk.get_slice(0);
      auto cta_tme_w   = params.tme_w.get_slice(0);

      Tensor mQ      = params.tme_q.get_tme_tensor(make_shape(params.batch_size * NextN * NumHeads, HeadDim));
      Tensor gQ_full = local_tile(mQ, select<1, 2>(TileShape{}), make_coord(_, _));
      Tensor sQ      = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
      Tensor tQgQ    = cta_tme_q.partition_S(gQ_full(_, _, _, _0{}));
      Tensor tQsQ    = cta_tme_q.partition_D(sQ);

      Tensor mK      = params.tme_k.get_tme_tensor(make_shape(BLOCK_KV, HeadDim, params.num_kv_blocks));
      Tensor gK_full = local_tile(mK, select<0, 2>(TileShape{}), make_coord(_, _, _));
      Tensor sK      = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
      Tensor tKgK    = cta_tme_k.partition_S(gK_full(_, _, _0{}, _0{}, _));
      Tensor tKsK    = cta_tme_k.partition_D(sK(make_coord(_, kv_group_idx), _, _));

      Tensor mSFK      = params.tme_sfk.get_tme_tensor(make_shape(BLOCK_KV, params.num_kv_blocks));
      Tensor gSFK_full = local_tile(mSFK, Shape<Int<BLOCK_KV>, _1>{}, make_coord(_, _));
      Tensor sSFK      = make_tensor(make_smem_ptr(shared_storage.smem_sfk.data()), SmemLayoutSFK{});
      Tensor tSFKgSFK  = cta_tme_sfk.partition_S(gSFK_full(_, _, _0{}, _));
      Tensor tSFKsSFK  = cta_tme_sfk.partition_D(sSFK(_, _, _, kv_group_idx));

      Tensor mW      = params.tme_w.get_tme_tensor(make_shape(NextN * NumHeads, params.batch_size));
      Tensor gW_full = local_tile(mW, Shape<Int<NextN * NumHeads>, _1>{}, make_coord(_, _));
      Tensor sW      = make_tensor(make_smem_ptr(shared_storage.smem_weights.data()), SmemLayoutW{});
      Tensor tWgW    = cta_tme_w.partition_S(gW_full(_, _, _0{}, _));
      Tensor tWsW    = cta_tme_w.partition_D(sW);

      const auto& issue_tme_q = [&](const uint32_t& q_idx) {
        if (kv_group_idx == 0) {
          pipeline_q.producer_acquire(pipeline_q_producer_state);
          uint32_t bar_id = pipeline_q.producer_get_barrier_id(pipeline_q_producer_state);
          copy(params.tme_q.with(bar_id), tQgQ(_, _, _, q_idx), tQsQ(_, _, _, pipeline_q_producer_state.index()));
          copy(params.tme_w.with(bar_id), tWgW(_, _, _, q_idx), tWsW(_, _, _, pipeline_q_producer_state.index()));
          ++pipeline_q_producer_state;
        }
      };

      uint32_t q_idx = params.batch_size, kv_idx, num_kv;
      uint32_t next_q_idx, next_kv_idx, next_num_kv;
      bool     fetched_next_task;

      if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv))) {
        issue_tme_q(next_q_idx);
        q_iter_idx = 1;
      }

      int      kv_block_idx_ptr = 32;
      uint32_t kv_block_idx_storage;

      while (fetched_next_task) {
        bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_idx(next_q_idx + 1));

        q_idx  = next_q_idx;
        kv_idx = next_kv_idx;
        num_kv = next_num_kv;

        if (prefetch_q) {
          issue_tme_q(q_idx + 1);
          q_iter_idx++;
        }

        if (kv_idx == 0 or kv_block_idx_ptr == 32) {
          kv_block_idx_ptr     = 0;
          kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * NumMmaWarpSquads < num_kv
                                      ? __ldg(params.ptr_block_table + q_idx * get<0>(params.stride_block_table) +
                                              (kv_idx + kv_group_idx + lane_idx * NumMmaWarpSquads))
                                      : 0);
        }
        const auto& kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr++);

        {
          pipeline_k.producer_acquire(pipeline_k_producer_state);
          uint32_t bar_id = pipeline_k.producer_get_barrier_id(pipeline_k_producer_state);
          copy(
              params.tme_k.with(bar_id), tKgK(_, _, _, kv_block_idx), tKsK(_, _, _, pipeline_k_producer_state.index()));
          copy(params.tme_sfk.with(bar_id),
               tSFKgSFK(_, _, _, kv_block_idx),
               tSFKsSFK(_, _, _, pipeline_k_producer_state.index()));
          ++pipeline_k_producer_state;
        }

        fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
      }
    } else if (warp_squad_role == WarpSquadRole::Consumer0 || warp_squad_role == WarpSquadRole::Consumer1 ||
               warp_squad_role == WarpSquadRole::Consumer2 || warp_squad_role == WarpSquadRole::Consumer3) {
      TiledMma tiled_mma;
      auto     thr_mma = tiled_mma.get_thread_slice(thread_idx_in_warp_squad);

      constexpr int reduction_target = size(mate::attention::fmha::reduction_target_n(tiled_mma));

      Tensor weights = make_tensor<float>(Shape<Int<NextN>, Int<NumHeads / reduction_target>>{});

      Tensor sW   = make_tensor(make_smem_ptr(shared_storage.smem_weights.data()), SmemLayoutWView{});
      Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_sfk.data()), SmemLayoutSFK{});

      uint32_t q_idx = params.batch_size, kv_idx, num_kv;
      uint32_t next_q_idx, next_kv_idx, next_num_kv;

      Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
      Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});

      Tensor tQsQ = thr_mma.partition_B(sQ);
      Tensor tQrQ = thr_mma.make_fragment_B(tQsQ);

      Tensor tKsK = thr_mma.partition_A(sK(make_coord(_, kv_group_idx), _, _));
      Tensor tKrK = thr_mma.make_fragment_A(tKsK);

      while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
        Tensor accum    = partition_fragment_C(TiledMma{}, take<0, 2>(TileShape{}));
        Tensor accum_mn = make_tensor(accum.data(), mate::attention::fmha::layout_acc_mn(tiled_mma, accum.layout()));

        clear(accum);

        if (q_idx != next_q_idx) {
          if (q_iter_idx > 0) {
            pipeline_q.consumer_release(pipeline_q_consumer_state);
            ++pipeline_q_consumer_state;
          }
          pipeline_q.consumer_wait(pipeline_q_consumer_state);

          MUTE_UNROLL
          for (int i = 0; i < NextN; ++i) {
            MUTE_UNROLL
            for (int j = 0; j < NumHeads / reduction_target; ++j) {
              weights(i, j) =
                  sW(i, j * reduction_target + lane_idx % reduction_target, pipeline_q_consumer_state.index());
            }
          }
          ++q_iter_idx;
        }

        q_idx  = next_q_idx;
        kv_idx = next_kv_idx;

        pipeline_k.consumer_wait(pipeline_k_consumer_state);
        uint32_t kv_stage_idx = pipeline_k_consumer_state.index();
        mute::gemm(tiled_mma, tKrK(_, _, _, kv_stage_idx), tQrQ(_, _, _, pipeline_q_consumer_state.index()), accum);
        mate::warpsquad_commit_batch();

        static_assert(BLOCK_KV == 64);
        using LayoutC_TV = typename TiledMma::LayoutC_TV;

        constexpr auto separated =
            mate::attention::fmha::layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
                                                   mute::make_layout(mute::shape<0>(LayoutC_TV{})),
                                                   mute::stride<0>(LayoutC_TV{}));
        constexpr int lanes_per_pos = mute::size(get<1>(separated));

        constexpr int kSubWarpStride = mutlass::NumThreadsPerWarp / lanes_per_pos;
        constexpr int kKVPack        = mute::size(get<0>(separated));
        constexpr int kRows          = int(BLOCK_KV) / kKVPack;
        constexpr int kBurst         = 4;
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

        pipeline_k.consumer_release(pipeline_k_consumer_state);
        ++pipeline_k_consumer_state;

        const int64_t kv_block_base = static_cast<int64_t>(kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset;
        const int64_t stride_row    = get<0>(params.stride_logits);
        const int     J             = int(NumHeads / reduction_target);

        MUTE_UNROLL
        for (int row_base = 0; row_base + (kBurst - 1) < kRows; row_base += kBurst) {
          MUTE_UNROLL
          for (int ni = 0; ni < NextN; ++ni) {
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

            const int64_t row_idx = static_cast<int64_t>(q_idx) * NextN + ni;
            float*        out_row = params.ptr_logits + row_idx * stride_row;

            const int64_t kv_base = kv_block_base + base_v_offset + static_cast<int64_t>(row_base) * kKVPack;
            float*        p       = out_row + kv_base;
            p[0]                  = sum4[0];
            p[1 * kKVPack]        = sum4[1];
            p[2 * kKVPack]        = sum4[2];
            p[3 * kKVPack]        = sum4[3];
          }
        }
      }
    }
  }
};

}  // namespace mate::deep_gemm
