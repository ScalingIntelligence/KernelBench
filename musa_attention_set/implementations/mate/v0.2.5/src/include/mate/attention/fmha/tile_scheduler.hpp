#pragma once

#include <cstdint>

namespace mate::attention::fmha {

struct TileSchedulerArguments {
  int32_t const         num_blocks;
  int32_t const         num_head;
  int32_t const         num_batch;
  int32_t const         num_splits;
  int32_t const         qhead_per_khead;
  int32_t const         seqlen_q;
  int32_t const         seqlen_k;
  int32_t const         headdim;
  int32_t const         headdim_v;
  int32_t const         element_size;
  uint32_t const* const cu_seqlens = nullptr;
  uint32_t const* const seqused    = nullptr;

  // For Split-KV
  int32_t const* const num_splits_dynamic_ptr = nullptr;
  int32_t const* const batch_table_ptr        = nullptr;
  int32_t const* const num_m_blocks_ptr       = nullptr;
};

template <bool HasCuseqlensQ, bool HasSequsedQ, bool PackGQA, int TileM, bool IsSplit>
struct SingleTileScheduler {
  using Arguments                   = TileSchedulerArguments;
  static constexpr bool HasMetadata = false;

  struct Params {
    int32_t const         num_blocks;
    int32_t const         num_head;
    int32_t const         num_batch;
    int32_t const         num_splits;
    int32_t const         qhead_per_khead;
    mutlass::FastDivmod   nsplits_divmod;
    int32_t               seqlen_q;
    uint32_t const* const cu_seqlens;
    uint32_t const* const seqused;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    return Params{.num_blocks      = args.num_blocks,
                  .num_head        = args.num_head,
                  .num_batch       = args.num_batch,
                  .num_splits      = args.num_splits,
                  .qhead_per_khead = args.qhead_per_khead,
                  .nsplits_divmod  = mutlass::FastDivmod(!IsSplit ? 1 : args.num_splits),
                  .seqlen_q        = args.seqlen_q,
                  .cu_seqlens      = args.cu_seqlens,
                  .seqused         = args.seqused};
  }

  static dim3 get_grid_shape(Params const& params, int num_mp) {
    return {static_cast<uint32_t>(params.num_blocks),
            static_cast<uint32_t>((IsSplit ? 1 : params.num_splits) * params.num_head),
            static_cast<uint32_t>(params.num_batch)};
  }

  struct WorkTileInfo {
    int32_t block_idx = 0;
    int32_t head_idx  = 0;
    int32_t batch_idx = 0;
    int32_t split_idx = 0;

    MUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return batch_idx >= 0;
    }

    MUTLASS_DEVICE
    mute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(Params const& params) const {
      return {block_idx, head_idx, batch_idx, !IsSplit ? 0 : split_idx};
    }
  };

  MUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const {
    WorkTileInfo work_info{
        static_cast<int32_t>(blockIdx.x), static_cast<int32_t>(blockIdx.y), static_cast<int32_t>(blockIdx.z), 0};

    if constexpr (IsSplit) {
      int32_t split_idx;
      work_info.batch_idx = params.nsplits_divmod.divmod(split_idx, work_info.batch_idx);
      work_info.split_idx = split_idx;
    }
    bool is_valid_tile = true;
    if constexpr (HasCuseqlensQ || HasSequsedQ) {
      uint32_t seqlen = HasCuseqlensQ
                            ? (params.cu_seqlens[work_info.batch_idx + 1] - params.cu_seqlens[work_info.batch_idx])
                        : HasSequsedQ ? params.seqused[work_info.batch_idx]
                                      : params.seqlen_q;
      if constexpr (PackGQA) {
        seqlen *= params.qhead_per_khead;
      }
      is_valid_tile = work_info.block_idx * TileM < seqlen;
    }
    work_info.batch_idx = is_valid_tile ? work_info.batch_idx : -1;
    return work_info;
  }

  MUTLASS_DEVICE
  WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {0, 0, -1, 0};
  }
};

template <bool IsSplit = false>
struct StaticPersistentTileScheduler {
 protected:
  using SharedStorage = int;

 public:
  using Arguments                   = TileSchedulerArguments;
  static constexpr bool HasMetadata = true;

  struct Params {
    int32_t             total_blocks;    // Total number of tiles
    mutlass::FastDivmod m_block_divmod;  // num_blocks
    mutlass::FastDivmod head_divmod;     // num_head
    mutlass::FastDivmod nsplits_divmod;  // num_splits
    int32_t const*      num_splits_dynamic_ptr;
    int32_t const*      batch_table_ptr;
    int32_t const*      num_m_blocks_ptr;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    int total_blocks = args.num_blocks * args.num_head * args.num_batch * (!IsSplit ? 1 : args.num_splits);
    // std::cout << "total_blocks: " << total_blocks << ", num_blocks: " << args.num_blocks;
    // std::cout << ", num_head: " << args.num_head << ", num_batch: " << args.num_batch;
    // std::cout << ", num_splits: " << args.num_splits << std::endl;
    return Params{
        .total_blocks   = total_blocks,
        .m_block_divmod = mutlass::FastDivmod(args.num_blocks),
        .head_divmod    = mutlass::FastDivmod(args.num_head * (!IsSplit ? 1 : args.num_splits)),
        .nsplits_divmod = mutlass::FastDivmod(!IsSplit ? 1 : args.num_splits),

        .num_splits_dynamic_ptr = args.num_splits_dynamic_ptr,
        .batch_table_ptr        = args.batch_table_ptr,
        .num_m_blocks_ptr       = args.num_m_blocks_ptr,
    };
  }

  static dim3 get_grid_shape(Params const& params, int num_mp) {
    return {static_cast<uint32_t>(num_mp)};
  }

  struct WorkTileInfo {
    int32_t tile_idx;

    MUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    MUTLASS_DEVICE
    mute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(Params const& params) const {
      int32_t block_idx, head_idx, batch_idx, split_idx = 0;
      // batch_idx = params.m_block_divmod.divmod(block_idx, params.head_divmod.divmod(head_idx, tile_idx));
      batch_idx = params.head_divmod.divmod(head_idx, params.m_block_divmod.divmod(block_idx, tile_idx));
      if constexpr (IsSplit) {
        head_idx = params.nsplits_divmod.divmod(split_idx, head_idx);
        // block_idx = params.nsplits_divmod.divmod(split_idx, block_idx); // TODO: WARN:is this correct???
      }
      return {block_idx, head_idx, batch_idx, split_idx};
    }
  };

  MUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const {
    return {static_cast<int32_t>(blockIdx.x)};
  }

  MUTLASS_DEVICE
  void init_consumer() const {
  }

  MUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
  }

  MUTLASS_DEVICE
  WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + static_cast<int32_t>(gridDim.x)};

    // int round_idx    = current_work.tile_idx / static_cast<int32_t>(gridDim.x) + 1;
    // int is_odd_round = round_idx & 1;

    // if (is_odd_round) {
    //   return {static_cast<int32_t>((round_idx + 1) * gridDim.x - blockIdx.x - 1)};
    // } else {
    //   return {current_work.tile_idx + static_cast<int32_t>(gridDim.x)};
    // }
  }
};

}  // namespace mate::attention::fmha
