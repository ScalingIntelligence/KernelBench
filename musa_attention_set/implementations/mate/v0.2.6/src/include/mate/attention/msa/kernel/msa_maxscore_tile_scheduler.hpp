#pragma once

#include <mutlass/fast_math.h>

#include <algorithm>
#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::attention::msa {

template <class TileShape_, int HeadRatio_, bool ParallelKTiles_>
struct MsaMaxScoreTileScheduler {
  using TileShape = TileShape_;

  static constexpr int  TileQ          = mute::get<0>(TileShape{});
  static constexpr int  TileKV         = mute::get<1>(TileShape{});
  static constexpr int  HeadRatio      = HeadRatio_;
  static constexpr int  QTokensPerTile = TileQ / HeadRatio;
  static constexpr bool ParallelKTiles = ParallelKTiles_;
  static constexpr int  CtasPerMp      = ParallelKTiles ? 3 : 1;

  struct Arguments {
    int32_t const* ptr_cu_seqlens_q = nullptr;
    int32_t const* ptr_cu_seqlens_k = nullptr;
  };

  struct Params {
    int                 total_works;
    int                 total_q_works;
    int                 q_tiles;
    int                 k_tiles;
    int                 k_partitions;
    int                 num_kv_heads;
    int                 batch_size;
    int32_t const*      ptr_cu_seqlens_q;
    int32_t const*      ptr_cu_seqlens_k;
    mutlass::FastDivmod k_partitions_divmod;
    mutlass::FastDivmod q_tiles_divmod;
    mutlass::FastDivmod head_divmod;
  };

  struct WorkTileInfo {
    int work_idx;
    int q_work_idx;
    int batch_idx;
    int q_tile_idx;
    int k_partition_idx;
    int k_tile_idx;
    int head_kv;
    int q_batch_begin;
    int k_batch_begin;
    int q_len;
    int kv_len;
    int q_local_begin;
    int q_abs_begin;
    int q_count;
    int k_tile_begin;
    int valid_k_tiles;

    MUTLASS_DEVICE bool is_valid(Params const& params) const {
      return work_idx < params.total_works;
    }
  };

  static MUTLASS_DEVICE mute::tuple<int, int, int> decode_q_work_coord(Params const& params, int q_work_idx) {
    int q_tile_idx = 0;
    int head_kv    = 0;
    int tile       = params.q_tiles_divmod.divmod(q_tile_idx, q_work_idx);
    int batch_idx  = params.head_divmod.divmod(head_kv, tile);
    return {q_tile_idx, head_kv, batch_idx};
  }

  static MUTLASS_DEVICE void fill_q_meta(Params const& params, WorkTileInfo& work_info) {
    work_info.q_batch_begin = params.ptr_cu_seqlens_q[work_info.batch_idx];
    int q_end               = params.ptr_cu_seqlens_q[work_info.batch_idx + 1];
    work_info.q_len         = q_end - work_info.q_batch_begin;
    work_info.q_local_begin = work_info.q_tile_idx * QTokensPerTile;
    work_info.q_abs_begin   = work_info.q_batch_begin + work_info.q_local_begin;
    int q_count             = work_info.q_len - work_info.q_local_begin;
    q_count                 = q_count > 0 ? q_count : 0;
    work_info.q_count       = q_count < QTokensPerTile ? q_count : QTokensPerTile;
  }

  static MUTLASS_DEVICE void copy_q_meta(WorkTileInfo const& src, WorkTileInfo& dst) {
    dst.q_batch_begin = src.q_batch_begin;
    dst.q_len         = src.q_len;
    dst.q_local_begin = src.q_local_begin;
    dst.q_abs_begin   = src.q_abs_begin;
    dst.q_count       = src.q_count;
  }

  static MUTLASS_DEVICE void fill_k_meta(Params const& params, WorkTileInfo& work_info) {
    work_info.k_batch_begin = params.ptr_cu_seqlens_k[work_info.batch_idx];
    int k_end               = params.ptr_cu_seqlens_k[work_info.batch_idx + 1];
    work_info.kv_len        = k_end - work_info.k_batch_begin;
    work_info.valid_k_tiles = mutlass::ceil_div(work_info.kv_len, TileKV);
  }

  static MUTLASS_DEVICE void copy_k_meta(WorkTileInfo const& src, WorkTileInfo& dst) {
    dst.k_batch_begin = src.k_batch_begin;
    dst.kv_len        = src.kv_len;
    dst.valid_k_tiles = src.valid_k_tiles;
  }

  static MUTLASS_DEVICE WorkTileInfo make_q_work_tile(Params const& params, int work_idx) {
    WorkTileInfo work_info{};
    work_info.work_idx = work_idx;
    if (!work_info.is_valid(params)) {
      return work_info;
    }

    int q_work_idx      = work_idx;
    int k_partition_idx = 0;
    if constexpr (ParallelKTiles) {
      q_work_idx = params.k_partitions_divmod.divmod(k_partition_idx, work_idx);
    }
    auto [q_tile_idx, head_kv, batch_idx] = decode_q_work_coord(params, q_work_idx);
    work_info.q_work_idx                  = q_work_idx;
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.k_partition_idx             = k_partition_idx;
    work_info.k_tile_idx                  = 0;
    work_info.head_kv                     = head_kv;
    work_info.k_tile_begin                = 0;
    fill_q_meta(params, work_info);
    fill_k_meta(params, work_info);
    return work_info;
  }

  static MUTLASS_DEVICE WorkTileInfo make_next_q_work_tile(Params const& params, WorkTileInfo const& current_work) {
    WorkTileInfo work_info{};
    work_info.work_idx = current_work.work_idx + static_cast<int>(gridDim.x);
    if (!work_info.is_valid(params)) {
      return work_info;
    }

    int q_work_idx      = work_info.work_idx;
    int k_partition_idx = 0;
    if constexpr (ParallelKTiles) {
      q_work_idx = params.k_partitions_divmod.divmod(k_partition_idx, work_info.work_idx);
    }
    auto [q_tile_idx, head_kv, batch_idx] = decode_q_work_coord(params, q_work_idx);
    work_info.q_work_idx                  = q_work_idx;
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.k_partition_idx             = k_partition_idx;
    work_info.k_tile_idx                  = 0;
    work_info.head_kv                     = head_kv;
    work_info.k_tile_begin                = 0;

    if (q_work_idx == current_work.q_work_idx) {
      copy_q_meta(current_work, work_info);
    } else {
      fill_q_meta(params, work_info);
    }

    if (batch_idx == current_work.batch_idx) {
      copy_k_meta(current_work, work_info);
    } else {
      fill_k_meta(params, work_info);
    }
    return work_info;
  }

  static MUTLASS_DEVICE bool is_valid_q_tile(WorkTileInfo const& work_info) {
    return work_info.q_count > 0;
  }

  static MUTLASS_DEVICE bool is_valid_k_partition(WorkTileInfo const& work_info) {
    return work_info.k_partition_idx < work_info.valid_k_tiles;
  }

  static MUTLASS_DEVICE WorkTileInfo make_k_work_tile(WorkTileInfo const& q_work_info, int k_tile_idx) {
    WorkTileInfo work_info = q_work_info;
    work_info.k_tile_idx   = k_tile_idx;
    work_info.k_tile_begin = k_tile_idx * TileKV;
    return work_info;
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args, int mp_count) {
    int q_tiles       = mutlass::ceil_div(problem_size.max_seqlen_q, QTokensPerTile);
    int k_tiles       = mutlass::ceil_div(problem_size.max_seqlen_k, TileKV);
    int total_q_works = q_tiles * problem_size.num_kv_heads * problem_size.batch_size;
    int k_partitions  = 1;
    if constexpr (ParallelKTiles) {
      int q_works      = std::max(total_q_works, 1);
      int target_works = std::max(mp_count, 1) * CtasPerMp;
      k_partitions     = std::min(k_tiles, std::max(1, (target_works + q_works / 2) / q_works));
    }
    int total_works = total_q_works * k_partitions;
    return {
        total_works,
        total_q_works,
        q_tiles,
        k_tiles,
        k_partitions,
        problem_size.num_kv_heads,
        problem_size.batch_size,
        args.ptr_cu_seqlens_q,
        args.ptr_cu_seqlens_k,
        mutlass::FastDivmod(k_partitions),
        mutlass::FastDivmod(q_tiles),
        mutlass::FastDivmod(problem_size.num_kv_heads),
    };
  }

  static dim3 get_grid_shape(Params const& params, int mp_count) {
    int target_ctas = std::max(mp_count, 1) * CtasPerMp;
    // Use a persistent launch for both scheduling modes. Each CTA advances by
    // gridDim.x in get_next_work(), so one CTA per work only adds waves.
    int grid_x = std::max(1, std::min(target_ctas, std::max(params.total_works, 1)));
    return dim3(static_cast<uint32_t>(grid_x), 1, 1);
  }

  MUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    return make_q_work_tile(params, static_cast<int>(blockIdx.x));
  }

  MUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return make_next_q_work_tile(params, current_work);
  }
};

}  // namespace mate::attention::msa
