#pragma once

#include <mutlass/fast_math.h>

#include <algorithm>
#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::attention::msa {

// Two adjacent query tokens share one physical M16 SQMMA tile.  This
// scheduler intentionally keeps pairs inside one varlen request; the last
// tile of an odd-length request carries q_count == 1.
struct MsaFwdPairTileScheduler {
  static constexpr int QTokensPerTile = 2;
  static constexpr int CtasPerMp      = 4;

  struct Arguments {
    int32_t const* ptr_cu_seqlens_q = nullptr;
    int32_t const* ptr_seqused_k    = nullptr;
    int32_t const* ptr_qo_offset    = nullptr;
  };

  struct Params {
    int                 total_works;
    int                 q_tiles;
    int                 num_kv_heads;
    int                 batch_size;
    int32_t const*      ptr_cu_seqlens_q;
    int32_t const*      ptr_seqused_k;
    int32_t const*      ptr_qo_offset;
    mutlass::FastDivmod q_tiles_divmod;
    mutlass::FastDivmod head_divmod;
  };

  struct WorkTileInfo {
    int work_idx;
    int batch_idx;
    int q_tile_idx;
    int head_kv;
    int q_batch_begin;
    int q_len;
    int kv_len;
    int qo_offset;
    int q_local;
    int q_abs;
    int q_count;

    MUTLASS_DEVICE bool is_valid(Params const& params) const {
      return work_idx < params.total_works;
    }
  };

  static MUTLASS_DEVICE mute::tuple<int, int, int> decode_work_coord(Params const& params, int work_idx) {
    int q_tile_idx = 0;
    int head_kv    = 0;
    int tile       = params.q_tiles_divmod.divmod(q_tile_idx, work_idx);
    int batch_idx  = params.head_divmod.divmod(head_kv, tile);
    return {q_tile_idx, head_kv, batch_idx};
  }

  static MUTLASS_DEVICE void fill_query_meta(Params const& params, WorkTileInfo& work_info) {
    work_info.q_batch_begin = params.ptr_cu_seqlens_q[work_info.batch_idx];
    int q_end               = params.ptr_cu_seqlens_q[work_info.batch_idx + 1];
    work_info.q_len         = q_end - work_info.q_batch_begin;
    work_info.q_local       = work_info.q_tile_idx * QTokensPerTile;
    work_info.q_abs         = work_info.q_batch_begin + work_info.q_local;
    work_info.q_count       = max(0, min(QTokensPerTile, work_info.q_len - work_info.q_local));
  }

  static MUTLASS_DEVICE void fill_key_meta(Params const& params, WorkTileInfo& work_info) {
    work_info.kv_len    = params.ptr_seqused_k[work_info.batch_idx];
    work_info.qo_offset = params.ptr_qo_offset[work_info.batch_idx];
  }

  static MUTLASS_DEVICE void copy_query_meta(WorkTileInfo const& source, WorkTileInfo& destination) {
    destination.q_batch_begin = source.q_batch_begin;
    destination.q_len         = source.q_len;
    destination.q_local       = source.q_local;
    destination.q_abs         = source.q_abs;
    destination.q_count       = source.q_count;
  }

  static MUTLASS_DEVICE void copy_key_meta(WorkTileInfo const& source, WorkTileInfo& destination) {
    destination.kv_len    = source.kv_len;
    destination.qo_offset = source.qo_offset;
  }

  static MUTLASS_DEVICE WorkTileInfo make_work_tile(Params const& params, int work_idx) {
    WorkTileInfo work_info{};
    work_info.work_idx = work_idx;
    if (!work_info.is_valid(params)) {
      return work_info;
    }
    auto [q_tile_idx, head_kv, batch_idx] = decode_work_coord(params, work_idx);
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.head_kv                     = head_kv;
    fill_query_meta(params, work_info);
    fill_key_meta(params, work_info);
    return work_info;
  }

  static MUTLASS_DEVICE WorkTileInfo make_next_work_tile(Params const& params, WorkTileInfo const& current_work) {
    WorkTileInfo work_info{};
    work_info.work_idx = current_work.work_idx + static_cast<int>(gridDim.x);
    if (!work_info.is_valid(params)) {
      return work_info;
    }

    auto [q_tile_idx, head_kv, batch_idx] = decode_work_coord(params, work_info.work_idx);
    work_info.batch_idx                   = batch_idx;
    work_info.q_tile_idx                  = q_tile_idx;
    work_info.head_kv                     = head_kv;

    if (batch_idx == current_work.batch_idx && q_tile_idx == current_work.q_tile_idx) {
      copy_query_meta(current_work, work_info);
    } else {
      fill_query_meta(params, work_info);
    }
    if (batch_idx == current_work.batch_idx) {
      copy_key_meta(current_work, work_info);
    } else {
      fill_key_meta(params, work_info);
    }
    return work_info;
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    int q_tiles     = (problem_size.max_seqlen_q + QTokensPerTile - 1) / QTokensPerTile;
    int total_works = q_tiles * problem_size.num_kv_heads * problem_size.batch_size;
    return {
        total_works,
        q_tiles,
        problem_size.num_kv_heads,
        problem_size.batch_size,
        args.ptr_cu_seqlens_q,
        args.ptr_seqused_k,
        args.ptr_qo_offset,
        mutlass::FastDivmod(q_tiles),
        mutlass::FastDivmod(problem_size.num_kv_heads),
    };
  }

  template <class ProblemSize>
  static bool can_implement(ProblemSize const& problem_size, Arguments const& args) {
    return problem_size.total_q == 0 ||
           (problem_size.batch_size > 0 && problem_size.max_seqlen_q > 0 && args.ptr_cu_seqlens_q != nullptr &&
            args.ptr_seqused_k != nullptr && args.ptr_qo_offset != nullptr);
  }

  static dim3 get_grid_shape(Params const& params, int mp_count) {
    int target_ctas = std::max(mp_count, 1) * CtasPerMp;
    int grid_x      = std::max(1, std::min(target_ctas, std::max(params.total_works, 1)));
    return dim3(static_cast<uint32_t>(grid_x), 1, 1);
  }

  MUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    return make_work_tile(params, static_cast<int>(blockIdx.x));
  }

  MUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return make_next_work_tile(params, current_work);
  }

  static MUTLASS_DEVICE bool is_valid_query(WorkTileInfo const& work_info) {
    return work_info.q_count > 0;
  }
};

}  // namespace mate::attention::msa
