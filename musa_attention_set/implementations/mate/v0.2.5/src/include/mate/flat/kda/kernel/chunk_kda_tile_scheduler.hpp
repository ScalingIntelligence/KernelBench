#pragma once

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

namespace mate::flat::kda {

namespace detail {

MUTLASS_HOST_DEVICE constexpr int grouped_head_idx(int head_idx, int dst_heads, int src_heads) {
  return head_idx / (dst_heads / src_heads);
}

}  // namespace detail

template <bool IsVarlen_, class TileShape_>
struct ChunkKdaWorkDesc {
  using TileShape                = TileShape_;
  static constexpr bool IsVarlen = IsVarlen_;

  static constexpr int kChunk = decltype(mute::get<0>(TileShape{}))::value;

  int64_t bos;
  int     seq_idx;
  int     head_idx;
  int     qk_head_idx;
  int64_t seq_len;
  int     n_chunks;

  MUTLASS_DEVICE bool is_valid() const {
    return seq_idx >= 0 && head_idx >= 0;
  }

  template <class Scheduler>
  MUTLASS_DEVICE bool is_valid(Scheduler const& scheduler) const {
    return is_valid() && seq_idx < scheduler.num_seqs && head_idx < scheduler.num_heads;
  }

  MUTLASS_DEVICE int64_t chunk_start(int chunk_idx) const {
    return int64_t(chunk_idx) * kChunk;
  }

  MUTLASS_DEVICE int actual_len(int chunk_idx) const {
    return min(kChunk, int(seq_len - chunk_start(chunk_idx)));
  }

  MUTLASS_DEVICE int64_t tme_token(int chunk_idx) const {
    if constexpr (IsVarlen) {
      return bos + chunk_start(chunk_idx);
    } else {
      return chunk_start(chunk_idx);
    }
  }

  MUTLASS_DEVICE int tme_batch() const {
    if constexpr (IsVarlen) {
      return 0;
    } else {
      return seq_idx;
    }
  }

  MUTLASS_DEVICE int64_t token(int chunk_idx, int row) const {
    return bos + chunk_start(chunk_idx) + row;
  }
};

template <class TileShape_, class CuSeqlensElement_ = int64_t, class... Options_>
struct ChunkKdaTileScheduler {
  using TileShape        = TileShape_;
  using CuSeqlensElement = CuSeqlensElement_;
  static constexpr bool IsVarlen =
      mate::flat::find_option_t<mate::flat::Tag::IsVarlen, std::false_type, Options_...>::value;
  using WorkDesc = ChunkKdaWorkDesc<IsVarlen, TileShape>;
  static_assert(std::is_integral_v<CuSeqlensElement>);

  static constexpr int kChunk = decltype(mute::get<0>(TileShape{}))::value;

  struct Params {
    dim3 grid;
  };

  int  seq_idx;
  int  head_idx;
  int  num_seqs;
  int  num_heads;
  bool scheduled = false;

  MUTLASS_DEVICE explicit ChunkKdaTileScheduler(Params const&)
      : seq_idx(int(blockIdx.x)),
        head_idx(int(blockIdx.y)),
        num_seqs(int(gridDim.x)),
        num_heads(int(gridDim.y)),
        scheduled(false) {
  }

  template <class ProblemSize>
  static MUTLASS_HOST_DEVICE Params to_underlying_arguments(ProblemSize const& problem_size) {
    return Params{dim3(problem_size.N, problem_size.H, 1)};
  }

  static MUTLASS_HOST_DEVICE dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  template <class ProblemSize>
  MUTLASS_DEVICE WorkDesc get_next_work(ProblemSize const& problem_size) {
    int64_t bos = 0;
    int64_t eos = 0;
    if constexpr (IsVarlen) {
      bos = int64_t(problem_size.cu_seqlens[seq_idx]);
      eos = int64_t(problem_size.cu_seqlens[seq_idx + 1]);
    } else {
      bos = int64_t(seq_idx) * problem_size.T;
      eos = bos + problem_size.T;
    }
    int64_t seq_len = eos - bos;

    if (scheduled) {
      return WorkDesc{
          bos,
          -1,
          head_idx,
          detail::grouped_head_idx(head_idx, problem_size.H, problem_size.Hqk),
          seq_len,
          0,
      };
    }

    scheduled = true;
    return WorkDesc{
        bos,
        seq_idx,
        head_idx,
        detail::grouped_head_idx(head_idx, problem_size.H, problem_size.Hqk),
        seq_len,
        mutlass::ceil_div(seq_len, kChunk),
    };
  }
};

}  // namespace mate::flat::kda
