#pragma once

#include <mutlass/fast_math.h>

#include <mute/tensor.hpp>

#include "utils.hpp"

namespace mate::attention::fmha {

using namespace mute;

template <class SeqlenInfo,
          int  TileM,
          int  TileN,
          int  HeadRatio,
          bool IsCausal,
          bool IsLocal,
          bool IsPackGQA,
          bool HasAttentionChunk,
          bool IsSplitKV = false,
          bool EnableCP  = false>
struct BlockInfo {
  static MUTLASS_DEVICE mute::tuple<int, int> get_n_block_min_max(SeqlenInfo const&          seqlen_info,
                                                                  int const                  m_block,
                                                                  int const                  split_idx,
                                                                  int const                  num_splits,
                                                                  int const                  window_size_left,
                                                                  int const                  window_size_right,
                                                                  mutlass::FastDivmod const& attention_chunk_divmod) {
    int n_block_max = mute::ceil_div(seqlen_info.seqlen_k, TileN);
    if constexpr (IsCausal || IsLocal) {
      uint32_t m_idx_max = (m_block + 1) * TileM;

      if constexpr (IsPackGQA) {
        m_idx_max = mute::ceil_div(m_idx_max, HeadRatio);
      }

      // CP uses tot_seqlen_k (global), non-CP and Local use local seqlen_k
      int const tot_sk      = IsLocal ? static_cast<int>(seqlen_info.seqlen_k) : seqlen_info.tot_seqlen_k;
      int const n_idx       = static_cast<int>(m_idx_max) + tot_sk - static_cast<int>(seqlen_info.seqlen_q);
      int       n_token_max = n_idx + (IsLocal ? window_size_right : 0);
      if constexpr (IsLocal && HasAttentionChunk) {
        int const chunk       = attention_chunk_divmod.divisor;
        int const chunk_right = div_ceil(attention_chunk_divmod, n_idx) * chunk;
        n_token_max           = std::min(n_token_max, chunk_right);
      }

      if constexpr (EnableCP && !IsLocal) {
        // Convert global token limit to local n_block count.
        // Local K index j maps to global K = j * cp_world_size + cp_rank.
        // Number of local tokens where global_k <= n_token_max:
        //   ceil_div(n_token_max - cp_rank + 1, cp_world_size)  ... but vllm uses ceil_div directly:
        //   ceil_div(n_token_max - cp_rank, cp_world_size)
        // Keep consistent with vllm.
        n_token_max = mute::ceil_div(n_token_max - seqlen_info.cp_rank, seqlen_info.cp_world_size);
      }
      n_block_max = std::min(n_block_max, mute::ceil_div(n_token_max, TileN));
    }

    int n_block_min = 0;
    if constexpr (IsLocal) {
      uint32_t m_idx_min = m_block * TileM;

      if constexpr (IsPackGQA) {
        m_idx_min = m_idx_min / HeadRatio;
      }

      int32_t const n_idx =
          static_cast<int>(m_idx_min) + static_cast<int>(seqlen_info.seqlen_k) - static_cast<int>(seqlen_info.seqlen_q);
      int32_t const n_idx_left       = n_idx - window_size_left;
      int32_t       n_idx_left_chunk = n_idx_left;
      if constexpr (HasAttentionChunk) {
        int const chunk      = attention_chunk_divmod.divisor;
        int const chunk_left = div_floor(attention_chunk_divmod, n_idx) * chunk;
        n_idx_left_chunk     = std::max(n_idx_left_chunk, chunk_left);
      }
      n_block_min = std::max(0, n_idx_left_chunk / TileN);
    }
    if constexpr (IsSplitKV) {
      uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
      int      num_splits_dynamic   = reinterpret_cast<int const&>(num_splits_dynamic_u);
      int      split_idx_actual     = split_idx & 0xFFFF;
      int      num_splits_actual    = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
      int      num_n_blocks_per_split =
          n_block_max <= n_block_min ? 0 : mute::ceil_div(n_block_max - n_block_min, num_splits_actual);
      n_block_min += split_idx_actual * num_n_blocks_per_split;
      n_block_max = std::min(n_block_max, n_block_min + num_n_blocks_per_split);
    }
    return {n_block_min, n_block_max};
  }

  static MUTLASS_DEVICE mute::tuple<int, int> get_n_block_k_new_min_max(
      SeqlenInfo const&          seqlen_info,
      int const                  m_block,
      int const                  bidb,
      int const                  split_idx,
      int const                  num_splits,
      int const                  window_size_left,
      int const                  window_size_right,
      mutlass::FastDivmod const& attention_chunk_divmod) {
    auto [n_block_min, n_block_max] = get_n_block_min_max(
        seqlen_info, m_block, split_idx, num_splits, window_size_left, window_size_right, attention_chunk_divmod);
    int const seqlen_k_og     = static_cast<int>(seqlen_info.seqlen_k_og);
    int const seqlen_k_new    = static_cast<int>(seqlen_info.seqlen_k_new);
    int const idx_k_new_min   = std::max(n_block_min * TileN - seqlen_k_og, 0);
    int const idx_k_new_max   = std::min(n_block_max * TileN - seqlen_k_og, seqlen_k_new);
    int const n_block_new_min = idx_k_new_min / TileN;
    int const n_block_new_max = idx_k_new_max > idx_k_new_min ? mute::ceil_div(idx_k_new_max, TileN) : n_block_new_min;

    return {n_block_new_min, n_block_new_max};
  }

  // Returns the first n_block that needs per-element causal/local masking
  // (tiles before this index are fully attended and can skip masking).
  static MUTLASS_DEVICE int get_n_block_min_causal_local_mask(SeqlenInfo const&          seqlen_info,
                                                              int const                  m_block,
                                                              int const                  n_block_min,
                                                              int const                  window_size_right,
                                                              mutlass::FastDivmod const& attention_chunk_divmod) {
    uint32_t m_idx_min = m_block * TileM;
    if constexpr (IsPackGQA) {
      m_idx_min = m_idx_min / HeadRatio;
    }

    int const tot_sk      = IsLocal ? static_cast<int>(seqlen_info.seqlen_k) : seqlen_info.tot_seqlen_k;
    int       n_idx       = static_cast<int>(m_idx_min) + tot_sk - static_cast<int>(seqlen_info.seqlen_q);
    int       n_idx_right = !IsLocal ? n_idx : n_idx + window_size_right;
    if constexpr (IsLocal && HasAttentionChunk) {
      int const chunk       = attention_chunk_divmod.divisor;
      int const chunk_right = div_ceil(attention_chunk_divmod, n_idx) * chunk;
      n_idx_right           = std::min(n_idx_right, chunk_right);
    }

    if constexpr (EnableCP && !IsLocal) {
      // We need the number of valid local tokens first, then convert to tile index.
      // local token j is valid iff: j * cp_world_size + cp_rank <= n_idx_right
      // => j <= floor((n_idx_right - cp_rank) / cp_world_size)
      // so:
      //   num_valid_local = floor((n_idx_right - cp_rank)/cp_world_size) + 1, if >= 0
      int x               = n_idx_right - seqlen_info.cp_rank;
      int num_valid_local = x >= 0 ? (x / seqlen_info.cp_world_size + 1) : 0;
      return std::max(n_block_min, num_valid_local / TileN);
    } else {
      return std::max(n_block_min, n_idx_right / TileN);
    }
  }

  // Returns the first n_block that needs left-window local masking.
  // Tiles at or above this index are in the "no mask" zone.
  static MUTLASS_DEVICE int get_n_block_min_before_local_mask(SeqlenInfo const&          seqlen_info,
                                                              int const                  m_block,
                                                              int const                  n_block_min,
                                                              int const                  window_size_left,
                                                              mutlass::FastDivmod const& attention_chunk_divmod) {
    uint32_t m_idx_max = (m_block + 1) * TileM;
    if constexpr (IsPackGQA) {
      m_idx_max = mute::ceil_div(m_idx_max, HeadRatio);
    }

    int const n_idx =
        static_cast<int>(m_idx_max) + static_cast<int>(seqlen_info.seqlen_k) - static_cast<int>(seqlen_info.seqlen_q);
    int const n_idx_left       = !IsLocal ? n_idx : n_idx - window_size_left;
    int       n_idx_left_chunk = n_idx_left;
    if constexpr (IsLocal && HasAttentionChunk) {
      int const chunk      = attention_chunk_divmod.divisor;
      int const chunk_left = div_floor(attention_chunk_divmod, n_idx) * chunk;
      n_idx_left_chunk     = std::max(n_idx_left_chunk, chunk_left);
    }

    return !IsLocal ? n_block_min : std::max(n_block_min, mute::ceil_div(n_idx_left_chunk, TileN));
  }
};

}  // namespace mate::attention::fmha
