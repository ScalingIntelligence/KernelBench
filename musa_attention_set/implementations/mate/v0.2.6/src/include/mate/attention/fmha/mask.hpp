#pragma once

#include <mutlass/fast_math.h>

#include <mute/tensor.hpp>

#include "utils.hpp"

namespace mate::attention::fmha {

using namespace mute;

namespace detail {

template <class TileMN, class ThrMMA>
MUTLASS_HOST_DEVICE auto make_identity_tensor_acc_qk(TileMN const& tile_mn, ThrMMA const& thr_mma) {
  Tensor cS      = make_identity_tensor(TileMN{});
  Tensor tScS    = thr_mma.partition_C(cS);
  Tensor tScS_mn = make_tensor(tScS.data(), layout_acc_mn(thr_mma, tScS.layout()));

  return tScS_mn;
}

}  // namespace detail

template <class TiledMmaQK,
          class SeqlenInfo,
          int  TileM,
          int  TileN,
          int  HeadRatio,
          bool IsPackGQA,
          bool IsCausal,
          bool IsLocal,
          bool HasAttentionChunk,
          bool EnableCP = false>
struct Mask {
  static_assert(!(IsCausal && IsLocal), "IsCausal and IsLocal cannot be both true");

  using Element = float;
  using TileMN  = Shape<Int<TileM>, Int<TileN>>;

  using ThrMMAQK  = decltype(TiledMmaQK{}.get_thread_slice(0));
  using ThrMMAQK0 = decltype(TiledMmaQK{}.get_thread_slice(_0{}));

  using IndexTensor = decltype(detail::make_identity_tensor_acc_qk(TileMN{}, ThrMMAQK{}));

  int                 seqlen_k_limit_base;
  int                 seqlen_q;
  int                 seqlen_k;
  int                 causal_limit_base;
  int                 local_limit_left_base;
  int                 local_limit_right_base;
  int                 sink_limit_base;
  mutlass::FastDivmod attention_chunk_divmod;

  // CP fields, read from seqlen_info at construction time
  int thread_col_offset;
  int cp_world_size;
  int cp_rank;
  int cp_kq_diff;    // = tot_seqlen_k - seqlen_q, used in CP causal check
  int tot_seqlen_k;  // global seqlen_k across all CP ranks (for padding mask)

  int mma_thread_idx;

  ThrMMAQK    thr_mma_qk;
  IndexTensor tScS_mn;

  MUTLASS_DEVICE
  Mask(int const                  mma_thread_idx_,
       SeqlenInfo const&          seqlen_info,
       int const                  window_size_left,
       int const                  window_size_right,
       int const                  sink_token_length,
       mutlass::FastDivmod const& attention_chunk_divmod_) {
    auto thr_mma_qk_ = TiledMmaQK{}.get_thread_slice(mma_thread_idx_);

    Tensor tScS_mn_ = detail::make_identity_tensor_acc_qk(TileMN{}, thr_mma_qk_);

    thread_col_offset = get<1>(tScS_mn_(_0{}, _0{}));

    seqlen_q            = seqlen_info.seqlen_q;
    seqlen_k            = seqlen_info.seqlen_k;
    seqlen_k_limit_base = seqlen_k - thread_col_offset;

    causal_limit_base = seqlen_k - seqlen_q - thread_col_offset + 1;

    local_limit_left_base  = causal_limit_base - window_size_left - 1;
    local_limit_right_base = causal_limit_base + window_size_right;

    sink_limit_base        = sink_token_length;
    attention_chunk_divmod = attention_chunk_divmod_;

    // CP: store fields from seqlen_info for use in apply()
    cp_world_size = seqlen_info.cp_world_size;
    cp_rank       = seqlen_info.cp_rank;
    cp_kq_diff    = seqlen_info.tot_seqlen_k - static_cast<int>(seqlen_info.seqlen_q);
    tot_seqlen_k  = seqlen_info.tot_seqlen_k;

    mma_thread_idx = mma_thread_idx_;

    thr_mma_qk = thr_mma_qk_;
    tScS_mn    = tScS_mn_;
  }

  template <bool IsSeqlenKMask, class SEngine, class SLayout>
  MUTLASS_DEVICE void apply(Tensor<SEngine, SLayout>& tSrS, int const m_blk_idx, int const n_blk_idx) {
    auto   thr_mma_qk0 = TiledMmaQK{}.get_thread_slice(_0{});
    Tensor tS0cS_mn    = detail::make_identity_tensor_acc_qk(TileMN{}, thr_mma_qk0);

    Tensor tSrS_mn = make_tensor(tSrS.data(), layout_acc_mn(thr_mma_qk, tSrS.layout()));

    int seqlen_k_limit;
    if constexpr (IsSeqlenKMask) {
      seqlen_k_limit = seqlen_k_limit_base - n_blk_idx * TileN;
    }

    if constexpr (IsSeqlenKMask && !IsCausal && !IsLocal) {
      // SeqlenKMask  ONLY

      MUTE_UNROLL
      for (int n = 0; n < size<1>(tSrS_mn); ++n) {
        int const col_idx = static_cast<int>(get<1>(tS0cS_mn(_0{}, n)));
        bool      masked;
        if constexpr (EnableCP) {
          int const abs_k_local  = col_idx + thread_col_offset + n_blk_idx * TileN;
          int const abs_k_global = abs_k_local * cp_world_size + cp_rank;
          // mask if beyond actual local seqlen_k boundary OR beyond global tot_seqlen_k
          masked = (col_idx >= seqlen_k_limit) || (abs_k_global >= tot_seqlen_k);
        } else {
          masked = (col_idx >= seqlen_k_limit);
        }
        if (masked) {
          MUTE_UNROLL
          for (int m = 0; m < size<0>(tSrS_mn); ++m) {
            tSrS_mn(m, n) = -std::numeric_limits<Element>::infinity();
          }
        }
      }
    }  // IsSeqlenKMask ONLY

    constexpr int mma_thr_per_row = size<0, 0>(typename TiledMmaQK::AtomLayoutC_TV{});

    int pack_m_idx;
    if constexpr (IsPackGQA && (IsCausal || IsLocal)) {
      pack_m_idx = (get<0>(tScS_mn(mma_thread_idx % mma_thr_per_row, _0{})) + TileM * m_blk_idx) / HeadRatio;
    }

    if constexpr (IsCausal) {
      // CausalMask ONLY
      // or
      // CausalMask + SeqlenK

      int const causal_limit = causal_limit_base - n_blk_idx * TileN;

      MUTE_UNROLL
      for (int m = 0; m < size<0>(tSrS_mn); ++m) {
        int row_idx;
        if constexpr (IsPackGQA) {
          row_idx = __shfl_sync(0xffffffff, pack_m_idx, m % mma_thr_per_row, mma_thr_per_row);
        } else {
          row_idx = get<0>(tScS_mn(m, _0{})) + TileM * m_blk_idx;
        }

        if constexpr (EnableCP) {
          // CP causal: map local K col back to global K index, then compare.
          // local K index j -> global K = j * cp_world_size + cp_rank
          // mask when: abs_k_global > row_idx + (tot_seqlen_k - seqlen_q)
          //         or abs_k_global >= tot_seqlen_k  (padding token from uneven split)
          int const k_limit = row_idx + cp_kq_diff;
          MUTE_UNROLL
          for (int n = 0; n < size<1>(tSrS_mn); ++n) {
            int const col_idx      = static_cast<int>(get<1>(tS0cS_mn(_0{}, n)));
            int const abs_k_local  = col_idx + thread_col_offset + n_blk_idx * TileN;
            int const abs_k_global = abs_k_local * cp_world_size + cp_rank;
            if (abs_k_global > k_limit || abs_k_global >= tot_seqlen_k) {
              tSrS_mn(m, n) = -std::numeric_limits<Element>::infinity();
            }
          }
        } else {
          int causal_limit_col = causal_limit + row_idx;

          if constexpr (IsSeqlenKMask) {
            causal_limit_col = std::min(causal_limit_col, seqlen_k_limit);
          }

          MUTE_UNROLL
          for (int n = 0; n < size<1>(tSrS_mn); ++n) {
            if (static_cast<int>(get<1>(tS0cS_mn(_0{}, n))) >= causal_limit_col) {
              tSrS_mn(m, n) = -std::numeric_limits<Element>::infinity();
            }
          }
        }
      }
    }  // IsCausal

    if constexpr (IsLocal) {
      // LocalMask ONLY
      // or
      // LocalMask + SeqlenK
      // CP + Local not yet supported

      int const local_limit_left  = local_limit_left_base - n_blk_idx * TileN;
      int const local_limit_right = local_limit_right_base - n_blk_idx * TileN;

      int sink_limit;
      sink_limit = sink_limit_base - n_blk_idx * TileN;

      MUTE_UNROLL
      for (int m = 0; m < size<0>(tSrS_mn); ++m) {
        int row_idx;
        if constexpr (IsPackGQA) {
          row_idx = __shfl_sync(0xffffffff, pack_m_idx, m % mma_thr_per_row, mma_thr_per_row);
        } else {
          row_idx = get<0>(tScS_mn(m, _0{})) + TileM * m_blk_idx;
        }

        int local_limit_left_col  = local_limit_left + row_idx;
        int local_limit_right_col = local_limit_right + row_idx;

        if constexpr (IsSeqlenKMask) {
          local_limit_right_col = std::min(local_limit_right_col, seqlen_k_limit);
        }
        if constexpr (HasAttentionChunk) {
          int const chunk                = attention_chunk_divmod.divisor;
          int const anchor               = row_idx + seqlen_k - seqlen_q;
          int       col_limit_left_chunk = div_floor(attention_chunk_divmod, anchor) * chunk;
          col_limit_left_chunk -= n_blk_idx * TileN + thread_col_offset;
          local_limit_left_col  = std::max(local_limit_left_col, col_limit_left_chunk);
          local_limit_right_col = std::min(local_limit_right_col, col_limit_left_chunk + chunk);
        }

        MUTE_UNROLL
        for (int n = 0; n < size<1>(tSrS_mn); ++n) {
          int const col_idx = static_cast<int>(get<1>(tS0cS_mn(_0{}, n)));

          // TODO: SINK
          // if (col_idx >= local_limit_right_col || (col_idx < local_limit_left_col && col_idx >= sink_limit)) {
          if (col_idx >= local_limit_right_col || (col_idx < local_limit_left_col)) {
            tSrS_mn(m, n) = -std::numeric_limits<Element>::infinity();
          }
        }
      }

    }  // IsLocal
  }

};  // struct Mask

}  // namespace mate::attention::fmha
