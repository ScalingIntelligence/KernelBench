#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::attention::msa {

using namespace mute;

// Lightweight KV helper for MSA work items.
//
// Paged sparse MSA uses one physical/logical page per selected sparse block.
// BlockSize is therefore both the sparse block size and the page size. This
// helper owns sequence lengths and page-table lookup:
//   1. recover logical seqlen_k for one batch,
//   2. tell us how many columns are valid in one page/block,
//   3. map logical page/block index -> physical page index.
template <bool IsPagedKV_, int BlockSize_>
struct KVManager {
  static constexpr bool IsPagedKV = IsPagedKV_;
  static constexpr int  BlockSize = BlockSize_;

  int32_t const* ptr_page_indices        = nullptr;
  int32_t const* ptr_kv_page_indptr      = nullptr;
  int64_t        page_table_batch_stride = 0;
  int32_t const* ptr_cu_seqlens_k        = nullptr;
  int32_t const* ptr_seqused_k           = nullptr;

  static MUTLASS_DEVICE KVManager create(int32_t const* ptr_page_indices,
                                         int32_t const* ptr_kv_page_indptr,
                                         int64_t        page_table_batch_stride,
                                         int32_t const* ptr_cu_seqlens_k,
                                         int32_t const* ptr_seqused_k) {
    return {
        ptr_page_indices,
        ptr_kv_page_indptr,
        page_table_batch_stride,
        ptr_cu_seqlens_k,
        ptr_seqused_k,
    };
  }

  MUTLASS_DEVICE int logical_length(int batch_idx) const {
    if (ptr_seqused_k != nullptr) {
      return ptr_seqused_k[batch_idx];
    }
    if (ptr_cu_seqlens_k != nullptr) {
      return ptr_cu_seqlens_k[batch_idx + 1] - ptr_cu_seqlens_k[batch_idx];
    }
    return 0;
  }

  MUTLASS_DEVICE static int block_begin(int kv_block_idx) {
    return kv_block_idx * BlockSize;
  }

  MUTLASS_DEVICE int valid_cols_in_block(int batch_idx, int kv_block_idx) const {
    int remaining = logical_length(batch_idx) - block_begin(kv_block_idx);
    remaining     = remaining > 0 ? remaining : 0;
    return remaining < BlockSize ? remaining : BlockSize;
  }

  MUTLASS_DEVICE int physical_page_index(int batch_idx, int logical_page_idx) const {
    if constexpr (IsPagedKV) {
      int64_t page_begin =
          ptr_kv_page_indptr != nullptr ? ptr_kv_page_indptr[batch_idx] : int64_t(batch_idx) * page_table_batch_stride;
      return ptr_page_indices[page_begin + logical_page_idx];
    } else {
      static_cast<void>(batch_idx);
      return logical_page_idx;
    }
  }

  MUTLASS_DEVICE int physical_block_index(int batch_idx, int kv_block_idx) const {
    return physical_page_index(batch_idx, kv_block_idx);
  }

  MUTLASS_DEVICE int physical_token_begin(int batch_idx, int kv_block_idx) const {
    if constexpr (IsPagedKV) {
      return physical_block_index(batch_idx, kv_block_idx) * BlockSize;
    } else {
      int batch_offset = ptr_cu_seqlens_k != nullptr ? ptr_cu_seqlens_k[batch_idx] : 0;
      return batch_offset + block_begin(kv_block_idx);
    }
  }
};

template <int BlockSize>
using PagedKVManager = KVManager<true, BlockSize>;

}  // namespace mate::attention::msa
