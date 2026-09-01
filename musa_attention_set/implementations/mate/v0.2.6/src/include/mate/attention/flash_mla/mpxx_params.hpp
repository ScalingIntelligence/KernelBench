// Adapted from https://github.com/deepseek-ai/FlashMLA
#pragma once

#include <cstdint>

namespace mate::flash_mla {

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx (inclusive), begin_block_idx (inclusive), end_idx (inclusive), end_block_idx (exclusive),
// begin_n_split_idx, _, _, _]

struct GetDecodingMetadataParams {
  int* __restrict__ seqlens_k_ptr;
  int* __restrict__ tile_scheduler_metadata_ptr;
  int* __restrict__ num_splits_ptr;
  int* __restrict__ topk_length_ptr;
  int* __restrict__ extra_topk_length_ptr;
  int batch_size;
  int block_size_n;
  int fixed_overhead_num_blocks;
  int num_mp_parts;
  int topk;
  int extra_topk;
};

struct MlaCombineParams {
  using index_t = int64_t;

  bool is_varlen_q;

  void* __restrict__ o_ptr;
  void* __restrict__ softmax_lse_ptr;

  void* __restrict__ softmax_lseaccum_ptr;
  void* __restrict__ oaccum_ptr;
  int* __restrict__ num_splits_ptr;
  int* __restrict__ seqlens_q_ptr;

  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  int max_q_seq_per_hk;
  int total_q;
  int h_r;
  int h_k;
  int batch_size;
  int num_mp_parts;
};

}  // namespace mate::flash_mla
