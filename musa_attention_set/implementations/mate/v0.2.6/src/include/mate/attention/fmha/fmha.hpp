#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace mate::attention::fmha {

struct FmhaFwdParams {
  using index_t = int64_t;
  // The QKV matrics
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrics
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;

  // The number of heads
  int h, h_k;

  // The O matrix (output)
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

  // The stride between rows of the O matrix
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the softmax sum
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  index_t lse_batch_stride;
  index_t lse_head_stride;

  // For FP8 scaling
  float* __restrict__ q_descale_ptr;
  float* __restrict__ k_descale_ptr;
  float* __restrict__ v_descale_ptr;
  index_t q_descale_batch_stride;
  index_t q_descale_head_stride;
  index_t k_descale_batch_stride;
  index_t k_descale_head_stride;
  index_t v_descale_batch_stride;
  index_t v_descale_head_stride;

  // The dimensions.
  int b;
  int seqlen_q;
  int seqlen_k;
  int seqlen_knew;
  int total_q;
  int total_k;
  int total_knew;
  int d;
  int d_rounded;
  int dv;
  int dv_rounded;
  int rotary_dim;

  int b_k;  // When having KV cache and with cache_batch_idx, K & V might have larger batch size
            // than Q

  // The scaling factors for the kernel.
  float scale_softmax;
  float softcap;

  // array of length b+1 holding starting offset of each sequence.
  uint32_t* __restrict__ cu_seqlens_q;
  uint32_t* __restrict__ cu_seqlens_k;
  uint32_t* __restrict__ cu_seqlens_knew;
  uint32_t* __restrict__ leftpad_k;

  // If provided, the actual length of each q/k sequence.
  uint32_t* __restrict__ seqused_q;
  uint32_t* __restrict__ seqused_k;

  // The stride between rows of Oaccum.
  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;

  // The stride between rows of LSEaccum.
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;

  void* __restrict__ qv_ptr;
  index_t qv_batch_stride;
  index_t qv_row_stride;
  index_t qv_head_stride;

  // The cos and sin matrices for rotary embedding.
  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;
  uint32_t* __restrict__ seqlens_rotary;

  // The indices to index into the KV cache.
  uint32_t* __restrict__ kv_batch_idx;

  // Learnable Sink
  void* __restrict__ learnable_sink_ptr;

  // Paged KV cache
  int* __restrict__ page_table;
  index_t page_table_batch_stride;
  int     page_size;
  int     num_pages;

  int num_mp;
  int num_splits;
  int* __restrict__ num_splits_dynamic_ptr;
  int* __restrict__ batch_table_ptr;

  bool is_causal;
  bool is_local;

  bool is_rotary_interleaved;

  int window_size_left, window_size_right;
  int attention_chunk;

  bool is_fp8;
};

inline std::ostream& operator<<(std::ostream& os, const FmhaFwdParams& p) {
  auto field = [&os](const char* name, auto&& value) {
    os << std::setw(28) << std::left << (std::string(name) + ": ") << value << "\n";
  };

  os << "=== FmhaFwdParams ===\n";

  // --- QKV Pointers ---
  field("q_ptr", p.q_ptr);
  field("k_ptr", p.k_ptr);
  field("v_ptr", p.v_ptr);

  // --- Q Strides ---
  os << "\n[Q Strides]\n";
  field("  q_batch_stride", p.q_batch_stride);
  field("  q_row_stride", p.q_row_stride);
  field("  q_head_stride", p.q_head_stride);

  // --- K Strides ---
  os << "\n[K Strides]\n";
  field("  k_batch_stride", p.k_batch_stride);
  field("  k_row_stride", p.k_row_stride);
  field("  k_head_stride", p.k_head_stride);

  // --- V Strides ---
  os << "\n[V Strides]\n";
  field("  v_batch_stride", p.v_batch_stride);
  field("  v_row_stride", p.v_row_stride);
  field("  v_head_stride", p.v_head_stride);
  field("  v_dim_stride", p.v_dim_stride);

  // --- Head counts ---
  os << "\n[Heads]\n";
  field("h", p.h);
  field("h_k", p.h_k);

  // --- Output pointers ---
  os << "\n[Output]\n";
  field("o_ptr", p.o_ptr);
  field("oaccum_ptr", p.oaccum_ptr);
  field("o_batch_stride", p.o_batch_stride);
  field("o_row_stride", p.o_row_stride);
  field("o_head_stride", p.o_head_stride);

  // --- Softmax LSE ---
  os << "\n[Softmax LSE]\n";
  field("softmax_lse_ptr", p.softmax_lse_ptr);
  field("softmax_lseaccum_ptr", p.softmax_lseaccum_ptr);
  field("lse_batch_stride", p.lse_batch_stride);
  field("lse_head_stride", p.lse_head_stride);

  //// --- FP8 Descale Pointers ---
  // os << "\n[FP8 Descale Pointers]\n";
  // field("q_descale_ptr", p.q_descale_ptr);
  // field("k_descale_ptr", p.k_descale_ptr);
  // field("v_descale_ptr", p.v_descale_ptr);

  // os << "\n[FP8 Descale Strides]\n";
  // field("q_descale_batch_stride", p.q_descale_batch_stride);
  // field("q_descale_head_stride",  p.q_descale_head_stride);
  // field("k_descale_batch_stride", p.k_descale_batch_stride);
  // field("k_descale_head_stride",  p.k_descale_head_stride);
  // field("v_descale_batch_stride", p.v_descale_batch_stride);
  // field("v_descale_head_stride",  p.v_descale_head_stride);

  // --- Dimensions ---
  os << "\n[Dimensions]\n";
  field("b", p.b);
  field("seqlen_q", p.seqlen_q);
  field("seqlen_k", p.seqlen_k);
  field("seqlen_knew", p.seqlen_knew);
  field("total_q", p.total_q);
  field("total_k", p.total_k);
  field("total_knew", p.total_knew);
  field("d", p.d);
  field("d_rounded", p.d_rounded);
  field("dv", p.dv);
  field("dv_rounded", p.dv_rounded);
  field("b_k", p.b_k);
  field("rotary_dim", p.rotary_dim);

  // --- Scaling ---
  os << "\n[Scaling]\n";
  field("scale_softmax", p.scale_softmax);
  field("softcap", p.softcap);

  // --- Cu seqlens ---
  os << "\n[Cu Seqlens]\n";
  field("cu_seqlens_q", p.cu_seqlens_q);
  field("cu_seqlens_k", p.cu_seqlens_k);
  field("cu_seqlens_knew", p.cu_seqlens_knew);
  field("leftpad_k", p.leftpad_k);

  // --- Seqused ---
  os << "\n[Seqused]\n";
  field("seqused_q", p.seqused_q);
  field("seqused_k", p.seqused_k);

  // --- Oaccum Strides ---
  os << "\n[Oaccum Strides]\n";
  field("oaccum_split_stride", p.oaccum_split_stride);
  field("oaccum_batch_stride", p.oaccum_batch_stride);
  field("oaccum_row_stride", p.oaccum_row_stride);
  field("oaccum_head_stride", p.oaccum_head_stride);

  // --- LSE Accum Strides ---
  os << "\n[LSE Accum Strides]\n";
  field("lseaccum_split_stride", p.lseaccum_split_stride);
  field("lseaccum_batch_stride", p.lseaccum_batch_stride);
  field("lseaccum_head_stride", p.lseaccum_head_stride);

  //// --- K_new / V_new ---
  // os << "\n[K/V New]\n";
  // field("knew_ptr", p.knew_ptr);
  // field("vnew_ptr", p.vnew_ptr);
  // field("knew_batch_stride", p.knew_batch_stride);
  // field("vnew_batch_stride", p.vnew_batch_stride);
  // field("knew_row_stride",   p.knew_row_stride);
  // field("vnew_row_stride",   p.vnew_row_stride);
  // field("knew_head_stride",  p.knew_head_stride);
  // field("vnew_head_stride",  p.vnew_head_stride);

  //// --- QV ---
  // os << "\n[QV]\n";
  // field("qv_ptr", p.qv_ptr);
  // field("qv_batch_stride", p.qv_batch_stride);
  // field("qv_row_stride",   p.qv_row_stride);
  // field("qv_head_stride",  p.qv_head_stride);

  //// --- Rotary Embedding ---
  // os << "\n[Rotary]\n";
  // field("rotary_cos_ptr", p.rotary_cos_ptr);
  // field("rotary_sin_ptr", p.rotary_sin_ptr);
  // field("seqlens_rotary", p.seqlens_rotary);

  // --- KV Indexing ---
  os << "\n[KV Indexing]\n";
  field("kv_batch_idx", p.kv_batch_idx);

  // --- Paged KV Cache ---
  os << "\n[Paged KV Cache]\n";
  field("page_table", p.page_table);
  field("page_table_batch_stride", p.page_table_batch_stride);
  field("page_size", p.page_size);
  field("num_pages", p.num_pages);

  return os;
}

}  // namespace mate::attention::fmha
