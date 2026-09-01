#include <mutlass/fast_math.h>

#include <cub/cub.cuh>

#include "fmha.hpp"

template <typename T, typename Compare>
__device__ __forceinline__ int merge_path_find_i(const T* A, int m, const T* B, int n, int d, Compare comp) {
  int lo = max(0, d - n);
  int hi = min(d, m);
  while (lo < hi) {
    int i = (lo + hi) >> 1;
    int j = d - i;

    // Want: A[i-1] not before B[j]  <=> B[j] before A[i-1]
    bool a_im1_after_bj = false;
    if (i > 0 && j < n) {
      // if comp(B[j], A[i-1]) then A[i-1] should be after B[j]
      a_im1_after_bj = comp(B[j], A[i - 1]);
    }

    // Want: B[j-1] before A[i]  (strict) to avoid crossing.
    // If NOT (B[j-1] before A[i]) then we need larger i.
    bool b_jm1_not_before_ai = false;
    if (j > 0 && i < m) {
      b_jm1_not_before_ai = !comp(B[j - 1], A[i]);
    }

    if (a_im1_after_bj) {
      hi = i;
    } else if (b_jm1_not_before_ai) {
      lo = i + 1;
    } else {
      return i;
    }
  }
  return lo;
}

// Block cooperative merge of two runs in shared memory.
// Each thread writes a contiguous output segment.
template <int BLOCK_THREADS, typename T, typename Compare>
__device__ __forceinline__ void block_merge_two_runs(const T* in,
                                                     T*       out,
                                                     int      run0_begin,
                                                     int      run0_len,
                                                     int      run1_begin,
                                                     int      run1_len,
                                                     int      out_begin,
                                                     int      out_len,
                                                     Compare  comp) {
  int tid = (int)threadIdx.x;

  // Partition output across threads.
  int chunk = (out_len + BLOCK_THREADS - 1) / BLOCK_THREADS;
  int o0    = tid * chunk;
  int o1    = min(out_len, o0 + chunk);
  if (o0 >= o1) return;

  const T* A = in + run0_begin;
  const T* B = in + run1_begin;
  int      m = run0_len, n = run1_len;

  int d0 = o0;
  int d1 = o1;
  int i0 = merge_path_find_i(A, m, B, n, d0, comp);
  int j0 = d0 - i0;
  int i1 = merge_path_find_i(A, m, B, n, d1, comp);
  int j1 = d1 - i1;

  int ia = i0, ib = j0;
  int out_idx = out_begin + o0;

  // Stable merge: if neither is strictly before the other, take from left (A).
  while (ia < i1 || ib < j1) {
    bool takeA;
    if (ib >= j1)
      takeA = true;
    else if (ia >= i1)
      takeA = false;
    else {
      // take A unless B should come before A
      takeA = !comp(B[ib], A[ia]);
    }
    out[out_idx++] = takeA ? A[ia++] : B[ib++];
  }
}

template <typename T, int BLOCK_THREADS>
struct BlockMergeSortLike_1Item {
  static constexpr int ITEMS_PER_THREAD = 1;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS;

  struct TempStorage {
    T buf0[TILE_ITEMS];
    T buf1[TILE_ITEMS];
  };

  TempStorage& temp;

  __device__ __forceinline__ explicit BlockMergeSortLike_1Item(TempStorage& s) : temp(s) {
  }

  template <typename Compare>
  __device__ __forceinline__ void Sort(T (&items)[1], Compare comp) {
    int tid = (int)threadIdx.x;

    // Write to shared
    temp.buf0[tid] = items[0];
    __syncthreads();

    T* in  = temp.buf0;
    T* out = temp.buf1;

    // Merge passes: run length doubles
    MUTLASS_PRAGMA_UNROLL
    for (int run = 1; run < TILE_ITEMS; run <<= 1) {
      int pair_len = run << 1;

      // Merge all pairs in this pass
      MUTLASS_PRAGMA_UNROLL
      for (int base = 0; base < TILE_ITEMS; base += pair_len) {
        int len0 = min(run, TILE_ITEMS - base);
        int len1 = min(run, TILE_ITEMS - (base + run));
        if (len1 <= 0) {
          // copy leftover run
          int chunk = (len0 + BLOCK_THREADS - 1) / BLOCK_THREADS;
          int o0    = tid * chunk;
          int o1    = min(len0, o0 + chunk);
          for (int k = o0; k < o1; ++k) out[base + k] = in[base + k];
        } else {
          block_merge_two_runs<BLOCK_THREADS>(in, out, base, len0, base + run, len1, base, len0 + len1, comp);
        }
      }

      __syncthreads();
      T* tmp = in;
      in     = out;
      out    = tmp;
      __syncthreads();
    }

    items[0] = in[tid];
  }
};

__device__ __forceinline__ int split_removal_cost(int num_n_blocks, int num_splits) {
  return num_splits > 1 ? mutlass::ceil_div(num_n_blocks, num_splits - 1) - mutlass::ceil_div(num_n_blocks, num_splits)
                        : INT_MAX;
}

// Number of splits is not the more the better
// fwd kernel gains more parallelism, but combine kernel will be much slower.
// If a split has very few N blocks, the split is usually not worth it;
// here we gradually reduce the number of splits until the shortest split is large enough.
__device__ __forceinline__ int clamp_num_splits_for_min_n_blocks(int num_n_blocks,
                                                                 int num_splits,
                                                                 int min_n_blocks_per_split) {
  if (num_n_blocks <= 0 || num_splits <= 1) return 1;
  num_splits = max(min(num_splits, max(num_n_blocks / min_n_blocks_per_split, 1)), 1);
  while (num_splits > 1) {
    int const n_blocks_per_split = mutlass::ceil_div(num_n_blocks, num_splits);
    int const tail_n_blocks      = num_n_blocks - (num_splits - 1) * n_blocks_per_split;
    if (tail_n_blocks >= min_n_blocks_per_split) break;
    --num_splits;
  }
  return num_splits;
}

namespace mate::attention::fmha {
// Sort in descending order
template <typename T>
struct PrepareSortOp {
  __device__ __forceinline__ bool operator()(T const& lhs, T const& rhs) {
    return lhs > rhs;
  }
};

template <>
struct PrepareSortOp<int2> {
  __device__ __forceinline__ bool operator()(int2 const& lhs, int2 const& rhs) const {
    return lhs.x > rhs.x;
  }
};

template <>
struct PrepareSortOp<int4> {
  __device__ __forceinline__ bool operator()(int4 const& lhs, int4 const& rhs) const {
    return lhs.x > rhs.x;
  }
};

template <int  NumWarps,
          bool Sort,
          bool PackGQA,
          bool Causal,
          bool IsLocal,
          bool L2Swizzle        = false,
          bool HasCuSeqlensQ    = false,
          bool HasSequsedQ      = false,
          bool HasCuSeqlensK    = false,
          bool HasSequsedK      = false,
          bool HasCuSeqlensKNew = false,
          bool HasLeftpadK      = false>
__global__ void get_metadata_kernel(int                   seqlen_q_static,
                                    int                   seqlen_k_static,
                                    int                   seqlen_k_new_static,
                                    uint32_t const* const cu_seqlens_q,
                                    uint32_t const* const cu_seqlens_k,
                                    uint32_t const* const cu_seqlens_k_new,
                                    uint32_t const* const seqused_q,
                                    uint32_t const* const seqused_k,
                                    uint32_t const* const leftpad_k_ptr,
                                    int                   num_batch,
                                    int                   num_head,
                                    int                   qhead_per_khead,
                                    int                   num_mp,
                                    int                   num_splits_static,
                                    mutlass::FastDivmod   blockm_divmod,
                                    mutlass::FastDivmod   blockn_divmod,
                                    int                   window_size_left,
                                    int                   window_size_right,
                                    // int* const tile_count_semaphore,
                                    int* const num_m_blocks_ptr,
                                    int* const num_splits_dynamic_ptr,
                                    int* const varlen_batch_idx_ptr,
                                    // int* const num_n_blocks_ptr,
                                    int* const num_nheads_in_l2_ptr,
                                    int        max_kvblocks_in_l2) {
  static constexpr int kNumBatchPerWarp = mutlass::NumThreadsPerWarp - 1;  // Each warp processes 31 batches
  static constexpr int BATCH_PER_THREAD = 1;
  static constexpr int BLOCK_DIM_X      = NumWarps * mutlass::NumThreadsPerWarp;
  static_assert(BLOCK_DIM_X * BATCH_PER_THREAD == NumWarps * mutlass::NumThreadsPerWarp);
  using LPT = BlockMergeSortLike_1Item<int4, BLOCK_DIM_X>;
  // using LPT = cub::BlockMergeSort<int4, BLOCK_DIM_X, BATCH_PER_THREAD>;
  using SplitReduce = cub::BlockReduce<int, BLOCK_DIM_X>;

  __shared__ int total_blocks_smem[1];

  // Shared buffers for budget-aware split refinement.
  __shared__ int total_splits_smem;
  __shared__ int select_flags[BLOCK_DIM_X];

  __shared__ typename SplitReduce::TempStorage splits_reduce_smem;

  struct SplitReductionCandidate {
    int          removal_cost;
    unsigned int thread_idx;
  };
  struct SplitReductionCandidateAsc {
    __device__ __forceinline__ bool operator()(SplitReductionCandidate const& lhs,
                                               SplitReductionCandidate const& rhs) const {
      return lhs.removal_cost < rhs.removal_cost ||
             (lhs.removal_cost == rhs.removal_cost && lhs.thread_idx < rhs.thread_idx);
    }
  };
  using SplitReductionSort = BlockMergeSortLike_1Item<SplitReductionCandidate, BLOCK_DIM_X>;
  // using SplitReductionSort = cub::BlockMergeSort<SplitReductionCandidate, BLOCK_DIM_X, BATCH_PER_THREAD>;
  __shared__ typename SplitReductionSort::TempStorage split_reduction_sort_storage;

  __shared__ typename LPT::TempStorage lpt_storage;

  if (threadIdx.x == 0) {
    total_blocks_smem[0] = 0;
  }
  __syncthreads();

  int lane                 = threadIdx.x % mutlass::NumThreadsPerWarp;
  int warp_idx             = threadIdx.x / mutlass::NumThreadsPerWarp;
  int batch_cta_idx_offset = static_cast<int>(blockIdx.x) * kNumBatchPerWarp * mutlass::NumThreadsPerWarp;
  int batch_offset         = batch_cta_idx_offset + warp_idx * kNumBatchPerWarp;
  int batch_idx            = batch_offset + lane;

  auto get_num_m_blocks = [&](int batch_idx) {
    int seqlen;
    if constexpr (HasSequsedQ) {
      seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
    } else if constexpr (HasCuSeqlensQ) {
      int cur_cu_seqlen  = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
      int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
      seqlen             = next_cu_seqlen - cur_cu_seqlen;
    } else {
      seqlen = seqlen_q_static;
    }
    if constexpr (PackGQA) {
      seqlen *= qhead_per_khead;
    }
    return batch_idx < num_batch && lane < kNumBatchPerWarp ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
  };

  auto get_num_n_blocks = [&](int batch_idx) {
    int leftpad_k;
    if constexpr (HasLeftpadK) {
      leftpad_k = batch_idx < num_batch ? leftpad_k_ptr[batch_idx] : 0;
    } else {
      leftpad_k = 0;
    }
    int seqlen;
    if constexpr (HasSequsedK) {
      seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
    } else if constexpr (HasCuSeqlensK) {
      int cur_cu_seqlen  = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
      int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
      seqlen             = next_cu_seqlen - cur_cu_seqlen;
    } else {
      seqlen = seqlen_k_static;
    }
    int seqlen_new;
    if constexpr (HasCuSeqlensKNew) {
      int cur_cu_seqlen_new  = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
      int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
      seqlen_new             = next_cu_seqlen_new - cur_cu_seqlen_new;
    } else {
      seqlen_new = seqlen_k_new_static;
    }
    seqlen = seqlen - leftpad_k + seqlen_new;
    if constexpr (IsLocal) {
      // Native FA3 metadata does not account for SWA/local invalid tiles. Cap
      // the estimate here until a VarlenDynamic-style scheduler can skip them,
      // then compare performance.
      int const seqlen_loaded =
          std::max(0, std::min(seqlen, window_size_left + window_size_right + 1 + blockm_divmod.divisor));
      seqlen = seqlen_loaded;
    }
    return batch_idx < num_batch && lane < kNumBatchPerWarp ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
  };

  auto get_nheads_in_l2 = [&](int n_blocks) {
    int nheads_in_l2 = n_blocks * 16 <= max_kvblocks_in_l2  ? 16
                       : n_blocks * 8 <= max_kvblocks_in_l2 ? 8
                       : n_blocks * 4 <= max_kvblocks_in_l2 ? 4
                       : n_blocks * 2 <= max_kvblocks_in_l2 ? 2
                                                            : 1;
    if constexpr (!PackGQA) {
      nheads_in_l2 *= qhead_per_khead;
    }
    return min(nheads_in_l2, num_head);
  };

  int num_m_blocks = get_num_m_blocks(batch_idx);
  int num_n_blocks = get_num_n_blocks(batch_idx);

  int num_splits_dynamic;
  if (num_splits_static == 1 || static_cast<int>(gridDim.x) != 1) {
    num_splits_dynamic = 1;
  } else {
    int total_blocks = num_m_blocks * num_n_blocks;
    // BlockSum
    MUTLASS_PRAGMA_UNROLL
    for (int i = mutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
      total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
    }
    if (lane == 0) atomicAdd(total_blocks_smem, total_blocks);
    __syncthreads();
    total_blocks = total_blocks_smem[0];

    int blocks_per_mp =
        static_cast<int>(ceilf(static_cast<float>(total_blocks) * 1.1f * static_cast<float>(num_head) / num_mp));
    // if (threadIdx.x == 0) printf("total_block=%d, blocks_per_mp=%d\n", total_blocks, blocks_per_mp);
    num_splits_dynamic = std::max(std::min(mutlass::ceil_div(num_n_blocks, blocks_per_mp), num_splits_static), 1);

    if (num_m_blocks == 1 && num_batch * num_head <= 3) {
      // Large number of splits is possible when total workload is small
      // We decide each split should have at least 4 N blocks, otherwise the combine kernel will be too slow.
      num_splits_dynamic = clamp_num_splits_for_min_n_blocks(num_n_blocks, num_splits_dynamic, 4);
    }

    int valid_thread = (lane < kNumBatchPerWarp && batch_idx < num_batch) ? 1 : 0;

    // Start from the upstream-style base heuristic, then enforce the global
    // SM budget by repeatedly removing one split from the cheapest requests.
    int total_splits = SplitReduce(splits_reduce_smem).Sum(valid_thread ? num_splits_dynamic : 0);
    __syncthreads();

    if (threadIdx.x == 0) {
      total_splits_smem = total_splits;
    }
    __syncthreads();

    int overflow = max(total_splits_smem - num_mp, 0);
    while (overflow > 0) {
      SplitReductionCandidate items[BATCH_PER_THREAD];
      items[0] = {valid_thread ? split_removal_cost(num_n_blocks, num_splits_dynamic) : INT_MAX,
                  static_cast<unsigned int>(threadIdx.x)};

      SplitReductionSort(split_reduction_sort_storage).Sort(items, SplitReductionCandidateAsc());
      __syncthreads();

      select_flags[threadIdx.x] = 0;
      if (threadIdx.x < overflow && items[0].removal_cost != INT_MAX) {
        select_flags[items[0].thread_idx] = 1;
      }
      __syncthreads();

      int reduced_this_round = valid_thread && select_flags[threadIdx.x] && num_splits_dynamic > 1 ? 1 : 0;
      if (reduced_this_round) --num_splits_dynamic;
      int total_reduced = SplitReduce(splits_reduce_smem).Sum(reduced_this_round);
      __syncthreads();

      if (threadIdx.x == 0) {
        total_splits_smem = total_reduced;
      }
      __syncthreads();

      if (total_splits_smem == 0) break;
      overflow -= total_splits_smem;
    }
    if (num_m_blocks == 1 && num_batch * num_head <= 3) {
      // ceil_div may produce low-workload splits, so check again.
      num_splits_dynamic = clamp_num_splits_for_min_n_blocks(num_n_blocks, num_splits_dynamic, 4);
    }
    num_n_blocks = mutlass::ceil_div(num_n_blocks, num_splits_dynamic);
  }

  if constexpr (Sort) {
    // thread 31 is invalid
    if (lane == kNumBatchPerWarp || batch_idx >= num_batch) {
      num_n_blocks = INT_MIN;
    } else if constexpr (Causal) {
      num_n_blocks =
          num_n_blocks * blockn_divmod.divisor - num_m_blocks * blockm_divmod.divisor;  // n = n * bn - m * bm
    }
    int4 batch_coords[BATCH_PER_THREAD];
    batch_coords[0] = make_int4(num_n_blocks, num_m_blocks, num_splits_dynamic, batch_idx);
    // Sort
    LPT(lpt_storage).Sort(batch_coords, PrepareSortOp<int4>());

    if constexpr (Causal) {
      batch_coords[0].x =
          blockn_divmod.div(batch_coords[0].x + batch_coords[0].y * blockm_divmod.divisor);  // recover num_n_blocks
    }
    // After sorting, only first 31 * 32 threads contains valid data
    batch_idx = batch_cta_idx_offset + threadIdx.x;
    if (batch_idx < num_batch && threadIdx.x < kNumBatchPerWarp * 32) {
      if constexpr (L2Swizzle) {
        num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(batch_coords[0].x, 1));
      }
      num_m_blocks_ptr[batch_idx]       = batch_coords[0].y;
      num_splits_dynamic_ptr[batch_idx] = batch_coords[0].z;
      varlen_batch_idx_ptr[batch_idx]   = batch_coords[0].w;
    }
  } else {
    if (batch_idx < num_batch && lane < kNumBatchPerWarp) {
      if constexpr (L2Swizzle) {
        num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(num_n_blocks, 1));
      }
      num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic;
      num_m_blocks_ptr[batch_idx]       = num_m_blocks;
    }
  }
}

inline bool is_valid_num_splits(int num_n_blocks, int num_splits, int min_n_blocks_per_split) {
  if (min_n_blocks_per_split <= 0 || num_splits <= 1) return true;
  // ceil_div may produce low-workload splits.
  int const n_blocks_per_split = mutlass::ceil_div(num_n_blocks, num_splits);
  int const tail_n_blocks      = num_n_blocks - (num_splits - 1) * n_blocks_per_split;
  return tail_n_blocks >= min_n_blocks_per_split;
}

inline int get_min_n_blocks_per_split(int total_mblocks) {
  // M blocks provide parallelism themselves
  // The more M blocks, the coarser the N direction should be cut to avoid excessive combine workload.
  // min value is ceil(4 * sqrt(M)); for M = 1, 2, 3, each split should have at least 4, 6, 7 N blocks respectively.
  int const target = 16 * std::max(total_mblocks, 1);
  int       result = 1;
  while (result * result < target) ++result;
  return result;
}

inline int get_combine_split_cap(int num_n_blocks, int tile_n, int max_splits) {
  // combine templates are available for 16, 32, 64 splits
  // once we cross a threshold, the fixed workload will increase immediately.
  int cap = tile_n % 128 == 0 ? 32 : 16;
  cap     = std::min(cap, max_splits);
  while (cap < max_splits && static_cast<long long>(num_n_blocks) >= 4LL * cap * cap) {
    cap = std::min(cap * 2, max_splits);
  }
  return cap;
}

inline int num_splits_heuristic(int  total_mblocks,
                                int  num_mp,
                                int  num_n_blocks,
                                int  num_m_blocks,
                                int  size_one_kv_head,
                                bool is_causal_or_local,
                                int  max_splits,
                                int  min_n_blocks_per_split = 0) {
  constexpr float kMinWaveOccupancy = 0.8f;
  constexpr float kEfficiencySlack  = 0.85f;

  // If we have enough to almost fill the SMs, then just use 1 split
  // However, in the case of super long seqlen where each head of KV doesn't even fit into
  // L2 (we assume that L2 size is 50MB), we want to split.
  if (total_mblocks >= kMinWaveOccupancy * num_mp) {
    int const size_l2 = 1.5 * 1024 * 1024;  // 1.5 MB
    // Only split if there are enough queries to go over the KV at least twice
    // Don't split if causal
    if (size_one_kv_head > size_l2 && num_m_blocks >= num_mp * 2 && !is_causal_or_local) {
      return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
    } else {
      return 1;
    }
  }
  // If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
  if (num_n_blocks <= 4) {
    return 1;
  }
  max_splits                    = std::min({max_splits, num_mp, num_n_blocks});
  int const one_wave_cap        = total_mblocks > 0 ? num_mp / total_mblocks : 0;
  int const one_wave_max_splits = one_wave_cap > 0 ? std::min(one_wave_cap, max_splits) : 0;
  if (one_wave_max_splits > 0) {
    float const one_wave_eff = static_cast<float>(total_mblocks * one_wave_max_splits) / num_mp;
    if (one_wave_eff >= kMinWaveOccupancy && max_splits > one_wave_max_splits) {
      max_splits = one_wave_max_splits;
    }
  }
  float              max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_valid_num_splits(num_n_blocks, num_splits, min_n_blocks_per_split)) {
      efficiency.push_back(0.f);
      continue;
    }
    float n_waves = float(total_mblocks * num_splits) / num_mp;
    float eff     = n_waves / ceil(n_waves);
    // printf("num_splits = %d, eff = %f\n", num_splits, eff);
    if (eff > max_efficiency) {
      max_efficiency = eff;
    }
    efficiency.push_back(eff);
  }
  int chosen = 1;
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (is_valid_num_splits(num_n_blocks, num_splits, min_n_blocks_per_split) &&
        efficiency[num_splits - 1] >= kEfficiencySlack * max_efficiency) {
      chosen = num_splits;
      break;
    }
  }
  return chosen;
}

inline int get_num_splits(FmhaFwdParams const& params, int tile_m, int tile_n, bool verbose = false) {
  // TODO: leftpad_k
  bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k;

  int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
  if (verbose) {
    printf("varlen: %d, seqlen_q_packgqa: %d\n", varlen, seqlen_q_packgqa);
  }
  // metadata kernel need to set params for local.
  // int const seqlen_k_loaded = params.seqlen_k;
  int const seqlen_k_loaded =
      !params.is_local
          ? params.seqlen_k
          : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + tile_m));
  int const num_n_blocks = (seqlen_k_loaded + tile_n - 1) / tile_n;
  int const num_m_blocks = (seqlen_q_packgqa + tile_m - 1) / tile_m;
  if (verbose) {
    printf("tile_n: %d, tile_m: %d\n", tile_n, tile_m);
    printf("seqlen_k_loaded: %d, num_n_blocks: %d, num_m_blocks: %d\n", seqlen_k_loaded, num_n_blocks, num_m_blocks);
  }
  int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_fp8 ? 1 : 2);
  // TODO: for non-scheduled, this should be batch
  int total_mblocks = params.b * params.h_k * num_m_blocks;
  if (verbose) {
    printf("size_one_kv_head: %d, total_mblocks: %d\n", size_one_kv_head, total_mblocks);
  }

  int max_splits             = 128;
  int min_n_blocks_per_split = 0;
  if (num_m_blocks == 1 && total_mblocks <= 3) {
    // Small workload case: when each request has only 1 M block, and the total is no more than 3
    // the original rule tends to increase splits to fill the GPU.
    min_n_blocks_per_split    = get_min_n_blocks_per_split(total_mblocks);
    int const combine_cap     = get_combine_split_cap(num_n_blocks, tile_n, max_splits);
    int const granularity_cap = std::max(num_n_blocks / min_n_blocks_per_split, 1);
    max_splits                = std::min(combine_cap, granularity_cap);
  }

  return num_splits_heuristic(total_mblocks,
                              params.num_mp,
                              num_n_blocks,
                              num_m_blocks,
                              size_one_kv_head,
                              params.is_causal || params.is_local,
                              max_splits,
                              min_n_blocks_per_split);
}

}  // namespace mate::attention::fmha
