#pragma once

#include <mute/tensor.hpp>

namespace mate::attention::fmha {

using namespace mute;

#define SHOW(x)   \
  print(#x ": "); \
  print(x);       \
  print("\n")

template <bool IsPagedKV,
          class Element,
          int  NumThreads,
          int  TileN,
          int  HeadDimQK,
          int  HeadDimVO,
          bool IsKVSameIter   = false,
          int  LoadsPerRow_LB = 1,
          int  VectorBits     = 128>
struct PagedKVManager {
  using ShapePageTable  = Shape<int32_t, int32_t>;
  using StridePageTable = Stride<int64_t, _1>;

  using TensorPageTable = decltype(make_tensor(
      make_gmem_ptr(static_cast<int32_t const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int32_t(0), _));

  using ShapeKV  = Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideKV = Stride<int64_t, _1, int64_t, int64_t>;

  using TensorKV =
      decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, 0, _));

  static constexpr bool SameHeadDim = (HeadDimQK == HeadDimVO);
  static constexpr int  HeadDimGCD  = mute::gcd(HeadDimQK, HeadDimVO);
  // For Lsu Paged Load
  static constexpr int ElementsPerLoad = VectorBits / sizeof_bits_v<Element>;
  static_assert(HeadDimGCD % ElementsPerLoad == 0, "HeadDimQK and HeadDimVO must be a multiple of ElementsPerLoad");
  static_assert(HeadDimGCD % LoadsPerRow_LB == 0, "HeadDimQK and HeadDimVO must be a multiple of LoadsPerRow_LB");
  static constexpr int BytePerRow = HeadDimGCD / LoadsPerRow_LB * sizeof(Element);
  static constexpr int BlockKGmem = (BytePerRow % 128 == 0 ? 128 : (BytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int GmemThreadsPerRow = BlockKGmem / ElementsPerLoad;
  using FragmentType                     = mute::uint_bit_t<VectorBits>;
  using GmemCopyAtom                     = Copy_Atom<MP31_ROBUST_LDGSTS<FragmentType>, Element>;
  using GmemLayoutAtom =
      Layout<Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>, Stride<Int<GmemThreadsPerRow>, _1>>;

  using GmemTiledCopy =
      decltype(make_tiled_copy(GmemCopyAtom{}, GmemLayoutAtom{}, Layout<Shape<_1, Int<ElementsPerLoad>>>{}));
  using ThrGmemTiledCopy = decltype(GmemTiledCopy{}.get_thread_slice(0));

  using GmemTiledCopyKVStore =
      decltype(make_tiled_copy(Copy_Atom<MP31_ROBUST_STORE<FragmentType>, Element>{},
                               GmemLayoutAtom{},
                               Layout<Shape<_1, Int<ElementsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load
  using ThrGmemTiledCopyStore = decltype(GmemTiledCopyKVStore{}.get_thread_slice(0));

  using TensortKcK = decltype(GmemTiledCopy{}.get_thread_slice(0).partition_S(
      make_identity_tensor(Shape<Int<TileN>, Int<HeadDimQK>>{})));
  using TensortVcV = decltype(GmemTiledCopy{}.get_thread_slice(0).partition_S(
      make_identity_tensor(Shape<Int<TileN>, Int<HeadDimVO>>{})));

  static_assert(MUTE_STATIC_V(size<1>(TensortKcK{})) == MUTE_STATIC_V(size<1>(TensortVcV{})));

  static constexpr int PageEntryPerThread = ceil_div(size<1>(TensortKcK{}), GmemThreadsPerRow);
  using TensorKVPtr                       = decltype(make_tensor<Element*>(Shape<Int<PageEntryPerThread>>{}));

  using TensorPageOffset = decltype(make_tensor<mute::tuple<int32_t, int32_t>>(Shape<Int<PageEntryPerThread>>{}));

  // Permutation traits
  static constexpr int MmaAtomN = 8;
  static constexpr int Fragment = ElementsPerLoad;  // Sts vector width
  static constexpr int Repeats  = TileN / (MmaAtomN * Fragment);
  static_assert(TileN % 64 == 0);
  using PermuteTile =
      decltype(filter(make_ordered_layout(Shape<Int<MmaAtomN>, Int<Fragment>, Int<Repeats>>{}, Step<_2, _1, _3>{})));

  int32_t const* const ptr_page_table;
  TensorPageTable      mPageTable;

  int                    bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;
  int const              thread_idx;
  int const              seqlen_k, leftpad_k;
  TensorKV               mK, mV;
  RobustDescriptor       desc_K, desc_V, desc_page_table;
  TensorPageOffset       tPrPageOffsetK, tPrPageOffsetV;
  ThrGmemTiledCopy const gmem_thr_copy_kv;

  TensorKVPtr tPrVPtr;

  mutlass::FastDivmod const& page_size_divmod;

  MUTLASS_DEVICE
  PagedKVManager(int const* const           ptr_page_table,
                 ShapePageTable const&      shape_page_table,
                 StridePageTable const&     stride_page_table,
                 RobustDescriptor const&    desc_page_table,
                 Element*                   ptr_K,
                 ShapeKV const&             shape_K,
                 StrideKV const&            stride_K,
                 RobustDescriptor const&    desc_K,
                 Element*                   ptr_V,
                 int const&                 headdim_v,
                 StrideKV const&            stride_V,
                 RobustDescriptor const&    desc_V,
                 mutlass::FastDivmod const& page_size_divmod,
                 int const                  seqlen_k,
                 int const                  leftpad_k,
                 int const                  thread_idx,
                 int const                  bidb_kv,
                 int const                  bidh_kv,
                 int                        bidb_kv_idx)
      : ptr_page_table(ptr_page_table),
        desc_page_table(desc_page_table),
        bidb_kv_idx(bidb_kv_idx),
        bidb_kv_idx_prev(bidb_kv_idx),
        desc_K(desc_K),
        desc_V(desc_V),
        gmem_thr_copy_kv(GmemTiledCopy{}.get_thread_slice(thread_idx)),
        seqlen_k(seqlen_k),
        leftpad_k(leftpad_k),
        thread_idx(thread_idx),
        page_size_divmod(page_size_divmod) {
    mPageTable   = make_tensor(make_gmem_ptr(ptr_page_table), shape_page_table, stride_page_table)(bidb_kv, _);
    mK           = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh_kv, _);
    auto shape_V = make_shape(get<0>(shape_K), headdim_v, get<2>(shape_K), get<3>(shape_K));
    mV           = make_tensor(make_gmem_ptr(ptr_V), shape_V, stride_V)(_, _, bidh_kv, _);
  }

  template <bool FirstIter = false>
  MUTLASS_DEVICE void load_page_table_for_tme(int const n_block) {
    if constexpr (IsPagedKV) {
      bidb_kv_idx = mPageTable(n_block);
    } else {
      n_block_idx = n_block;
    }
    if constexpr (FirstIter && !IsKVSameIter) {
      bidb_kv_idx_prev = bidb_kv_idx;
      n_block_idx_prev = n_block_idx;
    }
  }

  // K page offsets must use the same lane grouping as the later K pointer broadcast. Rotary store may use
  // GmemThreadsPerRow=4 for qk=64, while V keeps the default grouping used by load_V/store_V.
  template <bool FirstIter = false, bool PermuteK = true, bool PermuteV = false, int KThreadsPerRow = GmemThreadsPerRow>
  MUTLASS_DEVICE void load_page_table_for_lsu(int const n_block) {
    static_assert(NumThreads % KThreadsPerRow == 0);
    if constexpr (IsPagedKV) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < PageEntryPerThread; ++i) {
        int32_t const k_row = i * NumThreads + (NumThreads / KThreadsPerRow) * (thread_idx % KThreadsPerRow) +
                              (thread_idx / KThreadsPerRow);
        int32_t const v_row = i * NumThreads + (NumThreads / GmemThreadsPerRow) * (thread_idx % GmemThreadsPerRow) +
                              (thread_idx / GmemThreadsPerRow);
        int32_t const k_permute_row = PermuteTile{}(k_row);
        int32_t const v_permute_row = PermuteTile{}(v_row);

        int32_t const k_row_idx         = n_block * TileN + k_row;
        int32_t const k_permute_row_idx = n_block * TileN + k_permute_row;
        int32_t const v_row_idx         = n_block * TileN + v_row;
        int32_t const v_permute_row_idx = n_block * TileN + v_permute_row;

        // K
        {
          int32_t page_idx, page_offset;
          if constexpr (PermuteK) {
            page_idx = page_size_divmod.divmod(page_offset, k_permute_row_idx + leftpad_k);
          } else {
            page_idx = page_size_divmod.divmod(page_offset, k_row_idx + leftpad_k);
          }
          int32_t page;
          mute::MP31_ROBUST_LOAD<int32_t>::copy(mPageTable(page_idx), page, true, desc_page_table);
          tPrPageOffsetK(i) = {page, page_offset};
        }

        // V
        {
          int32_t page_idx, page_offset;
          if constexpr (PermuteV) {
            page_idx = page_size_divmod.divmod(page_offset, v_permute_row_idx + leftpad_k);
          } else {
            page_idx = page_size_divmod.divmod(page_offset, v_row_idx + leftpad_k);
          }
          int32_t page;
          mute::MP31_ROBUST_LOAD<int32_t>::copy(mPageTable(page_idx), page, true, desc_page_table);
          tPrPageOffsetV(i) = {page, page_offset};
        }
      }
      if constexpr (FirstIter && !IsKVSameIter) {
        compute_V_ptr();
      }
    }
  }

  MUTLASS_DEVICE
  auto compute_K_ptr() {
    TensorKVPtr tPrKPtr;

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PageEntryPerThread; ++i) {
      auto [page_idx, page_offset] = tPrPageOffsetK(i);
      tPrKPtr(i)                   = &mK(page_offset, _0{}, page_idx);
    }
    return tPrKPtr;
  }

  MUTLASS_DEVICE
  auto compute_V_ptr() {
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PageEntryPerThread; ++i) {
      auto [page_idx, page_offset] = tPrPageOffsetV(i);
      tPrVPtr(i)                   = &mV(page_offset, _0{}, page_idx);
    }
  }

  template <class TensorK>
  MUTLASS_DEVICE void load_K(int const n_block, TensorK&& sK) {
    Tensor cK   = make_identity_tensor(Shape<Int<TileN>, Int<HeadDimQK>>{});
    Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
    Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);  // cpy, cpy_seq, cpy_hd

    Tensor tPrKPtr = compute_K_ptr();

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tKsK); ++i) {
      // TODO: check use mutlass address space
      Element const* k_ptr = (Element const*)__musa_ptr_gen_to_global(
          (void*)(__shfl_sync(0xffffffff,
                              reinterpret_cast<uint64_t>(tPrKPtr(i / GmemThreadsPerRow)),
                              i % GmemThreadsPerRow,
                              GmemThreadsPerRow)));
      Tensor mK_paged_cur      = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<HeadDimQK>>{});
      Tensor mK_paged_cur_copy = mute::tiled_divide(mK_paged_cur, Shape<Int<ElementsPerLoad>>{});

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tKsK); ++j) {
        int const j_idx = get<1>(tKcK(_0{}, _0{}, j)) / ElementsPerLoad;
        mute::copy(GmemTiledCopy{}.with(desc_K).with(true), mK_paged_cur_copy(_, j_idx), tKsK(_, i, j));
      }
    }
  }

  template <class TensorV>
  MUTLASS_DEVICE void load_V(int const n_block, TensorV&& sV) {
    Tensor cV   = make_identity_tensor(Shape<Int<TileN>, Int<HeadDimVO>>{});
    Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
    Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);  // cpy, cpy_seq, cpy_hd

    if constexpr (IsKVSameIter) {
      compute_V_ptr();
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tVsV); ++i) {
      // TODO: check use mutlass address space
      Element const* v_ptr = (Element const*)__musa_ptr_gen_to_global(
          (void*)(__shfl_sync(0xffffffff,
                              reinterpret_cast<uint64_t>(tPrVPtr(i / GmemThreadsPerRow)),
                              i % GmemThreadsPerRow,
                              GmemThreadsPerRow)));
      Tensor mV_paged_cur      = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<HeadDimVO>>{});
      Tensor mV_paged_cur_copy = mute::tiled_divide(mV_paged_cur, Shape<Int<ElementsPerLoad>>{});

      // TODO: clear oob if needed
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tVsV); ++j) {
        int const j_idx = get<1>(tVcV(_0{}, _0{}, j)) / ElementsPerLoad;
        mute::copy(GmemTiledCopy{}.with(desc_V).with(true), mV_paged_cur_copy(_, j_idx), tVsV(_, i, j));
      }
    }
    if constexpr (!IsKVSameIter) {
      compute_V_ptr();
    }
  }

  template <class TensorK>
  MUTLASS_DEVICE void store_K(int const n_block, TensorK&& tKrK) {
    Tensor tPrKPtr = compute_K_ptr();

    auto gmem_thr0_copy_kv = GmemTiledCopyKVStore{}.get_thread_slice(_0{});

    Tensor cK    = make_identity_tensor(Shape<Int<TileN>, Int<HeadDimQK>>{});
    Tensor tKcK  = gmem_thr_copy_kv.partition_S(cK);
    Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

    GmemTiledCopyKVStore gmem_tiled_copy_kv_store;

    int const seqlenk_row_limit = std::min(seqlen_k - n_block * TileN, TileN) - get<0>(tKcK(_0{}, _0{}, _0{}));

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tKrK); ++i) {
      int const  row         = get<0>(t0KcK(_0{}, i, _0{}));
      bool const should_load = row < seqlenk_row_limit;
      // TODO: check use mutlass address space
      Element* k_ptr = (Element*)__musa_ptr_gen_to_global(
          (void*)(__shfl_sync(0xffffffff,
                              reinterpret_cast<uint64_t>(tPrKPtr(i / GmemThreadsPerRow)),
                              i % GmemThreadsPerRow,
                              GmemThreadsPerRow)));

      Tensor mK_paged_cur      = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<HeadDimQK>>{});
      Tensor mK_paged_cur_copy = tiled_divide(mK_paged_cur, Shape<Int<ElementsPerLoad>>{});

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tKrK); ++j) {
        int const j_idx = get<1>(tKcK(_0{}, _0{}, j)) / ElementsPerLoad;
        copy(gmem_tiled_copy_kv_store.with(desc_K).with(should_load), tKrK(_, i, j), mK_paged_cur_copy(_, j_idx));
      }
    }
  }

  template <class TensorV>
  MUTLASS_DEVICE void store_V(int const n_block, TensorV&& tVrV) {
    if constexpr (IsKVSameIter) {
      compute_V_ptr();
    }

    auto                 gmem_thr0_copy_kv = GmemTiledCopyKVStore{}.get_thread_slice(_0{});
    Tensor               cV                = make_identity_tensor(Shape<Int<TileN>, Int<HeadDimVO>>{});
    Tensor               tVcV              = gmem_thr_copy_kv.partition_S(cV);
    Tensor               t0VcV             = gmem_thr0_copy_kv.partition_S(cV);
    GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
    int const seqlenk_row_limit = std::min(seqlen_k - n_block * TileN, TileN) - get<0>(tVcV(_0{}, _0{}, _0{}));

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tVrV); ++i) {
      bool const should_load = get<0>(t0VcV(_0{}, i, _0{})) < seqlenk_row_limit;
      // TODO: check use mutlass address space
      Element* v_ptr = (Element*)__musa_ptr_gen_to_global(
          (void*)(__shfl_sync(0xffffffff,
                              reinterpret_cast<uint64_t>(tPrVPtr(i / GmemThreadsPerRow)),
                              i % GmemThreadsPerRow,
                              GmemThreadsPerRow)));
      Tensor mV_paged_cur      = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<HeadDimVO>>{});
      Tensor mV_paged_cur_copy = mute::tiled_divide(mV_paged_cur, Shape<Int<ElementsPerLoad>>{});

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tVrV); ++j) {
        int const j_idx = get<1>(tVcV(_0{}, _0{}, j)) / ElementsPerLoad;
        copy(gmem_tiled_copy_kv_store.with(desc_V).with(should_load), tVrV(_, i, j), mV_paged_cur_copy(_, j_idx));
      }
    }

    if constexpr (!IsKVSameIter) {
      compute_V_ptr();
    }
  }

  MUTLASS_DEVICE
  mute::tuple<int, int> get_indices_for_tme_k() {
    return {n_block_idx, bidb_kv_idx};
  }

  MUTLASS_DEVICE
  mute::tuple<int, int> get_indices_for_tme_v() {
    if constexpr (IsKVSameIter) {
      return {n_block_idx, bidb_kv_idx};
    } else {
      mute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};

      bidb_kv_idx_prev = bidb_kv_idx;
      n_block_idx_prev = n_block_idx;
      return indices;
    }
  }
};

}  // namespace mate::attention::fmha
