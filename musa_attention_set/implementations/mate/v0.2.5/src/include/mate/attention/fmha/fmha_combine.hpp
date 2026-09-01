#pragma once

#include <mutlass/array.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>
#include <mutlass/numeric_types.h>

#include <mute/tensor.hpp>

#include "seqlen.hpp"
#include "utils.hpp"

namespace mate::attention::fmha {

using namespace mute;

template <class TileShape_MK_,
          int  MaxSplits_,
          int  kNThreads,
          int  AlignmentLSE_,
          bool Varlen,
          bool HasCuseqlensQ,
          bool HasSequsedQ,
          class Element,
          class ElementPartial>
struct FmhaFwdCombine {
  template <typename T>
  struct MaxOp {
    MUTLASS_DEVICE T operator()(T const& x, T const& y) {
      return x > y ? x : y;
    }
  };

  template <>
  struct MaxOp<float> {
    // This is slightly faster
    MUTLASS_DEVICE float operator()(float const& x, float const& y) {
      return max(x, y);
    }
  };

  template <typename T>
  struct SumOp {
    MUTLASS_DEVICE T operator()(T const& x, T const& y) {
      return x + y;
    }
  };

  template <int THREADS>
  struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template <typename T, typename Operator>
    static MUTLASS_DEVICE T run(T x, Operator& op) {
      constexpr int OFFSET = THREADS / 2;
      x                    = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
      return Allreduce<OFFSET>::run(x, op);
    }
  };

  template <>
  struct Allreduce<2> {
    template <typename T, typename Operator>
    static MUTLASS_DEVICE T run(T x, Operator& op) {
      x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
      return x;
    }
  };

  template <typename Engine, typename Layout, typename EngineOut>
  MUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const& tensor, Tensor<EngineOut, Layout>& out) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using From_type = typename Engine::value_type;
    using To_type   = typename EngineOut::value_type;
    static constexpr int FragmentSize =
        std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(MUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag    = recast<mutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<mutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    mutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(frag); ++i) {
      out_frg[i] = convert_op(frag[i]);
    }
  }

  using TileShape_MK                      = TileShape_MK_;
  static constexpr int SmemAlignmentBytes = 128;
  static constexpr int kMaxSplits         = MaxSplits_;
  static constexpr int AlignmentLSE       = std::min(AlignmentLSE_, int(128 / 8 / sizeof(float)));
  static_assert(AlignmentLSE >= 1);

  static constexpr uint32_t MaxThreadsPerBlock         = kNThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;

  static constexpr int TileM   = get<0>(TileShape_MK{});
  static constexpr int kBlockK = get<1>(TileShape_MK{});

  static constexpr int kGmemElemsPerLoad = sizeof(mute::uint128_t) / sizeof(ElementPartial);
  static_assert(kBlockK % kGmemElemsPerLoad == 0, "kBlockK must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBlockKGmem        = kBlockK % 128 == 0 ? 128 : (kBlockK % 64 == 0 ? 64 : 32);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0,
                "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
  using GmemCopyAtom   = mute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>;
  using GmemLayoutAtom = Layout<Shape<Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
  static_assert(TileM % MUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
  using GmemTiledCopyAccum = decltype(make_tiled_copy(
      GmemCopyAtom{}, GmemLayoutAtom{}, Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 4 vals per load
  using GmemTiledCopy =
      decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                               GmemLayoutAtom{},
                               Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 4 vals per load

  using AlignmentTypeLSE                    = mute::uint_byte_t<static_cast<int>(sizeof(float)) * AlignmentLSE>;
  static constexpr int kGmemElemsPerLoadLSE = sizeof(AlignmentTypeLSE) / sizeof(float);
  static_assert(TileM % kGmemElemsPerLoadLSE == 0, "TileM must be a multiple of kGmemElemsPerLoadLSE");
  static_assert(TileM % 8 == 0, "TileM must be a multiple of 8");
  static constexpr int TileMSmem =
      TileM % 128 == 0 ? 128 : (TileM % 64 == 0 ? 64 : (TileM % 32 == 0 ? 32 : (TileM % 16 == 0 ? 16 : 8)));
  static constexpr int kGmemThreadsPerRowLSE = TileMSmem / kGmemElemsPerLoadLSE;
  static_assert(MaxThreadsPerBlock % kGmemThreadsPerRowLSE == 0,
                "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRowLSE");
  using GmemLayoutAtomLSE = Layout<Shape<Int<MaxThreadsPerBlock / kGmemThreadsPerRowLSE>, Int<kGmemThreadsPerRowLSE>>,
                                   Stride<Int<kGmemThreadsPerRowLSE>, _1>>;
  static_assert(kMaxSplits % MUTE_STATIC_V(shape<0>(GmemLayoutAtomLSE{})) == 0);
  using GmemCopyAtomLSE = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentLSE * sizeof(float) * 8>, float>;
  using GmemTiledCopyLSE =
      decltype(make_tiled_copy(GmemCopyAtomLSE{},
                               GmemLayoutAtomLSE{},
                               Layout<Shape<_1, Int<kGmemElemsPerLoadLSE>>>{}));  // Val layout, 4 vals per load

  // Otherwise we get IMA when some threads access sLSE, as we're not doing any masking
  static_assert((TileM * kMaxSplits * AlignmentLSE) % kNThreads == 0,
                "kNThreads must divide TileM * kMaxSplits * AlignmentLSE");
  // This works for TileMSmem = 8, 16, 32, 64, 128, no bank conflicts
  using SmemLSESwizzle = std::conditional_t<TileMSmem == 8,
                                            Swizzle<5, 0, 5>,
                                            std::conditional_t<TileMSmem == 16, Swizzle<4, 0, 4>, Swizzle<3, 2, 3>>>;
  using SmemLayoutAtomLSE =
      decltype(composition(SmemLSESwizzle{}, Layout<Shape<Int<8>, Int<TileMSmem>>, Stride<Int<TileMSmem>, _1>>{}));
  using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<kMaxSplits>, Int<TileM>>{}));

  // We want each column (kMaxSplits) to be processed by threads in the same warp.
  // To reduce the number of shuffles, we want as few threads on the same column as possible.
  // E.g., if TileM is divisible by 64, and there are 256 threads, we want 4 threads (0, 1, 2, 4) per column
  // have have 64 such quads.
  static_assert(MaxThreadsPerBlock % TileMSmem == 0, "MaxThreadsPerBlock must be a multiple of TileMSmem");
  static constexpr int kSmemThreadsPerColLSEt = MaxThreadsPerBlock / TileMSmem;
  static_assert(mutlass::NumThreadsPerWarp % kSmemThreadsPerColLSEt == 0,
                "kSmemThreadsPerColLSEt must divide NumThreadsPerWarp");
  using S2RLayoutAtomLSE = Layout<Shape<Int<kSmemThreadsPerColLSEt>, Int<MaxThreadsPerBlock / kSmemThreadsPerColLSEt>>>;
  using S2RTiledCopyLSE =
      decltype(make_tiled_copy(mute::Copy_Atom<mute::DefaultCopy, float>{}, S2RLayoutAtomLSE{}, Layout<_1>{}));

  using ShapeOPartial    = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, num_splits, head, batch)
  using StrideOPartial   = Stride<int64_t, _1, int64_t, int64_t, int64_t>;
  using ShapeLSEPartial  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, num_splits, head, batch)
  using StrideLSEPartial = Stride<_1, int64_t, int64_t, int64_t>;      // (seqlen, num_splits, head, batch)
  using ShapeO           = Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
  using StrideO          = Stride<int64_t, _1, int64_t, int64_t>;
  using ShapeLSE         = Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
  using StrideLSE        = Stride<_1, int64_t, int64_t>;      // (seqlen, head, batch)

  struct SharedStorage : mute::aligned_struct<128> {
    mute::array_aligned<float, cosize_v<SmemLayoutLSE>> smem_lse_partial;
    mute::array_aligned<int, TileM>                     smem_max_valid_split;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    ElementPartial const* const ptr_O_partial;
    ShapeOPartial const         shape_O_partial;
    StrideOPartial const        stride_O_partial;
    float const* const          ptr_LSE_partial;
    ShapeLSEPartial const       shape_LSE_partial;
    StrideLSEPartial const      stride_LSE_partial;
    Element* const              ptr_O;
    StrideO const               stride_O;
    float* const                ptr_LSE;
    StrideLSE const             stride_LSE;
    uint32_t const* const       cu_seqlens             = nullptr;
    uint32_t const* const       seqused                = nullptr;
    int const* const            num_splits_dynamic_ptr = nullptr;
    int const* const            varlen_batch_idx_ptr   = nullptr;
  };

  // Kernel entry point API
  struct Params {
    ElementPartial const* const ptr_O_partial;
    ShapeOPartial const         shape_O_partial;
    StrideOPartial const        stride_O_partial;
    float const* const          ptr_LSE_partial;
    ShapeLSEPartial const       shape_LSE_partial;
    StrideLSEPartial const      stride_LSE_partial;
    Element* const              ptr_O;
    StrideO const               stride_O;
    float* const                ptr_LSE;
    StrideLSE const             stride_LSE;
    mutlass::FastDivmod         seqlen_divmod, head_divmod;
    uint32_t const* const       cu_seqlens             = nullptr;
    uint32_t const* const       seqused                = nullptr;
    int const* const            num_splits_dynamic_ptr = nullptr;
    int const* const            varlen_batch_idx_ptr   = nullptr;
  };

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    assert(get<1>(args.shape_LSE_partial) <= kMaxSplits);
    return {
        args.ptr_O_partial,
        args.shape_O_partial,
        args.stride_O_partial,
        args.ptr_LSE_partial,
        args.shape_LSE_partial,
        args.stride_LSE_partial,
        args.ptr_O,
        args.stride_O,
        args.ptr_LSE,
        args.stride_LSE,
        mutlass::FastDivmod(get<0>(args.shape_LSE_partial)),
        mutlass::FastDivmod(get<2>(args.shape_LSE_partial)),
        args.cu_seqlens,
        args.seqused,
        args.num_splits_dynamic_ptr,
        args.varlen_batch_idx_ptr,

    };
  }

  MUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    Tensor sLSE           = make_tensor(make_smem_ptr(shared_storage.smem_lse_partial.data()), SmemLayoutLSE{});
    Tensor sMaxValidSplit = make_tensor(make_smem_ptr(shared_storage.smem_max_valid_split.data()), Shape<Int<TileM>>{});

    int const thread_idx          = threadIdx.x;
    int const m_block             = blockIdx.x;
    int const k_block             = blockIdx.y;
    int const maybe_virtual_batch = blockIdx.z;
    int const batch =
        params.varlen_batch_idx_ptr ? params.varlen_batch_idx_ptr[maybe_virtual_batch] : maybe_virtual_batch;
    int const num_splits = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[maybe_virtual_batch]
                                                         : get<1>(params.shape_LSE_partial);

    // If no split, exit
    if (num_splits <= 1) {
      return;
    }
    SeqlenInfoAny<HasCuseqlensQ, HasSequsedQ> seqlen_info{static_cast<uint32_t>(batch),
                                                          static_cast<uint32_t>(size<0>(params.shape_LSE_partial)),
                                                          (params.cu_seqlens),
                                                          (params.seqused)};
    int const                                 offset  = seqlen_info.offset;
    int const                                 seqlen  = seqlen_info.seqlen;
    int                                       max_idx = seqlen * get<2>(params.shape_LSE_partial);  // seqlen * h -> M
    // If exceeds cur_seqlen * num_head_q for current batch, exit.
    if constexpr (Varlen) {
      if (m_block * TileM >= max_idx) {
        return;
      }
    }

    // Used for VarlenQ divmod
    mutlass::FastDivmod seqlen_divmod_dynamic(seqlen);

    // Step 1: load LSE_partial from gmem -> smem
    Tensor mLSEpartial =
        make_tensor(make_gmem_ptr(params.ptr_LSE_partial + offset * get<0>(params.stride_LSE_partial)),
                    select<1, 0, 2, 3>(params.shape_LSE_partial),  // (num_splits, seqlen, head, batch)
                    select<1, 0, 2, 3>(params.stride_LSE_partial))(_, _, _, !HasCuseqlensQ ? batch : 0);

    Tensor           mLSEpartial_copy = tiled_divide(mLSEpartial, Shape<_1, Int<kGmemElemsPerLoadLSE>>{});
    GmemTiledCopyLSE gmem_tiled_copy_LSE;
    auto             gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_thread_slice(thread_idx);
    Tensor           tLSEsLSE          = gmem_thr_copy_LSE.partition_D(sLSE);

    // Construct identity layout for sLSE
    Tensor cLSE =
        make_identity_tensor(make_shape(size<0>(sLSE), size<1>(sLSE)));  // (NUM_SPLITS, BLK_M) -> (num_splits, blk_m)
    // Repeat the partitioning with identity layouts
    Tensor tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE);
    // if (m_block == 0 && k_block == 0 && thread_idx == 0) {
    //   printf("mLSEpartial: ");print(mLSEpartial);print("\n");
    //   printf("mLSEpartial_copy: ");print(mLSEpartial_copy);print("\n");
    //   printf("tLSEcLSE:");print(tLSEcLSE);print("\n");
    // }

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<2>(tLSEcLSE); ++m) {
      int mi  = int(get<1>(tLSEcLSE(_0{}, _0{}, m)));
      int idx = m_block * TileM + mi;
      if (idx < max_idx) {
        int m_idx, bidh;
        if constexpr (!Varlen) {
          bidh = params.seqlen_divmod.divmod(m_idx, idx);
        } else {
          bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
        }
        Tensor mLSEpartial_cur_copy = mLSEpartial_copy(_, _, m_idx, bidh);
        MUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < size<1>(tLSEcLSE); ++s) {
          int si = get<0>(tLSEcLSE(_0{}, s, _0{}));
          // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && thread_idx < 32) { printf("thread_idx = %d, m
          // = %d, s = %d, addr = %p, bank = %d\n", thread_idx, m, s, reinterpret_cast<float *>(&(tLSEsLSE(_0{}, s,
          // m))), reinterpret_cast<int>(&(tLSEsLSE(_0{}, s, m))) / 4 % 32);}
          if (si < num_splits) {
            copy(gmem_tiled_copy_LSE, mLSEpartial_cur_copy(_, si), tLSEsLSE(_, s, m));
          } else {
            fill(tLSEsLSE(_, s, m), -INFINITY);
          }
        }
      } else {
        // We don't need to zero out the rest of the LSEs, as we will not write the output to gmem
        // fill(tLSEsLSE(_, _, m), -INFINITY);
      }
    }

    // Step 2: Load O_partial - compute address.
    GmemTiledCopyAccum gmem_tiled_copy_O_partial;
    auto               gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_thread_slice(thread_idx);
    // Construct identity layout for gO
    Tensor cO = make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O_partial.partition_D(cO);
    Tensor mOpartial =
        make_tensor(make_gmem_ptr(params.ptr_O_partial + offset * get<0>(params.stride_O_partial)),
                    params.shape_O_partial,
                    params.stride_O_partial)(_, _, _, _, !HasCuseqlensQ ? batch : 0);  // (seqlen, d, num_splits, head)

    // Precompute these values to avoid recomputing them in the loop
    Tensor tOmidx  = make_tensor<int>(make_shape(size<1>(tOcO)));
    Tensor tObidh  = make_tensor<int>(make_shape(size<1>(tOcO)));
    Tensor tOrOptr = make_tensor<ElementPartial const*>(make_shape(size<1>(tOcO)));
    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(tOcO); ++m) {
      int mi  = get<0>(tOcO(_0{}, m, _0{}));
      int idx = m_block * TileM + mi;
      if constexpr (!Varlen) {
        tObidh(m) = params.seqlen_divmod.divmod(tOmidx(m), idx);
      } else {
        tObidh[m] = seqlen_divmod_dynamic.divmod(tOmidx(m), idx);
      }
      tOrOptr[m] = &mOpartial(tOmidx(m), k_block * kBlockK, _0{}, tObidh(m));
      if (idx >= max_idx) {
        tObidh[m] = -1;
      }
    }

    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
    MUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O_partial) - k_block * kBlockK;
    }

    // Step 3: load and transpose LSE_partial from smem -> rmem
    __syncthreads();

    S2RTiledCopyLSE s2r_tiled_copy_LSE;
    auto            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_thread_slice(thread_idx);
    Tensor          ts2rsLSE         = s2r_thr_copy_LSE.partition_S(sLSE);
    Tensor          ts2rrLSE         = make_fragment_like(ts2rsLSE);
    copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE);

    // Step 4: compute the final LSE along the split dimension
    Tensor lse_sum  = make_tensor<float>(make_shape(size<2>(ts2rrLSE)));
    Tensor ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE);
    // We compute the max valid split for each row to short-circuit the computation later
    Tensor max_valid_split = make_tensor<int>(make_shape(size<2>(ts2rrLSE)));
    static_assert(MUTE_STATIC_V(size<0>(ts2rrLSE)) == 1);
    // ts2rrLSE size: (, split, row)
    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
      float lse_max = ts2rrLSE(_0{}, _0{}, m);
      MUTLASS_PRAGMA_UNROLL
      for (int s = 1; s < size<1>(ts2rrLSE); ++s) {
        lse_max = max(lse_max, ts2rrLSE(_0{}, s, m));
      }
      MaxOp<float> max_op;
      lse_max           = Allreduce<kSmemThreadsPerColLSEt>::run(lse_max, max_op);
      int max_valid_idx = -1;
      MUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
        if (ts2rrLSE(_0{}, s, m) != -INFINITY) {
          max_valid_idx = get<0>(ts2rcLSE(_0{}, s, _0{}));
        }
      }
      MaxOp<int> max_int_op;
      max_valid_split[m] = Allreduce<kSmemThreadsPerColLSEt>::run(max_valid_idx, max_int_op);
      float lse_max_cur  = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
      float lse_sum_cur  = 0.f;
      MUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
        float scale = expf(ts2rrLSE(_0{}, s, m) - lse_max_cur);
        lse_sum_cur += scale;
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && thread_idx < 32) { printf("thread_idx = %d, m =
        // %d, s = %d, addr = %p, bank = %d\n", thread_idx, m, s, reinterpret_cast<float *>(&(ts2rsLSE(_0{}, s, m))),
        // reinterpret_cast<int>(&(ts2rsLSE(_0{}, s, m))) / 4 % 32);} ts2rsLSE(_0{}, m, s) = scale;
        ts2rrLSE(_0{}, s, m) = scale;
      }
      SumOp<float> sum_op;
      lse_sum_cur   = Allreduce<kSmemThreadsPerColLSEt>::run(lse_sum_cur, sum_op);
      lse_sum(m)    = logf(lse_sum_cur) + lse_max;
      float inv_sum = (lse_sum_cur == 0.f || lse_sum_cur != lse_sum_cur) ? 0.f : 1.f / lse_sum_cur;
      MUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
        ts2rrLSE(_0{}, s, m) *= inv_sum;
      }
    }
    // Store the scales exp(lse - lse_logsum) back to smem
    copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE);

    // Store max_valid_split to smem
    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
      if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {  // Only the thread responsible for s=0 writes to smem
        int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
        if (mi < TileM) {
          sMaxValidSplit[mi] = max_valid_split[m];
        }
      }
    }

    // Step 5: store final LSE back to gmem
    if (k_block == 0) {
      auto   shape_LSE = select<0, 2, 3>(params.shape_LSE_partial);
      Tensor mLSE =
          make_tensor(make_gmem_ptr(params.ptr_LSE + offset * get<0>(params.stride_LSE)), shape_LSE, params.stride_LSE)(
              _, _, !HasCuseqlensQ ? batch : 0);
      MUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
        if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {  // Only the thread responsible for s=0 writes to gmem
          int mi  = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
          int idx = m_block * TileM + mi;
          if (idx < max_idx) {
            int m_idx, bidh;
            if constexpr (!Varlen) {
              bidh = params.seqlen_divmod.divmod(m_idx, idx);
            } else {
              bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
            }
            // printf("thread_idx = %d, m = %d, mi = %d, idx = %d, m_idx = %d, bidh = %d, bidb = %d, lse_sum = %f\n",
            // thread_idx, m, mi, idx, m_idx, bidh, bidb, lse_sum(m));
            mLSE(m_idx, bidh) = lse_sum(m);
          }
        }
      }
    }

    // Step 6: read O_partial from gmem -> smem -> rmem and accumulate the final O
    __syncthreads();
    // Compute max split idx
    int thr_max_valid_split = sMaxValidSplit[get<0>(tOcO(_0{}, _0{}, _0{}))];
    MUTLASS_PRAGMA_UNROLL
    for (int m = 1; m < size<1>(tOcO); ++m) {
      thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[get<0>(tOcO(_0{}, m, _0{}))]);
    }
    Layout tOrOpartial_layout =
        gmem_thr_copy_O_partial.partition_S(make_tensor<ElementPartial>(TileShape_MK{})).layout();
    Tensor tOrOpartial = make_fragment_like<ElementPartial>(tOrOpartial_layout);
    Tensor tOrO        = make_fragment_like<float>(tOrOpartial);
    clear(tOrO);
#pragma unroll 4  // Already tuned for speed
    for (int s = 0; s <= thr_max_valid_split; ++s) {
      // Row-wise scale for current split.
      Tensor scale = make_tensor<float>(make_shape(size<1>(tOrOpartial)));
      MUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tOrOpartial); ++m) {
        int mi   = get<0>(tOcO(_0{}, m, _0{}));  // row inside TileM
        scale(m) = sLSE(s, mi);
      }

      MUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tOrOpartial); ++m) {
        if (tObidh(m) >= 0 && scale(m) > 0.f) {
          ElementPartial const* row_ptr_split =
              reinterpret_cast<ElementPartial const*>(tOrOptr[m] + s * get<2>(params.stride_O_partial));
          Tensor mOpartial_cur      = make_tensor(make_gmem_ptr(row_ptr_split), mOpartial(_0{}, _, _, _0{}).layout());
          Tensor mOpartial_cur_copy = tiled_divide(mOpartial_cur, Shape<Int<kGmemElemsPerLoad>>{});

          MUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < size<2>(tOrOpartial); ++k) {
            if (tOpO(k)) {
              int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;

              copy(gmem_tiled_copy_O_partial, mOpartial_cur_copy(_, k_idx, _0{}), tOrOpartial(_, m, k));

              Tensor rOpartial = make_tensor_like<float>(tOrOpartial(_, m, k));
              convert_type_out(tOrOpartial(_, m, k), rOpartial);

              // Adjust rOpartial
              MUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size<0>(tOrOpartial); ++i) {
                tOrO(i, m, k) += scale(m) * rOpartial[i];
              }
            }
          }
        }
      }
    }

    // Step 7: Write the final O to gmem
    Tensor rO = make_tensor_like<Element>(tOrO);
    convert_type_out(tOrO, rO);
    // shape_o: (seqlen_q, dv, head, batch)
    auto   shape_O = make_shape(get<0>(params.shape_O_partial),
                              get<1>(params.shape_O_partial) - k_block * kBlockK,
                              get<3>(params.shape_O_partial),
                              get<4>(params.shape_O_partial));
    Tensor mO      = make_tensor(
        make_gmem_ptr(params.ptr_O + offset * get<0>(params.stride_O) + k_block * kBlockK * get<1>(params.stride_O)),
        shape_O,
        params.stride_O)(_, _, _, !HasCuseqlensQ ? batch : 0);  // mO: (seqlen, d, head)
    Tensor mO_copy = tiled_divide(
        mO, Shape<_1, Int<kGmemElemsPerLoad>>{});  // mO_copy: (seqlen, d / kGmemElemsPerLoad, kGmemElemsPerLoad, head)
    GmemTiledCopy gmem_tiled_copy_O;
    auto          gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(tOcO); ++m) {
      if (tObidh(m) >= 0) {
        MUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOcO); ++k) {
          int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          if (tOpO(k)) {
            copy(gmem_tiled_copy_O, rO(_, m, k), mO_copy(_, tOmidx(m), k_idx, tObidh(m)));
          }
        }
      }
    }
  }
};

}  // namespace mate::attention::fmha
