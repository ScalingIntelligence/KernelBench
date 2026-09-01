#pragma once

#include <mute/tensor.hpp>

namespace mate::attention::fmha {

using namespace mute;

template <class Element, int HeadRatio, int TileM, int HeadDim, int NumThread>
struct PackGQAManager {
  // used by LSU Load Q
  static constexpr int BytePerRow = HeadDim * sizeof(Element);
  static constexpr int BlockKGmem = (BytePerRow % 128 == 0 ? 128 : (BytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int GmemElemsPerLoad = sizeof(mute::uint128_t) / sizeof(Element);
  static constexpr int GmemThreadPerRow = BlockKGmem / GmemElemsPerLoad;

  using GmemCopyAtomQ = mute::Copy_Atom<MP31_ROBUST_LDGSTS<mute::uint128_t>, Element>;
  using GmemLayoutAtomQ =
      Layout<Shape<Int<NumThread / GmemThreadPerRow>, Int<GmemThreadPerRow>>, Stride<Int<GmemThreadPerRow>, _1>>;
  using GmemTiledCopyQ =
      decltype(make_tiled_copy(GmemCopyAtomQ{}, GmemLayoutAtomQ{}, Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));

  template <int NumThrPerM, class GEngine, class GLayout, class CEngine, class CLayout>
  MUTLASS_DEVICE static auto compute_ptr(Tensor<GEngine, GLayout> const &gtensor,  // ((qhead_per_khead, seqlen_q))
                                         Tensor<CEngine, CLayout> const &ctensor,  // (NumRowPerThr)
                                         int const                       thread_idx,
                                         int const                       m_block_idx) {
    constexpr int NumRowPerThr = size(CLayout{});
    constexpr int NumPtrPerThr = ceil_div(NumRowPerThr, NumThrPerM);

    using TensorType = typename GEngine::value_type;
    Tensor ptensor   = make_tensor<TensorType const *>(Shape<Int<NumPtrPerThr>>{});

    uint32_t const row_base    = m_block_idx * TileM;
    uint32_t const thr_row_idx = thread_idx % NumThrPerM;
    MUTE_UNROLL
    for (int i = 0; i < NumPtrPerThr; ++i) {
      uint32_t const row = i * NumThread + row_base + get<0>(ctensor(thr_row_idx));

      auto const head_q_idx   = row % HeadRatio;
      auto const seqlen_q_idx = row / HeadRatio;

      ptensor(i) = &gtensor(make_coord(make_coord(head_q_idx, seqlen_q_idx)));
    }

    return ptensor;
  }

  template <bool IsQv = false, class Params, class GEngine, class GLayout, class SEngine, class SLayout>
  MUTLASS_DEVICE static void load_Q(Params const                   &params,
                                    Tensor<GEngine, GLayout> const &mQ,  // ((qhead_per_khead, seqlen_q), headdim_q)
                                    Tensor<SEngine, SLayout>       &sQ,  // (TileM, HeadDim)
                                    int const                       thread_idx,
                                    int const                       m_block_idx) {
    auto robust_desc_Q = [&]() {
      if constexpr (IsQv) {
        return params.desc_Qv;
      } else {
        return params.desc_Q;
      }
    }();
    // use lse load Q
    auto gmem_thr_copy_Q = GmemTiledCopyQ{}.get_thread_slice(thread_idx);

    Tensor cQ   = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    Tensor mQ_0   = mQ(_, _0{});
    Tensor tQcQ_m = tQcQ(_0{}, _, _0{});
    Tensor tQcQ_n = tQcQ(_0{}, _0{}, _);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    MUTE_UNROLL
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ_n(k)) < size<1>(mQ);
    }

    Tensor rPtr = compute_ptr<GmemThreadPerRow>(mQ_0, tQcQ_m, thread_idx, m_block_idx);

    int const row_base = m_block_idx * TileM;
    MUTE_UNROLL
    for (int m = 0; m < size<1>(tQsQ); ++m) {
      int const row = row_base + get<0>(tQcQ_m(m));

      Element const *ptrQ = (Element *)__musa_ptr_gen_to_global((void *)(__shfl_sync(
          0xffffffff, reinterpret_cast<uint64_t>(rPtr(m / GmemThreadPerRow)), m % GmemThreadPerRow, GmemThreadPerRow)));

      Tensor gQ_n   = make_tensor(make_gmem_ptr(ptrQ), Shape<Int<HeadDim>>{});
      Tensor tQgQ_n = tiled_divide(gQ_n, Shape<Int<GmemElemsPerLoad>>{});

      MUTE_UNROLL
      for (int k = 0; k < size<2>(tQsQ); ++k) {
        int const ki = get<1>(tQcQ_n(k)) / GmemElemsPerLoad;

        copy(GmemTiledCopyQ{}.with(robust_desc_Q).with(tQpQ(k)), tQgQ_n(_, ki), tQsQ(_, m, k));
      }
    }
  }

  template <bool StoreZero, class EngineS, class LayoutS, class EngineD, class LayoutD, class TiledMMAQK>
  MUTLASS_DEVICE static void store_LSE(TiledMMAQK const               &tiled_mma_qk,
                                       Tensor<EngineS, LayoutS> const &rLSE,  // (ThrTileM)
                                       Tensor<EngineD, LayoutD>       &mLSE,  // ((qhead_per_khead, seqlen_q))
                                       int const                       thread_idx,
                                       int const                       m_block_idx,
                                       int const                       seqlen_o) {
    using ElementSrcLSE = typename EngineS::value_type;
    using ElementDstLSE = typename EngineD::value_type;

    static_assert(is_same_v<ElementSrcLSE, ElementDstLSE>, "ElementSrcLSE and ElementDstLSE must be the same type");
    static_assert(is_same_v<float, ElementDstLSE>, "ElementDstLSE must be float.");

    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

    using StoreTile   = Shape<Int<TileM>, Int<HeadDim>>;
    Tensor cLSE       = make_identity_tensor(StoreTile{});
    Tensor tLSEcLSE   = thr_mma_qk.partition_C(cLSE);
    Tensor tLSEcLSE_m = make_tensor(tLSEcLSE.data(), layout_acc_mn(tiled_mma_qk, tLSEcLSE.layout()))(_, _0{});

    constexpr int NumThrPerM = size<0, 0>(typename TiledMMAQK::AtomLayoutC_TV{});

    Tensor rPtr = compute_ptr<NumThrPerM>(mLSE, tLSEcLSE_m, thread_idx, m_block_idx);

    static_assert(MUTE_STATIC_V(size(tLSEcLSE_m)) <= NumThrPerM, "NumThrPerM must be >= size(rPtr)");
    static_assert(MUTE_STATIC_V(size(rPtr)) == 1, "size(rPtr) must be 1");

    int const row_base = m_block_idx * TileM;
    MUTE_UNROLL
    for (int m = 0; m < size(rLSE); ++m) {
      int const row = row_base + get<0>(tLSEcLSE_m(m));

      ElementDstLSE *pLSE = (ElementDstLSE *)__musa_ptr_gen_to_global(
          (void *)__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(rPtr(_0{})), m % NumThrPerM, NumThrPerM));

      if (get<1>(tLSEcLSE_m(_0{})) == 0 && row < seqlen_o * HeadRatio) {
        if constexpr (StoreZero) {
          *pLSE = -std::numeric_limits<ElementDstLSE>::infinity();
        } else {
          *pLSE = rLSE(m);
        }
      }
    }
  }

  template <bool StoreZero, class EngineS, class LayoutS, class EngineD, class LayoutD, class TiledMMAPV>
  MUTLASS_DEVICE static void store_O(TiledMMAPV const               &tiled_mma_pv,
                                     Tensor<EngineS, LayoutS> const &rO,  // (ThrTileM, ThrHeadDimQ)
                                     Tensor<EngineD, LayoutD>       &mO,  // ((qhead_per_khead, seqlen_q), headdim_q)
                                     int const                       thread_idx,
                                     int const                       m_block_idx,
                                     int const                       seqlen_o) {
    using ElementSrcO = typename EngineS::value_type;
    using ElementDstO = typename EngineD::value_type;

    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    using StoreTile = Shape<Int<TileM>, Int<HeadDim>>;
    Tensor cO       = make_identity_tensor(StoreTile{});
    Tensor tOcO     = thr_mma_pv.partition_C(cO);
    Tensor tOcO_mn  = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma_pv, tOcO.layout()));
    Tensor tOcO_m   = tOcO_mn(_, _0{});
    Tensor tOcO_n   = tOcO_mn(_0{}, _);

    constexpr int NumThrPerM = size<0, 0>(typename TiledMMAPV::AtomLayoutC_TV{});

    Tensor mO_0 = mO(_, _0{});
    Tensor rPtr = compute_ptr<NumThrPerM>(mO_0, tOcO_m, thread_idx, m_block_idx);

    static_assert(mutlass::NumThreadsPerWarp % NumThrPerM == 0, "NumThrPerM must be a divisor of NumThreadsPerWarp");
    static_assert(MUTE_STATIC_V(size(tOcO_m)) <= NumThrPerM, "NumThrPerM must be <= NumThreadsPerWarp");
    static_assert(MUTE_STATIC_V(size(rPtr)) == 1, "size(rPtr) must be 1");

    int row_base = m_block_idx * TileM;
    MUTE_UNROLL
    for (int m = 0; m < size<0>(rO); ++m) {
      int const row = row_base + get<0>(tOcO_m(m));

      ElementDstO *pO_row = (ElementDstO *)__musa_ptr_gen_to_global(
          (void *)__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(rPtr(_0{})), m % NumThrPerM, NumThrPerM));

      if (row < seqlen_o * HeadRatio) {
        Tensor     mO_row = make_tensor(make_gmem_ptr(pO_row), make_shape(size<1>(mO)));
        auto const dim_v  = size<1>(mO);

        MUTE_UNROLL
        for (int n = 0; n < size<1>(rO); ++n) {
          // we do not check boundary along the dim V
          int const col = get<1>(tOcO_n(n));
          if (col < dim_v) {
            // mO_row(col) = ElementDstO(bidb*1000+split_idx*100 + bidh*10 + m_block);
            if constexpr (StoreZero) {
              mO_row(col) = ElementDstO(0.0f);
            } else {
              mO_row(col) = ElementDstO(rO(m, n));
            }
          }
        }
      }
    }
  }

};  // struct PackGQAManager

}  // namespace mate::attention::fmha
