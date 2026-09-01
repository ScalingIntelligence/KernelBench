#pragma once

#include "../../common/mma_mp31_sqmma.hpp"
#include "collective/fmha_collective_load.hpp"
#include "collective/fmha_collective_softmax.hpp"
#include "collective/fmha_common.hpp"
#include "collective/fmha_load_primitive_builder.hpp"
#include "mute/tensor.hpp"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/mutlass.h"
#include "mutlass/numeric_conversion.h"

namespace mate::flash_mla {

using namespace mute;
using namespace mutlass;
using namespace mutlass::fmha::collective;

template <class Element_,
          class ElementAccumulator_,
          class TileShape_,
          class StrideQ_,
          class StrideC_,
          class Fusion_,
          bool VarlenQ_,
          class... Options>
struct FmhaMlaMainloopTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = ElementAccumulator_;

  using TileShape = TileShape_;
  using StrideQ   = StrideQ_;
  using StrideC   = StrideC_;

  using Fusion                                 = Fusion_;
  static constexpr bool VarlenQ                = VarlenQ_;
  static constexpr int  NumLoadWarpSquads      = 1;
  static constexpr int  NumQKMmaWarpSquads     = 4;
  static constexpr int  NumPVMmaWarpSquads     = 4;
  static constexpr int  NumMmaWarpSquads       = std::max(NumQKMmaWarpSquads, NumPVMmaWarpSquads);
  static constexpr int  NumTransposeWarpSquads = 0;

  static constexpr int Alignment = 32 / sizeof_bits_v<Element>;

  using TileShapeD = tuple_element_t<2, TileShape>;

  using TileShapeL = tuple_element_t<0, TileShapeD>;
  using TileShapeR = tuple_element_t<1, TileShapeD>;

  using TileShapeSplit = _64;

  static_assert(TileShapeL{} % TileShapeR{} == 0);
  static_assert(TileShapeL{} % TileShapeSplit{} == 0);

  using TileQPerConsumer = Int<get<0>(TileShape{}) / NumQKMmaWarpSquads>;
  using TileOPerConsumer = Int<get<0>(TileShape{}) / NumPVMmaWarpSquads>;

  using TileShapeQKLatent = Shape<TileQPerConsumer, tuple_element_t<1, TileShape>, TileShapeSplit>;
  using TileShapeQKRope   = Shape<TileQPerConsumer, tuple_element_t<1, TileShape>, TileShapeR>;

  using TileShapePVLatent = Shape<TileOPerConsumer, TileShapeSplit, tuple_element_t<1, TileShape>>;

  using TileShapePV = Shape<TileOPerConsumer, TileShapeL, tuple_element_t<1, TileShape>>;

  static constexpr int IterationsQKLatent = TileShapeL{} / get<2>(TileShapeQKLatent{});
  static constexpr int IterationsQKRope   = TileShapeR{} / get<2>(TileShapeQKRope{});
  static constexpr int IterationsPV       = TileShapeL{} / get<1>(TileShapePVLatent{});

  static constexpr int StagesLatent = TileShapeL{} / tuple_element_t<2, TileShapeQKLatent>{};
  static constexpr int KCStages     = 256 / TileShapeSplit{};

  using CollectiveMmaQKLatent =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQ,
                                                            Alignment,
                                                            Element,
                                                            StrideQ,
                                                            Alignment,
                                                            ElementAccumulator,
                                                            TileShapeQKLatent,
                                                            Shape<_1, _1, _1>,
                                                            _2,
                                                            mutlass::gemm::KernelTme>::CollectiveOp;

  using CollectiveMmaQKRope =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQ,
                                                            Alignment,
                                                            Element,
                                                            StrideQ,
                                                            Alignment,
                                                            ElementAccumulator,
                                                            TileShapeQKRope,
                                                            Shape<_1, _1, _1>,
                                                            _2,
                                                            mutlass::gemm::KernelTme>::CollectiveOp;

  using CollectiveMmaPV = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                                mutlass::arch::OpClassTensorOp,
                                                                                Element,
                                                                                StrideC,
                                                                                Alignment,
                                                                                Element,
                                                                                decltype(select<1, 0, 2>(StrideC{})),
                                                                                Alignment,
                                                                                ElementAccumulator,
                                                                                TileShapePVLatent,
                                                                                Shape<_1, _1, _1>,
                                                                                _2,
                                                                                mutlass::gemm::KernelTme>::CollectiveOp;

  using TiledMmaQK = typename CollectiveMmaQKLatent::TiledMma;
  using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

  static_assert(is_same_v<TiledMmaQK, typename CollectiveMmaQKRope::TiledMma>);

  using SmemLayoutQLatentFull =
      decltype(tile_to_shape(typename CollectiveMmaQKLatent::SmemLayoutAtomA{},
                             Shape<tuple_element_t<0, TileShape>, TileShapeSplit, Int<StagesLatent>>{}));
  using SmemLayoutQRopeFull = decltype(tile_to_shape(typename CollectiveMmaQKRope::SmemLayoutAtomA{},
                                                     Shape<tuple_element_t<0, TileShape>, TileShapeR, Int<1>>{}));

  using SmemLayoutQLatent = decltype(composition(
      SmemLayoutQLatentFull{}, Tile<Layout<Shape<TileQPerConsumer, Int<NumQKMmaWarpSquads>>>, X, X>{}));
  using SmemLayoutQRope   = decltype(composition(SmemLayoutQRopeFull{},
                                               Tile<Layout<Shape<TileQPerConsumer, Int<NumQKMmaWarpSquads>>>, X, X>{}));

  using SmemLayoutKC = decltype(unstageSmemLayout(typename CollectiveMmaQKLatent::SmemLayoutB{}, Int<KCStages>{}));
  using SmemLayoutVC = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<KCStages>{}));

  using SmemLayoutKRope = decltype(unstageSmemLayout(typename CollectiveMmaQKRope::SmemLayoutB{}, Int<1>{}));

  using SmemLayoutP = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutA{}, Int<NumPVMmaWarpSquads>{}));
  using SmemLayoutS =
      decltype(composition(tile_to_shape(typename CollectiveMmaPV::SmemLayoutAtomA{}, take<0, 2>(TileShape{})),
                           Tile<Layout<Shape<TileQPerConsumer, Int<NumQKMmaWarpSquads>>>, X>{}));

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQLatentFull>, 256> smem_q_latent;
    mute::array_aligned<Element, cosize_v<SmemLayoutQRopeFull>, 256>   smem_q_rope;
    union {
      mute::array_aligned<Element, cosize_v<SmemLayoutP>, 256>     smem_p;
      mute::array_aligned<Element, cosize_v<SmemLayoutS>, 256>     smem_s;
      mute::array_aligned<Element, cosize_v<SmemLayoutKRope>, 256> smem_k_rope;
    };
    union {
      mute::array_aligned<Element, cosize_v<SmemLayoutKC>, 256> smem_kc;
      mute::array_aligned<Element, cosize_v<SmemLayoutVC>, 256> smem_vc;
    };
  };

  // static constexpr TME::CacheHint QCacheHint = TME::CacheHint::CACHE_ONCE;
  // static constexpr TME::CacheHint RopeCacheHint = TME::CacheHint::CACHE_ONCE;
  // static constexpr TME::CacheHint KCCacheHint = TME::CacheHint::CACHE_NORMAL;

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  using TmeLoadKCBuilder                 = Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutKC, StrideC>;
  static constexpr int FragmentSize      = TmeLoadKCBuilder::Fragment;
  using FragmentType                     = typename TmeLoadKCBuilder::FragmentType;
  using PermuteTile                      = typename TmeLoadKCBuilder::PermuteTile;
  using TmeTileShapeKC                   = typename TmeLoadKCBuilder::TmeTileShapeKD;

  using TmeLoadKRopeBuilder = Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutKRope, StrideC>;
  static_assert(FragmentSize == TmeLoadKRopeBuilder::Fragment);
  static_assert(is_same_v<FragmentType, typename TmeLoadKRopeBuilder::FragmentType>);
  static_assert(is_same_v<PermuteTile, typename TmeLoadKRopeBuilder::PermuteTile>);
  using TmeTileShapeKRope = typename TmeLoadKRopeBuilder::TmeTileShapeKD;

  using TME_Q_LATENT = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
      take<0, 2>(SmemLayoutQLatentFull{})));

  using TME_Q_ROPE = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
      take<0, 2>(SmemLayoutQRopeFull{})));

  using TME_C      = typename TmeLoadKCBuilder::TME_K;
  using TME_K_ROPE = typename TmeLoadKRopeBuilder::TME_K;

  using TME_C_TRANS = typename CollectiveMmaPV::Params::TME_B;

  using R2SCopyAtom = Copy_Atom<UniversalCopy<FragmentType>, Element>;
  using R2STiledCopy =
      decltype(make_tiled_copy_C(R2SCopyAtom{}, convert_to_permuted_sqmma(TiledMmaQK{}, PermuteTile{})));

  using MainloopPipelineQKLatent = mutlass::Mp31PipelineTmeAsync<KCStages>;
  using MainloopPipelineQKRope   = mutlass::Mp31PipelineTmeAsync<1>;
  using MainloopPipelinePV       = MainloopPipelineQKLatent;

  using PipelineQKLatentParams = typename MainloopPipelineQKLatent::Params;
  using PipelineQKLatentState  = typename MainloopPipelineQKLatent::PipelineState;

  using PipelineQKRopeParams = typename MainloopPipelineQKRope::Params;
  using PipelineQKRopeState  = typename MainloopPipelineQKRope::PipelineState;

  using PipelinePVParams = typename MainloopPipelinePV::Params;
  using PipelinePVState  = typename MainloopPipelinePV::PipelineState;

  using StridePageTable = Stride<int, _1>;

  static constexpr int TmeTransactionBytesQLatent =
      bits_to_bytes(size(take<0, 2>(SmemLayoutQLatent{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesQRope =
      bits_to_bytes(size(take<0, 2>(SmemLayoutQRope{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesC = bits_to_bytes(size(take<0, 2>(SmemLayoutKC{})) * sizeof_bits_v<Element>);

  static constexpr int TmeTransactionBytesKRope =
      bits_to_bytes(size(take<0, 2>(SmemLayoutKRope{})) * sizeof_bits_v<Element>);

  struct Arguments {
    Element const* ptr_Q_latent;
    StrideQ        stride_Q_latent;
    Element const* ptr_Q_rope;
    StrideQ        stride_Q_rope;
    Element const* ptr_C_latent;
    StrideC        stride_C_latent;
    Element const* ptr_K_rope;
    StrideC        stride_K_rope;

    int const*      ptr_page_table = nullptr;
    StridePageTable stride_page_table;
    int const*      ptr_seqlen = nullptr;

    float const sm_scale;

    typename Fusion::Arguments fusion;
    int64_t const*             ptr_cu_seqlens_q;
  };

  struct Params {
    TME_Q_LATENT tme_Q_latent;
    TME_Q_ROPE   tme_Q_rope;

    TME_C      tme_C_latent;
    TME_K_ROPE tme_K_rope;

    TME_C_TRANS tme_C_latent_transpose;

    int const*      ptr_page_table = nullptr;
    StridePageTable stride_page_table;
    int const*      ptr_seqlen = nullptr;

    float const sm_scale;
    float const sm_scale_log2;

    typename Fusion::Params fusion;
    int64_t const*          ptr_cu_seqlens_q;

    RobustDescriptor desc_pt;
  };

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize problem_size, Arguments const& args, void* workspace = nullptr) {
    float const log2e = std::log2(std::exp(1.0f));

    auto [Q_, PageSize_, D_, H_, PageCount_, B_, TotalQ_, MaxQ_] = problem_size;

    auto [L_, R_] = D_;
    int Q         = Q_;
    int PageSize  = PageSize_;
    int H         = H_;
    int PageCount = PageCount_;
    int B         = B_;
    int TotalQ    = TotalQ_;
    int MaxQ      = MaxQ_;
    int L         = L_;
    int R         = R_;

    auto problem_q_latent = [&] {
      if constexpr (!VarlenQ) {
        return make_shape(Q * H, L, make_shape(1, B));
      } else {
        return make_shape(TotalQ * H, L, make_shape(1, 1));
      }
    }();
    auto tme_q_latent =
        make_tme_copy(MP31_TME_LOAD{},
                      make_tensor(make_gmem_ptr(args.ptr_Q_latent), problem_q_latent, args.stride_Q_latent),
                      take<0, 2>(SmemLayoutQLatentFull{}));

    auto problem_q_rope = [&] {
      if constexpr (!VarlenQ) {
        return make_shape(Q * H, R, make_shape(1, B));
      } else {
        return make_shape(TotalQ * H, R, make_shape(1, 1));
      }
    }();
    auto tme_q_rope = make_tme_copy(MP31_TME_LOAD{},
                                    make_tensor(make_gmem_ptr(args.ptr_Q_rope), problem_q_rope, args.stride_Q_rope),
                                    take<0, 2>(SmemLayoutQRopeFull{}));

    auto tme_c_latent_paged = TmeLoadKCBuilder::make_tme_copy(
        make_tensor(args.ptr_C_latent, make_shape(PageSize, L, make_shape(1, PageCount)), args.stride_C_latent));

    auto tme_k_rope_paged = TmeLoadKRopeBuilder::make_tme_copy(
        make_tensor(args.ptr_K_rope, make_shape(PageSize, R, make_shape(1, PageCount)), args.stride_K_rope));

    auto params_pv_latent_paged =
        CollectiveMmaPV::to_underlying_arguments(make_shape(Q * H, L, PageSize, make_shape(1, PageCount)),
                                                 typename CollectiveMmaPV::Arguments{
                                                     args.ptr_Q_latent,
                                                     args.stride_C_latent,
                                                     args.ptr_C_latent,
                                                     select<1, 0, 2>(args.stride_C_latent),
                                                 },
                                                 nullptr);

    RobustDescriptor desc_pt = make_robust_desc(args.ptr_page_table, B * get<0>(args.stride_page_table));

    return {tme_q_latent,
            tme_q_rope,
            tme_c_latent_paged,
            tme_k_rope_paged,
            params_pv_latent_paged.tme_load_b,
            args.ptr_page_table,
            args.stride_page_table,
            args.ptr_seqlen,
            args.sm_scale,
            args.sm_scale * log2e,
            args.fusion,
            args.ptr_cu_seqlens_q,
            desc_pt};
  }

  template <class BlkCoord, class ProblemSize, class Workload>
  MUTE_DEVICE void load(Params const&             params,
                        SharedStorage&            shared_storage,
                        MainloopPipelineQKLatent& pipeline_qk_latent,
                        PipelineQKLatentState&    pipe_state_qk_latent,
                        MainloopPipelineQKRope&   pipeline_qk_rope,
                        PipelineQKRopeState&      pipe_state_qk_rope,
                        MainloopPipelinePV&       pipeline_pv,
                        PipelinePVState&          pipe_state_pv,
                        BlkCoord const&           blk_coord,
                        ProblemSize const&        problem_size,
                        int const                 warp_idx,
                        Workload const&           workload,
                        int const                 seq_q_begin) {
    enum class ProducerWarpRole {
      Load,
      Warp0,
      Warp1,
      Warp2,
    };

    auto producer_warp_role = ProducerWarpRole(warp_idx);

    if (producer_warp_role == ProducerWarpRole::Load) {
      auto [Q_, PageSize, D_, H_, PageCount, B_, TotalQ_, MaxQ_] = problem_size;
      auto [D_latent_, D_rope_]                                  = D_;

      int Q        = Q_;
      int H        = H_;
      int B        = B_;
      int TotalQ   = TotalQ_;
      int MaxQ     = MaxQ_;
      int D_latent = D_latent_;
      int D_rope   = D_rope_;
      // int cur_seqlen = params.ptr_seqlen[get<1>(blk_coord)];
      int cur_seqlen = workload.seqlen_k;

      // Init Q Latent
      auto mQL = [&] {
        if constexpr (!VarlenQ) {
          return params.tme_Q_latent.get_tme_tensor(make_shape(Q * H, D_latent, make_shape(1, B)));
        } else {  // in VarlenQ, Q is seqlen_q for current batch.
          return domain_offset(make_coord(seq_q_begin * H, _0{}, make_coord(_0{}, _0{})),
                               params.tme_Q_latent.get_tme_tensor(make_shape(Q * H, D_latent, make_shape(1, 1))));
        }
      }();
      Tensor gQL_full =
          local_tile(mQL, make_shape(get<0>(TileShape{}), get<1>(TileShapeQKLatent{})), make_coord(_, _, _));
      auto gQL = [&] {
        if constexpr (!VarlenQ) {
          return gQL_full(_, _, get<0>(blk_coord), _, make_coord(_0{}, get<1>(blk_coord)));
        } else {
          return gQL_full(_, _, get<0>(blk_coord), _, _0{});
        }
      }();
      Tensor sQL = make_tensor(make_smem_ptr(shared_storage.smem_q_latent.data()), SmemLayoutQLatent{});

      auto   cta_tme_q_latent = params.tme_Q_latent.get_slice(0);
      Tensor tQLgQL           = cta_tme_q_latent.partition_S(gQL);  // (CP, CPM, CPN, D_s)
      Tensor tQLsQL           = cta_tme_q_latent.partition_D(sQL);  // (CP, CPM, CPN, stage)

      // Init Q Rope
      auto mQR = [&] {
        if constexpr (!VarlenQ) {
          return params.tme_Q_rope.get_tme_tensor(make_shape(Q * H, D_rope, make_shape(1, B)));
        } else {  // in VarlenQ, Q is seqlen_q for current batch.
          return domain_offset(make_coord(seq_q_begin * H, _0{}, make_coord(_0{}, _0{})),
                               params.tme_Q_rope.get_tme_tensor(make_shape(Q * H, D_rope, make_shape(1, 1))));
        }
      }();
      Tensor gQR_full =
          local_tile(mQR, make_shape(get<0>(TileShape{}), get<1>(TileShapeQKRope{})), make_coord(_, _, _));
      auto gQR = [&] {
        if constexpr (!VarlenQ) {
          return gQR_full(_, _, get<0>(blk_coord), _, make_coord(_0{}, get<1>(blk_coord)));
        } else {
          return gQR_full(_, _, get<0>(blk_coord), _, _0{});
        }
      }();
      Tensor sQR = make_tensor(make_smem_ptr(shared_storage.smem_q_rope.data()), SmemLayoutQRope{});

      auto   cta_tme_q_rope = params.tme_Q_rope.get_slice(0);
      Tensor tQRgQR         = cta_tme_q_rope.partition_S(gQR);  // (CP, CPM, CPN, D_s)
      Tensor tQRsQR         = cta_tme_q_rope.partition_D(sQR);  // (CP, CPM, CPN, stage)

      // Init C
      auto problem_size_for_C =
          TmeLoadKCBuilder::get_problem_size(make_shape(Q, PageSize, D_latent, D_rope, 1, 1, PageCount));

      Tensor mCL = params.tme_C_latent.get_tme_tensor(
          make_shape(get<1>(problem_size_for_C), D_latent, make_shape(1, PageCount)));
      Tensor gCL_full = local_tile(mCL, TmeTileShapeKC{}, make_coord(_, _, _));
      Tensor gCL      = gCL_full(_, _, _0{}, _, _);
      Tensor sKC = make_tensor(make_smem_ptr(shared_storage.smem_kc.data()), typename TmeLoadKCBuilder::SmemLayoutK{});

      auto   cta_tme_c = params.tme_C_latent.get_slice(0);
      Tensor tCLgCL    = cta_tme_c.partition_S(gCL);  // (CP, CPM, CPN, D_s, #Page)
      Tensor tCLsCL    = cta_tme_c.partition_D(sKC);  // (CP, CPM, CPN, stage)

      // Init K Rope
      Tensor mKR =
          params.tme_K_rope.get_tme_tensor(make_shape(get<1>(problem_size_for_C), D_latent, make_shape(1, PageCount)));
      Tensor gKR_full = local_tile(mKR, TmeTileShapeKRope{}, make_coord(_, _, _));
      Tensor gKR      = gKR_full(_, _, _0{}, _, _);
      // reuse smem p
      Tensor sKR =
          make_tensor(make_smem_ptr(shared_storage.smem_k_rope.data()), typename TmeLoadKRopeBuilder::SmemLayoutK{});

      auto   cta_tme_k_rope = params.tme_K_rope.get_slice(0);
      Tensor tKRgKR         = cta_tme_k_rope.partition_S(gKR);
      Tensor tKRsKR         = cta_tme_k_rope.partition_D(sKR);

      // Init C latent transpose
      Tensor mCLT =
          params.tme_C_latent_transpose.get_tme_tensor(make_shape(D_latent, PageSize, make_shape(1, PageCount)));
      Tensor gCLT_full = local_tile(mCLT, select<1, 2>(TileShapePVLatent{}), make_coord(_, _, _));
      Tensor gCLT      = gCLT_full(_, _, _, _0{}, _);

      Tensor sCT = make_tensor(make_smem_ptr(shared_storage.smem_vc.data()), SmemLayoutVC{});

      auto   cta_tme_ct = params.tme_C_latent_transpose.get_slice(0);
      Tensor tCLTgCLT   = cta_tme_ct.partition_S(gCLT);
      Tensor tCLTsCLT   = cta_tme_ct.partition_D(sCT);

      // PageTable
      Tensor mPT =
          make_tensor(make_gmem_ptr(params.ptr_page_table), make_shape(B, PageCount), params.stride_page_table);
      Tensor gPT = mPT(get<1>(blk_coord), _);

      auto problem_size_for_fusion = make_shape(Q, cur_seqlen, D_latent, D_latent, H, 1, B);

      Fusion fusion{params.fusion};

      // int fusion_tile_count = fusion.get_trip_count(blk_coord, TileShape{},
      // problem_size_for_fusion); int page_index = 0;

      int page_index      = workload.start_block_idx;
      int start_block_idx = workload.start_block_idx;
      int end_block_idx   = workload.end_block_idx;

      int cur_page, next_page;

      MP31_ROBUST_LOAD<int>::copy(gPT(page_index), cur_page, true, params.desc_pt);
      MP31_ROBUST_LOAD<int>::copy(gPT(page_index + 1), next_page, true, params.desc_pt);
      {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IterationsQKLatent; ++i) {
          pipeline_qk_latent.producer_acquire(pipe_state_qk_latent);
          pipeline_qk_latent.producer_expect_transaction(pipe_state_qk_latent, TmeTransactionBytesQLatent);

          uint32_t bar_id = pipeline_qk_latent.producer_get_barrier_id(pipe_state_qk_latent);
          copy(params.tme_Q_latent.with(bar_id), tQLgQL(_, _, _, i), tQLsQL(_, _, _, i));
          copy(params.tme_C_latent.with(bar_id),
               tCLgCL(_, _, _, i, cur_page),
               tCLsCL(_, _, _, pipe_state_qk_latent.index()));
          ++pipe_state_qk_latent;
        }

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IterationsQKRope; ++i) {
          pipeline_qk_rope.producer_acquire(pipe_state_qk_rope);
          pipeline_qk_rope.producer_expect_transaction(pipe_state_qk_rope, TmeTransactionBytesQRope);
          uint32_t bar_id = pipeline_qk_rope.producer_get_barrier_id(pipe_state_qk_rope);

          copy(params.tme_Q_rope.with(bar_id), tQRgQR(_, _, _, i), tQRsQR(_, _, _, i));
          copy(params.tme_K_rope.with(bar_id),
               tKRgKR(_, _, _, i, cur_page),
               tKRsKR(_, _, _, pipe_state_qk_rope.index()));

          ++pipe_state_qk_rope;
          // Prefetch KR in next iteration
          prefetch(params.tme_K_rope, tKRgKR(_, _, _, i, next_page));
        }

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IterationsPV; ++i) {
          pipeline_pv.producer_acquire(pipe_state_pv);
          uint32_t bar_id = pipeline_pv.producer_get_barrier_id(pipe_state_pv);

          copy(params.tme_C_latent_transpose.with(bar_id),
               tCLTgCLT(_, _, _, i, cur_page),
               tCLTsCLT(_, _, _, pipe_state_pv.index()));
          ++pipe_state_pv;
          // Prefetch KL in next iteration
          prefetch(params.tme_C_latent, tCLgCL(_, _, _, i, next_page));
        }

        ++page_index;
        ++start_block_idx;
      }

      while (start_block_idx < end_block_idx) {
        cur_page = next_page;
        MP31_ROBUST_LOAD<int>::copy(gPT(page_index + 1), next_page, true, params.desc_pt);

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IterationsQKLatent; ++i) {
          pipeline_qk_latent.producer_acquire(pipe_state_qk_latent);
          uint32_t bar_id = pipeline_qk_latent.producer_get_barrier_id(pipe_state_qk_latent);
          copy(params.tme_C_latent.with(bar_id),
               tCLgCL(_, _, _, i, cur_page),
               tCLsCL(_, _, _, pipe_state_qk_latent.index()));
          ++pipe_state_qk_latent;
        }

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IterationsQKRope; ++i) {
          pipeline_qk_rope.producer_acquire(pipe_state_qk_rope);
          uint32_t bar_id = pipeline_qk_rope.producer_get_barrier_id(pipe_state_qk_rope);
          copy(params.tme_K_rope.with(bar_id),
               tKRgKR(_, _, _, i, cur_page),
               tKRsKR(_, _, _, pipe_state_qk_rope.index()));
          ++pipe_state_qk_rope;
          // Prefetch KR in next iteration
          prefetch(params.tme_K_rope, tKRgKR(_, _, _, i, next_page));
        }

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < IterationsPV; ++i) {
          pipeline_pv.producer_acquire(pipe_state_pv);
          uint32_t bar_id = pipeline_pv.producer_get_barrier_id(pipe_state_pv);

          copy(params.tme_C_latent_transpose.with(bar_id),
               tCLTgCLT(_, _, _, i, cur_page),
               tCLTsCLT(_, _, _, pipe_state_pv.index()));
          ++pipe_state_pv;
          // Prefetch KL in next iteration
          prefetch(params.tme_C_latent, tCLgCL(_, _, _, i, next_page));
        }
        ++page_index;
        ++start_block_idx;
      }
    }
  }

  template <class BlkCoord, class ProblemSize, class NamedBarrier, class Workload>
  MUTLASS_DEVICE auto compute(BlkCoord const&           blk_coord,
                              Params const&             params,
                              ProblemSize const&        problem_size,
                              MainloopPipelineQKLatent& pipeline_qk_latent,
                              PipelineQKLatentState&    smem_pipe_k_latent_read,
                              MainloopPipelineQKRope&   pipeline_qk_rope,
                              PipelineQKRopeState&      smem_pipe_k_rope_read,
                              MainloopPipelinePV&       pipeline_pv,
                              PipelinePVState&          smem_pipe_v_read,
                              NamedBarrier&             reuse_p,
                              SharedStorage&            storage,
                              int const                 thread_idx,
                              int const                 consumer_idx,
                              Workload const&           workload) {
    // Prepare QK MMA
    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

    Tensor sQL = make_tensor(make_smem_ptr(storage.smem_q_latent.data()), SmemLayoutQLatent{});
    Tensor sKL = make_tensor(make_smem_ptr(storage.smem_kc.data()), SmemLayoutKC{});

    Tensor tSsQL = thr_mma_qk.partition_A(sQL(make_coord(_, consumer_idx), _, _));
    Tensor tSrQL = thr_mma_qk.make_fragment_A(tSsQL);

    Tensor tSsKL = thr_mma_qk.partition_B(sKL);
    Tensor tSrKL = thr_mma_qk.make_fragment_B(tSsKL);

    Tensor sQR = make_tensor(make_smem_ptr(storage.smem_q_rope.data()), SmemLayoutQRope{});
    Tensor sKR = make_tensor(make_smem_ptr(storage.smem_k_rope.data()), SmemLayoutKRope{});

    Tensor tSsQR = thr_mma_qk.partition_A(sQR(make_coord(_, consumer_idx), _, _));
    Tensor tSrQR = thr_mma_qk.make_fragment_A(tSsQR);

    Tensor tSsKR = thr_mma_qk.partition_B(sKR);
    Tensor tSrKR = thr_mma_qk.make_fragment_B(tSsKR);

    // Prepare PV MMA
    TiledMmaPV tiled_mma_pv;
    auto       thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    Tensor sP = make_tensor(make_smem_ptr(storage.smem_p.data()), SmemLayoutP{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_vc.data()), SmemLayoutVC{});

    Tensor tOsP = thr_mma_pv.partition_A(sP(_, _, consumer_idx));
    Tensor tOrP = thr_mma_pv.make_fragment_A(tOsP);

    Tensor tOsV = thr_mma_pv.partition_B(sV);
    Tensor tOrV = thr_mma_pv.make_fragment_B(tOsV);

    // Prepare R2S
    R2STiledCopy tiled_copy_r2s;
    ThrCopy      thr_copy_r2s = tiled_copy_r2s.get_thread_slice(thread_idx);
    Tensor       tPsP         = thr_copy_r2s.partition_D(sP(_, _, consumer_idx));

    // AccQK index
    Tensor cP_full = make_identity_tensor(take<0, 2>(TileShape{}));
    Tensor cP      = local_tile(cP_full, make_shape(get<0>(TileShapeQKLatent{})), make_coord(consumer_idx));
    Tensor tPcP = convert_to_permuted_sqmma(tiled_mma_qk, PermuteTile{}).get_thread_slice(thread_idx).partition_C(cP);
    int    m_coord = get<0>(blk_coord);

    int start_block_idx = workload.start_block_idx;
    int end_block_idx   = workload.end_block_idx;

    tPcP.data() =
        tPcP.data() + E<0>{} * (m_coord * get<0>(TileShape{})) + E<1>{} * (start_block_idx * get<1>(TileShape{}));

    auto convert_and_sts = [&](auto& accum_qk) {
      Tensor accum_cvt = make_fragment_like<Element>(accum_qk);

      Tensor tPrP = thr_copy_r2s.retile_S(accum_cvt);

      Tensor tCvt_frg = recast<mutlass::Array<Element, FragmentSize>>(accum_cvt);
      Tensor tAcc_frg = recast<mutlass::Array<ElementAccumulator, FragmentSize>>(accum_qk);

      MUTE_UNROLL
      for (int i = 0; i < size(tCvt_frg); ++i) {
        tCvt_frg(i) = mutlass::
            NumericArrayConverter<Element, ElementAccumulator, FragmentSize, FloatRoundStyle::round_to_nearest>{}(
                tAcc_frg(i));
      }

      // TODO: sync before sts
      reuse_p.sync();

      copy(tiled_copy_r2s, tPrP, tPsP);
      __syncwarp();
    };

    Fusion fusion{params.fusion};

    // int kv_tile_count = fusion.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);
    int unmasked_tile_count = fusion.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    int total_tile = fusion.get_trip_count(blk_coord, TileShape{}, problem_size);

    int tile_count_iter = start_block_idx;

    Tensor acc_pv = partition_fragment_C(
        tiled_mma_pv,
        make_shape(
            get<0>(TileShapePVLatent{}), get<1>(TileShapePVLatent{}), TileShapeL{} / get<1>(TileShapePVLatent{})));

    Tensor acc_pv_final = make_tensor(acc_pv.data(), partition_shape_C(tiled_mma_pv, take<0, 2>(TileShapePV{})));
    static_assert(size(acc_pv) == size(acc_pv_final));

    // trigger zero init sqmma
    clear(acc_pv);

    CollectiveSoftmax<Fusion, decltype(params), false> softmax(params, fusion);
    auto                                               softmax_state = softmax.init(acc_pv_final, tiled_mma_pv);

    auto mla_fwd_step = [&](auto first_round, auto enable_fusion) {
      static constexpr bool FirstRound   = decltype(first_round)::value;
      static constexpr bool EnableFusion = decltype(enable_fusion)::value;

      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKLatent{}));

      // trigger zero init sqmma
      clear(acc_qk);

      // MMA QK
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < IterationsQKLatent; ++i) {
        pipeline_qk_latent.consumer_wait(smem_pipe_k_latent_read);
        int k_read_stage = smem_pipe_k_latent_read.index();
        mute::gemm(tiled_mma_qk, tSrQL(_, _, _, i), tSrKL(_, _, _, k_read_stage), acc_qk);

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        pipeline_qk_latent.consumer_release(smem_pipe_k_latent_read);
        ++smem_pipe_k_latent_read;
        if (i == 3) {
          reuse_p.sync();
        }
      }

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < IterationsQKRope; ++i) {
        pipeline_qk_rope.consumer_wait(smem_pipe_k_rope_read);
        int k_read_stage = smem_pipe_k_rope_read.index();
        mute::gemm(tiled_mma_qk, tSrQR(_, _, _, i), tSrKR(_, _, _, k_read_stage), acc_qk);

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        pipeline_qk_rope.consumer_release(smem_pipe_k_rope_read);
        ++smem_pipe_k_rope_read;
      }

      // Softmax
      if constexpr (FirstRound) {
        softmax.step(acc_qk, tiled_mma_qk, softmax_state, tPcP, problem_size);
      } else {
        softmax.template step<EnableFusion>(
            acc_qk, tiled_mma_qk, softmax_state, acc_pv_final, tiled_mma_pv, tPcP, problem_size);
      }

      // Sts
      convert_and_sts(acc_qk);

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < IterationsPV; ++i) {
        pipeline_pv.consumer_wait(smem_pipe_v_read);
        int pv_read_stage = smem_pipe_v_read.index();

        mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, pv_read_stage), acc_pv(_, _, _, i));

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        pipeline_pv.consumer_release(smem_pipe_v_read);

        ++smem_pipe_v_read;
        if (i == 3) {
          reuse_p.sync();
        }
      }

      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShape{});
      ++tile_count_iter;
    };

    mla_fwd_step(mute::true_type{}, mute::false_type{});

    while (tile_count_iter < min(unmasked_tile_count, end_block_idx)) {
      mla_fwd_step(mute::false_type{}, mute::false_type{});
    }

    while (tile_count_iter < end_block_idx) {
      mla_fwd_step(mute::false_type{}, mute::true_type{});
    }

    Tensor lse = softmax.tail(softmax_state, acc_pv_final, tiled_mma_pv);

    return make_tuple(acc_pv_final, lse);
  }
};

}  // namespace mate::flash_mla
