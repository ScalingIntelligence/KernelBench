#pragma once

#include "mute/tensor.hpp"

namespace mutlass::fmha::collective {

enum class LoadKind {
  LoadQ,
  LoadKWithTme,
  LoadKWithLsu,
  LoadV
};

template <
  LoadKind kind,
  class Pipeline,
  class Element,
  class SmemLayout,
  class CopyParams
>
struct CollectiveLoad {
  using Params = CopyParams;
  using SharedStorage = mute::array_aligned<Element, mute::cosize_v<SmemLayout>, 256>;
  using PipelineState = typename mutlass::PipelineState<Pipeline::Stages>;

  Params const& params;
  Pipeline& pipeline;
  SharedStorage& storage;

  MUTLASS_DEVICE
  CollectiveLoad(Params const& params, Pipeline& pipeline, SharedStorage& storage)
    : params(params), pipeline(pipeline), storage(storage) {}

  template <class ProblemSize, class TileShape, class BlockCoordQHB, class BlockOffset>
  MUTLASS_DEVICE
  auto
  init_g(ProblemSize const& problem_size, TileShape const& tile_shape,
         BlockCoordQHB const& blk_coord, BlockOffset const& blk_offset) {
    auto [Q, K, D_QK, D_VO, H, H_K, B] = problem_size;
    if constexpr (kind == LoadKind::LoadQ) {
      Tensor mQ_in = params.get_tme_tensor(make_shape(Q, D_QK, make_shape(H, B)));   // NDHB
      Tensor mQ = domain_offset(make_coord(get<0>(blk_offset), _0{}, make_coord(_0{}, _0{})), mQ_in);
      Tensor gQ_full = local_tile(mQ, select<0, 2>(tile_shape), make_coord(_, _, _)); // BQ, BD, q, d, h, b
      Tensor gQ = gQ_full(_, _, get<0>(blk_coord), _0{}, make_coord(get<1>(blk_coord), get<3>(blk_coord)));

      return gQ;
    } else if constexpr (kind == LoadKind::LoadKWithTme) {
      if constexpr (size<0>(SmemLayout{}) == 64) {
        Tensor mK_in = params.get_tme_tensor(make_shape(K, D_QK, make_shape(H_K, B)));
        Tensor mK = domain_offset(make_coord(get<1>(blk_offset), _0{}, _0{}), mK_in);

        Tensor gK_full = local_tile(mK, tile_shape, make_coord(_, _, _)); // (atom_n, frag, BK/granularity, BD, k, h, b)
        Tensor gK = gK_full(_,_, _, _0{}, make_coord(get<2>(blk_coord), get<3>(blk_coord))); // (atom_n, frag, BK/granularity, BD, k)

        return gK;
      } else {
        Tensor mK_in = params.get_tme_tensor(make_shape(K, D_QK, B));
        Tensor mK = domain_offset(make_coord(get<1>(blk_offset), _0{}, _0{}), mK_in);

        Tensor gK_full = local_tile(mK, tile_shape, make_coord(_, _, _)); // (atom_n, frag, BK/granularity, BD, k, h, b)
        Tensor gK = gK_full(_,_, _, get<2>(blk_coord), get<3>(blk_coord)); // (atom_n, frag, BK/granularity, BD, k)
        return gK;
      }
    } else if constexpr (kind == LoadKind::LoadKWithLsu) {
      Tensor mK_in = make_tensor(make_gmem_ptr(params.ptr_K),
                                 make_layout(make_shape(K, D_QK, make_shape(H_K, B)), params.stride_K));

      Tensor mK = domain_offset(make_coord(get<1>(blk_offset), _0{}, make_coord(_0{}, _0{})), mK_in);

      Tensor gK_full = local_tile(mK, tile_shape, make_coord(_,_,_));

      Tensor gK = gK_full(_, _, _, _0{}, make_coord(get<2>(blk_coord), get<3>(blk_coord)));

      return gK;
    } else if constexpr (kind == LoadKind::LoadV) {
      Tensor mV_in = params.get_tme_tensor(make_shape(D_VO, K, make_shape(H_K, B))); // DNHB
      Tensor mV = domain_offset(make_coord(_0{}, get<1>(blk_offset), make_coord(_0{}, _0{})), mV_in);
      Tensor gV_full = local_tile(mV, select<1, 2>(tile_shape), make_coord(_, _, _)); // BD, BK, d, k, h, b
      Tensor gV = gV_full(_, _, _0{}, _, make_coord(get<2>(blk_coord), get<3>(blk_coord)));

      return gV;
    }
  }

  template <class ProblemSize, class TileShape, class BlockCoordQHB, class BlockOffset>
  MUTLASS_DEVICE
  auto
  init_state(ProblemSize const& problem_size, TileShape const& tile_shape,
             BlockCoordQHB const& blk_coord, BlockOffset const& blk_offset) {
    Tensor G = init_g(problem_size, tile_shape, blk_coord, blk_offset);
    Tensor S = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});

    if constexpr (kind == LoadKind::LoadKWithLsu) {
      auto thr_copy = params.lsu_K.get_slice(threadIdx.x);
      Tensor tG = thr_copy.partition_S(G);
      Tensor tS = thr_copy.partition_D(S);

      return make_tuple(tG, tS);
    } else {
      auto cta_tme = params.get_slice(0);
      Tensor tG = cta_tme.partition_S(G);
      Tensor tS = cta_tme.partition_D(S);

      return make_tuple(tG, tS);
    }
  }

  template <class State>
  MUTLASS_DEVICE
  void step(int const tile_iter, State const& state, PipelineState& smem_pipe_write) {
    if constexpr (kind == LoadKind::LoadKWithTme || kind == LoadKind::LoadV) {
      pipeline.producer_acquire(smem_pipe_write);
      uint32_t bar_id = pipeline.producer_get_barrier_id(smem_pipe_write);
      copy(params.with(bar_id), get<0>(state)(_,_,_, tile_iter), get<1>(state)(_,_,_,smem_pipe_write.index()));
      ++smem_pipe_write;
    } else if constexpr (kind == LoadKind::LoadKWithLsu) {
      copy(params.lsu_K.with(params.desc_K), get<0>(state)(_,_,_,tile_iter), get<1>(state)(_,_,_,smem_pipe_write.index()));
    }
  }
};

} // namespace mutlass::fmha::collective
