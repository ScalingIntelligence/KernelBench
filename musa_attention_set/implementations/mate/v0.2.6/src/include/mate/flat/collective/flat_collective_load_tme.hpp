#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>

namespace mute {
template <class... Args>
struct Copy_Atom;
template <class CopyOperation, class... CopyOpArgs>
struct Copy_Traits;
}  // namespace mute

#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/tensor.hpp>

namespace mate::flat::collective {

using namespace mute;

enum class LoadKind {
  kQ,
  kK,
  kV,
  kG,
};

MUTLASS_HOST_DEVICE constexpr char const* to_string(LoadKind kind) {
  if (kind == LoadKind::kQ) {
    return "Q";
  } else if (kind == LoadKind::kK) {
    return "K";
  } else if (kind == LoadKind::kV) {
    return "V";
  } else if (kind == LoadKind::kG) {
    return "G";
  } else {
    return "unknown loadkind";
  }
}

template <LoadKind kKind, class Pipeline, class Element, class SmemLayout, class TME, int SmemAlignmentBytes = 256>
struct CollectiveLoadTme {
  using SharedStorage = array_aligned<Element, cosize_v<SmemLayout>, SmemAlignmentBytes>;
  using PipelineState = typename Pipeline::PipelineState;

  static constexpr LoadKind kind = kKind;

  TME const&     tme_load;
  Pipeline&      pipeline;
  SharedStorage& storage;

  MUTLASS_DEVICE CollectiveLoadTme(TME const& tme_load, Pipeline& pipeline, SharedStorage& storage)
      : tme_load(tme_load), pipeline(pipeline), storage(storage) {
  }

  template <class ProblemSize, class TileShape, class WorkDesc>
  MUTLASS_DEVICE auto partition_SD(ProblemSize const& problem_size,
                                   TileShape const&   tile_shape,
                                   WorkDesc const&    work_desc) {
    Tensor g_tile = [&] {
      if constexpr (kind == LoadKind::kQ || kind == LoadKind::kK) {
        auto seq_tile = [&] {
          if constexpr (kind == LoadKind::kQ) {
            return get<0>(tile_shape);
          } else {
            return get<1>(tile_shape);
          }
        }();
        auto   head_dim = get<2>(tile_shape);
        Tensor m_input =
            tme_load.get_tme_tensor(make_shape(problem_size.T, head_dim, make_shape(problem_size.Hqk, problem_size.B)));
        Tensor m_head   = m_input(_, _, make_coord(work_desc.qk_head_idx, work_desc.tme_batch()));
        Tensor m_offset = domain_offset(make_coord(work_desc.tme_token(0), _0{}), m_head);
        return local_tile(m_offset, make_shape(seq_tile, head_dim), make_coord(_, _0{}));
      } else if constexpr (kind == LoadKind::kV) {
        auto   seq_tile = get<1>(tile_shape);
        auto   head_dim = get<2>(tile_shape);
        Tensor m_input =
            tme_load.get_tme_tensor(make_shape(head_dim, problem_size.T, make_shape(problem_size.H, problem_size.B)));
        Tensor m_head   = m_input(_, _, make_coord(work_desc.head_idx, work_desc.tme_batch()));
        Tensor m_offset = domain_offset(make_coord(_0{}, work_desc.tme_token(0)), m_head);
        return local_tile(m_offset, make_shape(head_dim, seq_tile), make_coord(_0{}, _));
      } else {
        static_assert(kind == LoadKind::kG, "unsupported CollectiveLoadTme kind");
        auto   seq_tile = get<0>(tile_shape);
        auto   head_dim = get<2>(tile_shape);
        Tensor m_input =
            tme_load.get_tme_tensor(make_shape(problem_size.T, head_dim, make_shape(problem_size.H, problem_size.B)));
        Tensor m_head   = m_input(_, _, make_coord(work_desc.head_idx, work_desc.tme_batch()));
        Tensor m_offset = domain_offset(make_coord(work_desc.tme_token(0), _0{}), m_head);
        return local_tile(m_offset, make_shape(seq_tile, head_dim), make_coord(_, _0{}));
      }
    }();
    Tensor s_tile    = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});
    auto   block_tme = tme_load.get_slice(_0{});
    auto   src       = block_tme.partition_S(g_tile);
    auto   dst       = block_tme.partition_D(s_tile);
    return make_tuple(src, dst);
  }

  template <bool kAcquireBarrier = true, class SrcDst>
  MUTLASS_DEVICE void step(SrcDst const& src_dst, int src_iter, PipelineState& dst_pipe) {
    if (mutlass::canonical_lane_idx() == 0) {
      if constexpr (kAcquireBarrier) {
        pipeline.producer_acquire(dst_pipe);
      }
      uint32_t bar_id = pipeline.producer_get_barrier_id(dst_pipe);
      auto     src    = get<0>(src_dst);
      auto     dst    = get<1>(src_dst);
      copy(tme_load.with(bar_id), src(_, _, _, src_iter), dst(_, _, _, dst_pipe.index()));
      ++dst_pipe;
    }
  }
};

}  // namespace mate::flat::collective
