/*!
 * \file tl/musa/op/atomic.h
 * \brief Shared MUSA SIMT lowering for tiled atomic operations.
 */

#ifndef TVM_TL_MUSA_OP_ATOMIC_H_
#define TVM_TL_MUSA_OP_ATOMIC_H_

#include "backend/common/op/atomic_reduce.h"

namespace tvm {
namespace tl {
namespace musa {
namespace atomic {

inline LayoutMap InferSIMTLayout(const AtomicOpBaseNode &op,
                                 const LayoutInferArgs &layout_args,
                                 InferLevel level) {
  return backend::atomic_reduce::InferSIMTLayout(op, layout_args, level);
}

inline Stmt LowerSIMT(const AtomicOpBaseNode &op, const LowerArgs &lower_args,
                      arith::Analyzer *analyzer) {
  auto simt_loop = backend::atomic_reduce::MakeSIMTLoop(op, analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);
  for (auto level :
       {InferLevel::kCommon, InferLevel::kStrict, InferLevel::kFree}) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(
      fused_loop, loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true,
      /*should_vectorize=*/true, par_op->LoopLayoutRequiresPaddingGuard());
}

} // namespace atomic
} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_MUSA_OP_ATOMIC_H_
