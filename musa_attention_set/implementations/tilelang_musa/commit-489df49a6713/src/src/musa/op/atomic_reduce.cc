/*!
 * \file tl/musa/op/atomic_reduce.cc
 * \brief MUSA SIMT implementation for tiled atomic maximum and minimum.
 */

#include "backend/common/target_utils.h"
#include "musa/op/atomic.h"

namespace tvm {
namespace tl {

namespace {

LayoutMap InferMUSAAtomicReduceLayout(const AtomicOpBaseNode &op,
                                      const LayoutInferArgs &layout_args,
                                      InferLevel level) {
  return musa::atomic::InferSIMTLayout(op, layout_args, level);
}

Stmt LowerMUSAAtomicReduce(const AtomicOpBaseNode &op,
                           const LowerArgs &lower_args,
                           arith::Analyzer *analyzer) {
  return musa::atomic::LowerSIMT(op, lower_args, analyzer);
}

bool MatchMUSAAtomicReduceTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "musa.AtomicReduce",
      MatchMUSAAtomicReduceTarget,
      InferMUSAAtomicReduceLayout,
      LowerMUSAAtomicReduce,
  });
  return true;
}

const bool musa_atomic_reduce_registered = RegisterMUSAAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
