/*!
 * \file tl/musa/op/atomic_add.cc
 * \brief MUSA SIMT implementation for tiled atomic addition.
 */

#include "op/atomic_add.h"
#include "backend/common/target_utils.h"
#include "musa/op/atomic.h"

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

namespace {

bool UseTMA(const AtomicAddNode &op) {
  if (auto value = op.annotations.Get("use_tma")) {
    if (const auto *flag = value.value().as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

} // namespace

struct AtomicAdd {
  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    ICHECK(!UseTMA(op))
        << "TME atomic_add is not supported by the MUSA backend";
    return atomic::InferSIMTLayout(op, layout_args, level);
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    ICHECK(!UseTMA(op))
        << "TME atomic_add is not supported by the MUSA backend";
    return atomic::LowerSIMT(op, lower_args, analyzer);
  }
};

} // namespace musa

namespace {

bool MatchMUSAAtomicAddTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "musa.AtomicAdd",
      MatchMUSAAtomicAddTarget,
      musa::AtomicAdd::InferLayout,
      musa::AtomicAdd::Lower,
  });
  return true;
}

const bool musa_atomic_add_registered = RegisterMUSAAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm
