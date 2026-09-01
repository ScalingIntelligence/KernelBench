/*!
 * \file tl/musa/op/fill.cc
 * \brief MUSA implementation registration for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMUSAFillTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAFill() {
  RegisterFillImpl(FillImpl{
      "musa.Fill",
      MatchMUSAFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool musa_fill_registered = RegisterMUSAFill();

} // namespace

} // namespace tl
} // namespace tvm
