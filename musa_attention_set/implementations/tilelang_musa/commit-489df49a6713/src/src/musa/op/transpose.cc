/*!
 * \file tl/musa/op/transpose.cc
 * \brief MUSA implementation registration for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMUSATransposeTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSATranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "musa.Transpose",
      MatchMUSATransposeTarget,
      backend::Transpose::Lower,
  });
  return true;
}

const bool musa_transpose_registered = RegisterMUSATranspose();

} // namespace

} // namespace tl
} // namespace tvm
