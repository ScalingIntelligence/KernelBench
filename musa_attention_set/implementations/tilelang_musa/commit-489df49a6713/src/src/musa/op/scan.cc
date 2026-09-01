/*!
 * \file tl/musa/op/scan.cc
 * \brief MUSA implementation registration for tl scan lowering.
 */

#include "backend/common/op/scan.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMUSAScanTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAScan() {
  RegisterCumSumImpl(CumSumImpl{
      "musa.CumSum",
      MatchMUSAScanTarget,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "musa.CumMax",
      MatchMUSAScanTarget,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool musa_scan_registered = RegisterMUSAScan();

} // namespace

} // namespace tl
} // namespace tvm
