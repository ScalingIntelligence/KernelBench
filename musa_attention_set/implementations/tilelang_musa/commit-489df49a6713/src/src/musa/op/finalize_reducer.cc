/*!
 * \file tl/musa/op/finalize_reducer.cc
 * \brief MUSA implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "musa/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

struct FinalizeReducer
    : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target target) { return TargetMUSAGetWarpSize(target); }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset, PrimExpr,
                                        int batch, int workspace_stride,
                                        Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", tl::SyncThreadsBarrier, "
       << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset, PrimExpr,
                                         Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }
};

} // namespace musa

namespace {

bool MatchMUSAFinalizeReducerTarget(Target target) {
  return TargetIsMUSA(target);
}

bool RegisterMUSAFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "musa.FinalizeReducer",
      MatchMUSAFinalizeReducerTarget,
      musa::FinalizeReducer::Lower,
  });
  return true;
}

const bool musa_finalize_reducer_registered = RegisterMUSAFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
