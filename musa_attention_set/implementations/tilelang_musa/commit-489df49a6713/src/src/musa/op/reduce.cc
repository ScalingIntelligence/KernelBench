/*!
 * \file tl/musa/op/reduce.cc
 * \brief MUSA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

void CheckSyncThreadsScope(int reducing_threads, PrimExpr all_threads) {
  if (reducing_threads <= 32) {
    return;
  }
  const int64_t *all_threads_value = as_const_int(all_threads);
  ICHECK(all_threads_value != nullptr)
      << "MUSA tl.reduce with SyncThreadsBarrier requires a compile-time "
         "constant thread extent.";
  ICHECK_EQ(reducing_threads, *all_threads_value)
      << "MUSA tl.reduce with SyncThreadsBarrier only supports full-block "
         "reductions when more than 32 threads participate.";
}

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

  static int GetPreferedVectorizedSize(DataType, Target) { return 1; }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads,
                                        int batch, int workspace_stride,
                                        Target) {
    CheckSyncThreadsScope(reducing_threads, all_threads);
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", tl::SyncThreadsBarrier, "
       << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads,
                                         Target) {
    CheckSyncThreadsScope(reducing_threads, all_threads);
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }
};

} // namespace musa

namespace {

bool MatchMUSAReduceTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAReduce() {
  RegisterReduceImpl(ReduceImpl{
      "musa.Reduce",
      MatchMUSAReduceTarget,
      musa::Reduce::Lower,
  });
  return true;
}

const bool musa_reduce_registered = RegisterMUSAReduce();

} // namespace

} // namespace tl
} // namespace tvm
