/*!
 * \file tl/musa/op/gemm.cc
 * \brief MUSA implementation for tl.gemm instruction dispatch.
 */

#include "op/gemm.h"

#include "musa/op/gemm/mp31_sqmma.h"
#include "musa/target_utils.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace musa {

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (!op.annotations_.Get("is_sqmma")) {
      LOG(FATAL) << "MUSA T.gemm instruction selection is not implemented; "
                    "use T.sqmma_gemm to request MP31 SQMMA explicitly";
    }
    if (TargetIsMP31(target)) {
      return mp31::SQMMA::SelectInst(op, block_size, target);
    }
    LOG(FATAL) << "T.sqmma_gemm is only supported on MP31, target=" << target;
    return {};
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    if (TargetIsMP31(target) && mp31::SQMMA::IsInstruction(gemm_inst)) {
      return mp31::SQMMA::ComputeWarpPartition(policy, M, N, block_size, target,
                                               gemm_inst);
    }
    LOG(FATAL) << "T.sqmma_gemm has no warp partition implementation for "
               << "target=" << target << ", instruction=" << gemm_inst;
    return {0, 0};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    if (mp31::SQMMA::IsInstruction(gemm_inst)) {
      return mp31::SQMMA::ReuseExistingSharedLayout(gemm_inst);
    }
    LOG(FATAL) << "T.sqmma_gemm has no shared-layout policy for instruction "
               << gemm_inst;
    return false;
  }
};

} // namespace musa

namespace {

TVM_REGISTER_OP("tl.tileop.sqmma_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "sqmma_gemm")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_sqmma",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             });

bool MatchMUSAGemmTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAGemm() {
  RegisterGemmImpl(GemmImpl{
      "musa.Gemm",
      MatchMUSAGemmTarget,
      musa::Gemm::SelectInst,
      musa::Gemm::ComputeWarpPartition,
      musa::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool musa_gemm_registered = RegisterMUSAGemm();

} // namespace

} // namespace tl
} // namespace tvm
