/*!
 * \file tl/musa/op/gemm/mp31_sqmma.h
 * \brief MP31 SQMMA implementation hooks for tl.gemm.
 */

#ifndef TVM_TL_MUSA_OP_GEMM_MP31_SQMMA_H_
#define TVM_TL_MUSA_OP_GEMM_MP31_SQMMA_H_

#include "op/gemm.h"

namespace tvm {
namespace tl {
namespace musa {
namespace mp31 {

struct SQMMA {
  static bool IsInstruction(const String &gemm_inst);

  static String SelectInst(const GemmNode &op, int block_size, Target target);

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst);

  static Array<Integer> GetInstShape(const GemmNode &op, int block_size,
                                     int m_warp, int n_warp,
                                     const Target &target);

  static bool ReuseExistingSharedLayout(String gemm_inst);
};

} // namespace mp31
} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_MUSA_OP_GEMM_MP31_SQMMA_H_
