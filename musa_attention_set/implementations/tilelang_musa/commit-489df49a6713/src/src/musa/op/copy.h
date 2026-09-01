/*!
 * \file tl/musa/op/copy.h
 * \brief MUSA copy instruction classification helpers.
 */

#ifndef TVM_TL_BACKEND_MUSA_OP_COPY_H_
#define TVM_TL_BACKEND_MUSA_OP_COPY_H_

#include "op/copy.h"
#include "support/check.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace tvm {
namespace tl {
namespace musa {

using namespace tirx;
using namespace ffi;

enum class CopyInst : uint8_t {
  kNormal = 0,
  kAsync = 1,
  kTMELoad = 2,
  kTMEStore = 3,
  kInvalid = 255,
};

const char *CopyInstToString(CopyInst inst);
bool CopyInstIsTME(CopyInst inst);
bool CopyInstIsTMEStore(CopyInst inst);
bool CopyInstIsAsync(CopyInst inst);

struct TMEDesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape;
  Array<PrimExpr> global_stride;
  Array<PrimExpr> smem_box;
  Array<PrimExpr> smem_stride;
  PrimExpr global_addr;
  int interleave;
  int swizzle;
  int l2_promotion;
  int oob_fill;

  Array<PrimExpr> EncodeCallArgs() const {
    Array<PrimExpr> args;
    args.reserve(rank * 4 + 7);

    args.push_back(data_type);
    args.push_back(static_cast<int>(rank));
    args.push_back(global_addr);
    for (auto e : global_shape)
      args.push_back(e);
    for (auto e : global_stride)
      args.push_back(e);
    for (auto e : smem_box)
      args.push_back(e);
    for (auto e : smem_stride)
      args.push_back(e);
    args.push_back(interleave);
    args.push_back(swizzle);
    args.push_back(l2_promotion);
    args.push_back(oob_fill);

    return args;
  }
};

struct CopyAnalysisContext {
  Target target;
  const LayoutMap *layout_map = nullptr;
  arith::Analyzer *analyzer = nullptr;
  bool emit_diagnostics = false;
};

struct CopyInstSelection {
  CopyInst inst = CopyInst::kNormal;
  bool supported = true;
  std::string reason;
};

// Final MUSA lowering decision. Only explicit T.tma_copy/T.async_copy
// annotations select their respective asynchronous instruction families.
CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx);

} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_MUSA_OP_COPY_H_
