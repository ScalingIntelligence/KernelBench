/*!
 * \file tl/musa/target_utils.h
 * \brief MUSA target attribute helpers.
 */

#ifndef TVM_TL_MUSA_TARGET_UTILS_H_
#define TVM_TL_MUSA_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsMUSA(Target target);
bool TargetIsMP31(Target target);
bool TargetMUSACanPropagateKernelErrors(Target target);
bool TargetMUSAHasAsyncCopy(Target target);
int TargetMUSAGetWarpSize(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_MUSA_TARGET_UTILS_H_
