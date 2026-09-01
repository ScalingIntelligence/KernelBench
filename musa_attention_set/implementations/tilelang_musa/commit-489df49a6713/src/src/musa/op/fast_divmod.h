/*!
 * \file tl/musa/op/fast_divmod.h
 * \brief MUSA fast integer division intrinsic markers.
 */

#ifndef TVM_TL_MUSA_OP_FAST_DIVMOD_H_
#define TVM_TL_MUSA_OP_FAST_DIVMOD_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace tl {
namespace musa {

TVM_DLL const Op &fast_div();
TVM_DLL const Op &fast_mod();
TVM_DLL const Op &fast_div_precomputed();
TVM_DLL const Op &fast_mod_precomputed();
TVM_DLL const Op &fast_div_device();
TVM_DLL const Op &fast_mod_device();

} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_MUSA_OP_FAST_DIVMOD_H_
