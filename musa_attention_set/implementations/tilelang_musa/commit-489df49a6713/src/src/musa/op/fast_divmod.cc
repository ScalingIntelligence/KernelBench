/*!
 * \file tl/musa/op/fast_divmod.cc
 * \brief Register explicit MUSA fast integer division operations.
 */

#include "musa/op/fast_divmod.h"

#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
namespace musa {

const Op &fast_div() {
  static const Op &op = Op::Get("tl.musa.fast_div");
  return op;
}

const Op &fast_mod() {
  static const Op &op = Op::Get("tl.musa.fast_mod");
  return op;
}

const Op &fast_div_precomputed() {
  static const Op &op = Op::Get("tl.musa.fast_div_precomputed");
  return op;
}

const Op &fast_mod_precomputed() {
  static const Op &op = Op::Get("tl.musa.fast_mod_precomputed");
  return op;
}

const Op &fast_div_device() {
  static const Op &op = Op::Get("tl.musa.fast_div_device");
  return op;
}

const Op &fast_mod_device() {
  static const Op &op = Op::Get("tl.musa.fast_mod_device");
  return op;
}

} // namespace musa
} // namespace tl
} // namespace tvm

using namespace tvm;
using namespace tvm::tirx;

// Keep the public markers opaque until the MUSA pass runs.  This preserves
// adjacent Bind nodes produced by `q, r = T.fast_divmod(...)`, allowing the
// lowering to compute the quotient once and derive the remainder from it.
TVM_REGISTER_OP("tl.musa.fast_div")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl::fast_div")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "fast_div");

TVM_REGISTER_OP("tl.musa.fast_mod")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl::fast_mod")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "fast_mod");

// The quotient remains opaque so later simplification retains a shared Bind
// for fast_divmod.  The remainder marker is pure once its inputs are prepared.
TVM_REGISTER_OP("tl.musa.fast_div_precomputed")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl::fast_div");

TVM_REGISTER_OP("tl.musa.fast_mod_precomputed")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl::fast_mod");

TVM_REGISTER_OP("tl.musa.fast_div_device")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl::fast_div");

TVM_REGISTER_OP("tl.musa.fast_mod_device")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl::fast_mod");
