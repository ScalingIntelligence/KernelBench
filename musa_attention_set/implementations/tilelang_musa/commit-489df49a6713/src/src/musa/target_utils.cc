/*!
 * \file tl/musa/target_utils.cc
 * \brief MUSA target attribute helpers.
 */

#include "musa/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace tvm {
namespace tl {
namespace {

int GetMusaArchInt(Target target) {
  auto s = target->GetAttr<ffi::String>("arch");
  if (!s.has_value()) {
    return 0;
  }
  const std::string arch_str = s.value();
  if (arch_str.rfind("mp_", 0) != 0) {
    return 0;
  }
  const std::string arch_num = arch_str.substr(3);
  if (arch_num.empty() ||
      !std::all_of(arch_num.begin(), arch_num.end(), [](unsigned char ch) {
        return std::isdigit(ch);
      })) {
    return 0;
  }
  return std::stoi(arch_num);
}

} // namespace

bool TargetIsMUSA(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "musa";
}

bool TargetIsMP31(Target target) {
  if (!TargetIsMUSA(target)) {
    return false;
  }
  return GetMusaArchInt(target) == 31;
}

bool TargetMUSACanPropagateKernelErrors(Target) { return false; }

bool TargetMUSAHasAsyncCopy(Target target) {
  if (!TargetIsMUSA(target)) {
    return false;
  }
  return GetMusaArchInt(target) >= 21;
}

int TargetMUSAGetWarpSize(Target target) {
  if (!TargetIsMUSA(target)) {
    return 32;
  }
  int arch = GetMusaArchInt(target);
  return arch > 0 && arch <= 22 ? 128 : 32;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsMUSA",
           [](Target target) { return TargetIsMUSA(target); })
      .def("tl.TargetIsMP31",
           [](Target target) { return TargetIsMP31(target); })
      .def("tl.TargetMUSAHasAsyncCopy",
           [](Target target) { return TargetMUSAHasAsyncCopy(target); })
      .def("tl.TargetMUSAGetWarpSize",
           [](Target target) { return TargetMUSAGetWarpSize(target); });
}

} // namespace tl
} // namespace tvm
