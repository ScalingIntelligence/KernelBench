/* Host-side MP31 TME descriptor creation. */

#include "musa/runtime.h"

#include <musa.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace ffi;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tvm_musa_tensordesc_create_tiled, [](PackedArgs args, Any *ret) {
        ICHECK_GE(args.size(), 8);
        auto *desc = static_cast<MUtensorDescriptor *>(args[0].cast<void *>());
        int dtype = static_cast<int>(args[1].cast<int64_t>());
        int rank = static_cast<int>(args[2].cast<int64_t>());
        ICHECK_GE(rank, 1);
        ICHECK_LE(rank, 5);
        ICHECK_EQ(args.size(), static_cast<size_t>(rank * 4 + 8));
        void *global_addr = args[3].cast<void *>();

        muuint64_t global_dim[5] = {};
        muuint64_t global_stride[5] = {};
        int index = 4;
        for (int i = 0; i < rank; ++i) {
          global_dim[i] = args[index++].cast<muuint64_t>();
        }
        for (int i = 0; i < rank; ++i) {
          global_stride[i] = args[index++].cast<muuint64_t>();
        }
        auto oob_constant_fill = static_cast<muuint64_t>(
            args[args.size() - 1].cast<int64_t>());
        // Skip shared-memory box/stride and the descriptor policy fields. The
        // first MP31 slice uses no interleave and no swizzle. Forward the final
        // descriptor argument as the hardware OOB fill constant.
        MUresult result = muTensorDescriptorEncode(
            desc, static_cast<MUtensorDescriptorDataType>(dtype), rank,
            global_addr, global_dim, global_stride + 1,
            MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, oob_constant_fill);
        ICHECK_EQ(result, MUSA_SUCCESS)
            << "Failed to initialize MP31 TME descriptor, error=" << result;
        *ret = static_cast<int>(result);
      });
}

} // namespace tl
} // namespace tvm
