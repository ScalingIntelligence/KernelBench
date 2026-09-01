#pragma once

#include "intrin.h"

#include <math.h>

#if defined(TL_MUSA_ENABLE_FP16)
#include <musa_fp16.h>
#endif

#if defined(TL_MUSA_ENABLE_BF16)
#include <musa_bf16.h>
#endif

#if defined(TL_MUSA_ENABLE_FP16)
TL_DEVICE half htan(half x) {
  return __float2half_rn(tanf(__half2float(x)));
}
#endif

#if defined(TL_MUSA_ENABLE_BF16)
TL_DEVICE mt_bfloat16 htan(mt_bfloat16 x) {
  return __float2bfloat16(tanf(__bfloat162float(x)));
}
#endif

namespace tl {

TL_DEVICE float fast_rcp(float x) { return __frcp_rn(x); }

}  // namespace tl
