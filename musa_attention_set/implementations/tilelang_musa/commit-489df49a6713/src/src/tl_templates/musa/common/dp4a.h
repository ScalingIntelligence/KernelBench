#pragma once

#include <stdint.h>

namespace tl {

__device__ inline void DP4A(const int8_t* a, const int8_t* b, int32_t* c) {
  const char4 a_vec = *reinterpret_cast<const char4*>(a);
  const char4 b_vec = *reinterpret_cast<const char4*>(b);
  *c = __dp4a(a_vec, b_vec, *c);
}

}  // namespace tl
