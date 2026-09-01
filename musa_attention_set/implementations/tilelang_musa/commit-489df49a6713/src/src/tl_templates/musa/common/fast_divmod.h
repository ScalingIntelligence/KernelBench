/*!
 * \file tl_templates/musa/common/fast_divmod.h
 * \brief Common MUSA fast integer division helpers.
 */

#pragma once

#include <assert.h>
#include <stdint.h>

#ifndef TL_DEVICE
#define TL_DEVICE __forceinline__ __device__
#endif

namespace tl {

struct FastDivmodParams {
  uint32_t multiplier;
  uint32_t shift_right;
};

// Construct magic-number division parameters on the device when the divisor
// depends on device-only state.  __clz and __umulhi are public MUSA C device
// intrinsics supplied by MTCC; no mutlass/mute types are required.
TL_DEVICE FastDivmodParams make_fast_divmod_params(int divisor) {
  assert(divisor > 0);
  if (divisor == 1) {
    return {0u, 0u};
  }

  uint32_t denominator = static_cast<uint32_t>(divisor);
  uint32_t ceil_log2 = 32u - static_cast<uint32_t>(__clz(denominator - 1u));
  uint32_t exponent = 31u + ceil_log2;
  uint64_t numerator = uint64_t{1} << exponent;
  uint32_t multiplier = static_cast<uint32_t>(
      (numerator + denominator - 1u) / denominator);
  return {multiplier, exponent - 32u};
}

TL_DEVICE int fast_div(int dividend, int divisor, uint32_t multiplier,
                       uint32_t shift_right) {
  assert(dividend >= 0);
  assert(divisor > 0);
  if (divisor == 1) {
    return dividend;
  }
  return static_cast<int>(
      __umulhi(static_cast<uint32_t>(dividend), multiplier) >> shift_right);
}

TL_DEVICE int fast_mod(int dividend, int divisor, uint32_t multiplier,
                       uint32_t shift_right) {
  int quotient = fast_div(dividend, divisor, multiplier, shift_right);
  return dividend - quotient * divisor;
}

TL_DEVICE int fast_div(int dividend, int divisor) {
  FastDivmodParams params = make_fast_divmod_params(divisor);
  return fast_div(dividend, divisor, params.multiplier, params.shift_right);
}

TL_DEVICE int fast_mod(int dividend, int divisor) {
  FastDivmodParams params = make_fast_divmod_params(divisor);
  return fast_mod(dividend, divisor, params.multiplier, params.shift_right);
}

} // namespace tl
