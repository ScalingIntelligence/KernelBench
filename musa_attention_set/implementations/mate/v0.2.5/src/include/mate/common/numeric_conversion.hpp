#pragma once

#if !defined(__MUSACC_RTC__)
#include <cfenv>
#endif

#include "mutlass/array.h"
#include "mutlass/half.h"
#include "mutlass/mutlass.h"
#include "mutlass/numeric_conversion.h"
#include "mutlass/numeric_types.h"
#include "mutlass/transform/thread/unary_op.h"

namespace mutlass {
/// Partial specialization for Array<mutlass::float_e4m3_t, 4> <= Array<float, 4>, round to nearest
template <>
struct NumericArrayConverter<mutlass::float_e4m3_t, float, 4, FloatRoundStyle::round_to_nearest> {
  using result_type                        = Array<mutlass::float_e4m3_t, 4>;
  using source_type                        = Array<float, 4>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    reinterpret_cast<_char_v4 &>(result) = __musa_f2e4m3_rn_bst4(*reinterpret_cast<_float_v4 const *>(&source));

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<mutlass::float_e4m3_t, 16> <= Array<float, 16>, round to nearest
template <>
struct NumericArrayConverter<mutlass::float_e4m3_t, float, 16, FloatRoundStyle::round_to_nearest> {
  using result_type                        = Array<mutlass::float_e4m3_t, 16>;
  using source_type                        = Array<float, 16>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  MUTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    NumericArrayConverter<mutlass::float_e4m3_t, float, 4, round_style> convert_vector_;

    result_type result;

    Array<mutlass::float_e4m3_t, 4> *result_ptr = reinterpret_cast<Array<mutlass::float_e4m3_t, 4> *>(&result);
    Array<float, 4> const           *source_ptr = reinterpret_cast<Array<float, 4> const *>(&source);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  MUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

}  // namespace mutlass
