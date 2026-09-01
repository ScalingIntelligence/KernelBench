#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

namespace mate::flat {

template <auto TagValue, typename Default, typename... Options>
struct find_option;

template <auto TagValue, typename Default>
struct find_option<TagValue, Default> {
  using option_value = Default;
};

template <auto TagValue, typename Default, typename Option, typename... Options>
struct find_option<TagValue, Default, Option, Options...>
    : std::conditional_t<Option::tag == TagValue, Option, find_option<TagValue, Default, Options...>> {};

template <auto TagValue, typename Default, typename... Options>
using find_option_t = typename find_option<TagValue, Default, Options...>::option_value;

template <auto TagValue, class Value>
struct Option {
  static constexpr auto tag = TagValue;
  using option_value        = Value;
};

enum class Tag {
  HasStateIn,
  HasStateOut,
  HasGateParams,
  IsVarlen,
  NormalizeQK,
};

}  // namespace mate::flat
