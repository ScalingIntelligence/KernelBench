#pragma once

namespace mutlass::fmha {

template <auto Tag, typename Default, typename... Options>
struct find_option;

template <auto Tag, typename Default>
struct find_option<Tag, Default> {
  using option_value = Default;
};

template <auto Tag, typename Default, typename Option, typename... Options>
struct find_option<Tag, Default, Option, Options...> :
  std::conditional_t<
    Option::tag == Tag,
    Option,
    find_option<Tag, Default, Options...>
  >
{};

template <auto Tag, typename Default, typename... Options>
using find_option_t = typename find_option<Tag, Default, Options...>::option_value;

template <auto Tag, class Value>
struct Option {
  static constexpr auto tag = Tag;
  using option_value = Value;
};


enum class Tag {
  NumMmaWarpSquads,
  KStage,
  VStage,
  TmeLoadQ,
  Varlen,
};

} // namespace mutlass::fmha
