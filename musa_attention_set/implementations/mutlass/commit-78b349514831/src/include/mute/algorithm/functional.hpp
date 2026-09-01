/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <mute/config.hpp>          // MUTE_HOST_DEVICE
#include <mute/numeric/math.hpp>    // mute::max, mute::min
#include <mute/numeric/complex.hpp> // mute::conj

/** C++14 <functional> extensions */

namespace mute {

/**************/
/** Identity **/
/**************/

struct identity {
  template <class T>
  MUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&& arg) const {
    return static_cast<T&&>(arg);
  }
};

template <class R>
struct constant_fn {
  template <class... T>
  MUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&&...) const {
    return r_;
  }
  R r_;
};

/***********/
/** Unary **/
/***********/

#define MUTE_LEFT_UNARY_OP(NAME,OP)                                  \
  struct NAME {                                                      \
    template <class T>                                               \
    MUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& arg) const {                       \
      return OP static_cast<T&&>(arg);                                \
    }                                                                \
  }
#define MUTE_RIGHT_UNARY_OP(NAME,OP)                                 \
  struct NAME {                                                      \
    template <class T>                                               \
    MUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& arg) const {                       \
      return static_cast<T&&>(arg) OP ;                               \
    }                                                                \
  }
#define MUTE_NAMED_UNARY_OP(NAME,OP)                                 \
  struct NAME {                                                      \
    template <class T>                                               \
    MUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& arg) const {                       \
      return OP (static_cast<T&&>(arg));                              \
    }                                                                \
  }

MUTE_LEFT_UNARY_OP(unary_plus,       +);
MUTE_LEFT_UNARY_OP(negate,           -);
MUTE_LEFT_UNARY_OP(bit_not,          ~);
MUTE_LEFT_UNARY_OP(logical_not,      !);
MUTE_LEFT_UNARY_OP(dereference,      *);
MUTE_LEFT_UNARY_OP(address_of,       &);
MUTE_LEFT_UNARY_OP(pre_increment,   ++);
MUTE_LEFT_UNARY_OP(pre_decrement,   --);

MUTE_RIGHT_UNARY_OP(post_increment, ++);
MUTE_RIGHT_UNARY_OP(post_decrement, --);

MUTE_NAMED_UNARY_OP(abs_fn,           abs);
MUTE_NAMED_UNARY_OP(conjugate, mute::conj);

#undef MUTE_LEFT_UNARY_OP
#undef MUTE_RIGHT_UNARY_OP
#undef MUTE_NAMED_UNARY_OP

template <int Shift_>
struct shift_right_const {
  static constexpr int Shift = Shift_;

  template <class T>
  MUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&& arg) const {
    return static_cast<T&&>(arg) >> Shift;
  }
};

template <int Shift_>
struct shift_left_const {
  static constexpr int Shift = Shift_;

  template <class T>
  MUTE_HOST_DEVICE constexpr
  decltype(auto) operator()(T&& arg) const {
    return static_cast<T&&>(arg) << Shift;
  }
};

/************/
/** Binary **/
/************/

#define MUTE_BINARY_OP(NAME,OP)                                      \
  struct NAME {                                                      \
    template <class T, class U>                                      \
    MUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& lhs, U&& rhs) const {              \
      return static_cast<T&&>(lhs) OP static_cast<U&&>(rhs);           \
    }                                                                \
  }
#define MUTE_NAMED_BINARY_OP(NAME,OP)                                \
  struct NAME {                                                      \
    template <class T, class U>                                      \
    MUTE_HOST_DEVICE constexpr                                       \
    decltype(auto) operator()(T&& lhs, U&& rhs) const {              \
      return OP (static_cast<T&&>(lhs), static_cast<U&&>(rhs));        \
    }                                                                \
  }


MUTE_BINARY_OP(plus,                 +);
MUTE_BINARY_OP(minus,                -);
MUTE_BINARY_OP(multiplies,           *);
MUTE_BINARY_OP(divides,              /);
MUTE_BINARY_OP(modulus,              %);

MUTE_BINARY_OP(plus_assign,         +=);
MUTE_BINARY_OP(minus_assign,        -=);
MUTE_BINARY_OP(multiplies_assign,   *=);
MUTE_BINARY_OP(divides_assign,      /=);
MUTE_BINARY_OP(modulus_assign,      %=);

MUTE_BINARY_OP(bit_and,              &);
MUTE_BINARY_OP(bit_or,               |);
MUTE_BINARY_OP(bit_xor,              ^);
MUTE_BINARY_OP(left_shift,          <<);
MUTE_BINARY_OP(right_shift,         >>);

MUTE_BINARY_OP(bit_and_assign,      &=);
MUTE_BINARY_OP(bit_or_assign,       |=);
MUTE_BINARY_OP(bit_xor_assign,      ^=);
MUTE_BINARY_OP(left_shift_assign,  <<=);
MUTE_BINARY_OP(right_shift_assign, >>=);

MUTE_BINARY_OP(logical_and,         &&);
MUTE_BINARY_OP(logical_or,          ||);

MUTE_BINARY_OP(equal_to,            ==);
MUTE_BINARY_OP(not_equal_to,        !=);
MUTE_BINARY_OP(greater,              >);
MUTE_BINARY_OP(less,                 <);
MUTE_BINARY_OP(greater_equal,       >=);
MUTE_BINARY_OP(less_equal,          <=);

MUTE_NAMED_BINARY_OP(max_fn, mute::max);
MUTE_NAMED_BINARY_OP(min_fn, mute::min);

#undef MUTE_BINARY_OP
#undef MUTE_NAMED_BINARY_OP

/**********/
/** Fold **/
/**********/

#define MUTE_FOLD_OP(NAME,OP)                                        \
  struct NAME##_unary_rfold {                                        \
    template <class... T>                                            \
    MUTE_HOST_DEVICE constexpr                                       \
    auto operator()(T&&... t) const {                                \
      return (t OP ...);                                             \
    }                                                                \
  };                                                                 \
  struct NAME##_unary_lfold {                                        \
    template <class... T>                                            \
    MUTE_HOST_DEVICE constexpr                                       \
    auto operator()(T&&... t) const {                                \
      return (... OP t);                                             \
    }                                                                \
  };                                                                 \
  struct NAME##_binary_rfold {                                       \
    template <class U, class... T>                                   \
    MUTE_HOST_DEVICE constexpr                                       \
    auto operator()(U&& u, T&&... t) const {                         \
      return (t OP ... OP u);                                        \
    }                                                                \
  };                                                                 \
  struct NAME##_binary_lfold {                                       \
    template <class U, class... T>                                   \
    MUTE_HOST_DEVICE constexpr                                       \
    auto operator()(U&& u, T&&... t) const {                         \
      return (u OP ... OP t);                                        \
    }                                                                \
  }

MUTE_FOLD_OP(plus,                 +);
MUTE_FOLD_OP(minus,                -);
MUTE_FOLD_OP(multiplies,           *);
MUTE_FOLD_OP(divides,              /);
MUTE_FOLD_OP(modulus,              %);

MUTE_FOLD_OP(plus_assign,         +=);
MUTE_FOLD_OP(minus_assign,        -=);
MUTE_FOLD_OP(multiplies_assign,   *=);
MUTE_FOLD_OP(divides_assign,      /=);
MUTE_FOLD_OP(modulus_assign,      %=);

MUTE_FOLD_OP(bit_and,              &);
MUTE_FOLD_OP(bit_or,               |);
MUTE_FOLD_OP(bit_xor,              ^);
MUTE_FOLD_OP(left_shift,          <<);
MUTE_FOLD_OP(right_shift,         >>);

MUTE_FOLD_OP(bit_and_assign,      &=);
MUTE_FOLD_OP(bit_or_assign,       |=);
MUTE_FOLD_OP(bit_xor_assign,      ^=);
MUTE_FOLD_OP(left_shift_assign,  <<=);
MUTE_FOLD_OP(right_shift_assign, >>=);

MUTE_FOLD_OP(logical_and,         &&);
MUTE_FOLD_OP(logical_or,          ||);

MUTE_FOLD_OP(equal_to,            ==);
MUTE_FOLD_OP(not_equal_to,        !=);
MUTE_FOLD_OP(greater,              >);
MUTE_FOLD_OP(less,                 <);
MUTE_FOLD_OP(greater_equal,       >=);
MUTE_FOLD_OP(less_equal,          <=);

#undef MUTE_FOLD_OP

/**********/
/** Meta **/
/**********/

template <class Fn, class Arg>
struct bound_fn {

  template <class T>
  MUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(T&& arg) {
    return fn_(arg_, static_cast<T&&>(arg));
  }

  Fn fn_;
  Arg arg_;
};

template <class Fn, class Arg>
MUTE_HOST_DEVICE constexpr
auto
bind(Fn const& fn, Arg const& arg) {
  return bound_fn<Fn,Arg>{fn, arg};
}

} // end namespace mute
