/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <muComplex.h>

#include <musa_fp16.h>

#if defined(__MUSACC_RTC__)
#include <musa/std/cstdint>
#else
#include <cstdint>
#endif

#include "mutlass/mutlass.h"
#include "mutlass/functional.h"
#include "mutlass/real.h"

#include "mutlass/numeric_types.h"

#include "mutlass/fast_math.h"

#if !defined(__MUSACC_RTC__)
#include <iosfwd>
#endif

namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Enumeraed type describing a transformation on a complex value.
enum class ComplexTransform {
  kNone,
  kConjugate
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines ComplexTransform inversions
template <ComplexTransform kTransform>
struct InvertComplexTransform;

/// Invert ComplexTransform from kNone to kConjugate
template <>
struct InvertComplexTransform<ComplexTransform::kNone> {
  static ComplexTransform const transform = ComplexTransform::kConjugate;
};

/// Invert ComplexTransform from kConjugate to kNone
template <>
struct InvertComplexTransform<ComplexTransform::kConjugate> {
  static ComplexTransform const transform = ComplexTransform::kNone;
};
/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for MUSA complex types
//

#if !defined(__MUSACC_RTC__)
/// Returns the real part of the complex number
MUTLASS_HOST_DEVICE
float const &real(muFloatComplex const &z) { return z.x; }

/// Returns the real part of the complex number
MUTLASS_HOST_DEVICE
float &real(muFloatComplex &z) { return z.x; }

/// Returns the real part of the complex number
MUTLASS_HOST_DEVICE
double const &real(muDoubleComplex const &z) { return z.x; }

/// Returns the real part of the complex number
MUTLASS_HOST_DEVICE
double &real(muDoubleComplex &z) { return z.x; }

/// Returns the imaginary part of the complex number
MUTLASS_HOST_DEVICE
float const &imag(muFloatComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
MUTLASS_HOST_DEVICE
float &imag(muFloatComplex &z) { return z.y; }

/// Returns the imaginary part of the complex number
MUTLASS_HOST_DEVICE
double const &imag(muDoubleComplex const &z) { return z.y; }

/// Returns the imaginary part of the complex number
MUTLASS_HOST_DEVICE
double &imag(muDoubleComplex &z) { return z.y; }
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class for representing and manipulating complex numbers with conversions from built-in MUSA
/// complex types.

template <typename T>
class complex
{
 public:
  /// Type alias for scalar type
  using value_type = T;

 private:
  //
  // Data members
  //

  /// Real part
  T _real;

  /// Imaginary part
  T _imag;

 public:

//
// Methods
//

  /// Default constructor
  complex() = default;

  /// Constructor
  MUTLASS_HOST_DEVICE
  complex(T r) : _real(r), _imag(T(0)) {}

  /// Constructor
  MUTLASS_HOST_DEVICE
  complex(T r, T i) : _real(r), _imag(i) {}

  /// Constructor
  template<typename A>
  MUTLASS_HOST_DEVICE
  complex(complex<A> const &z) : _real(static_cast<T>(z.real())), _imag(static_cast<T>(z.imag())) {}


  #if !defined(__MUSACC_RTC__)
  /// Conversion from muFloatComplex
  MUTLASS_HOST_DEVICE
  complex(muFloatComplex const &z) : _real(static_cast<T>(muCrealf(z))), _imag(static_cast<T>(muCimagf(z))) {}

  /// Conversion from muDoubleComplex
  MUTLASS_HOST_DEVICE
  complex(muDoubleComplex const &z) : _real(static_cast<T>(muCreal(z))), _imag(static_cast<T>(muCimag(z))) {}
  #endif

  /// Equality operator
  MUTLASS_HOST_DEVICE bool operator==(complex<T> const &rhs) const {
    return this->real() == rhs.real() && this->imag() == rhs.imag();
  }

  /// Inequality operator
  MUTLASS_HOST_DEVICE bool operator!=(complex<T> const &rhs) const {
    return !(*this == rhs);
  }

  /// Addition
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> operator+(complex<A> const &rhs) const {
    return complex<T>(this->real() + rhs.real(), this->imag() + rhs.imag());
  }

  /// Reduction into memory address.  Components may update out of order.
  template <typename OtherT>
  MUTLASS_DEVICE void red(complex<OtherT> *ptr) const {
    static_assert(platform::is_same<T, OtherT>::value, "Component type must match");
    mutlass::atomic_add<T> reduce;
    reduce(&ptr->_real, _real);
    reduce(&ptr->_imag, _imag);
  }

  /// Reduction into memory address.  Components may update out of order.  (Half specialization)
  MUTLASS_DEVICE void red(complex<half_t> *ptr) const {
    static_assert(platform::is_same<T, half_t>::value, "Component type must match");
    half2 *h2_ptr = reinterpret_cast<half2*>(ptr);
    half2 h2_data = reinterpret_cast<half2&>(*this);
    mutlass::atomic_add<half2> reduce;
    reduce(h2_ptr, h2_data);
  }

  /// Subtraction
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> operator-(complex<A> const &rhs) const {
    return complex<T>(this->real() - rhs.real(), this->imag() - rhs.imag());
  }

  /// Multiplication
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> operator*(complex<A> const &rhs) const {
    return complex<T>(this->real() * rhs.real() - this->imag() * rhs.imag(),
                      this->real() * rhs.imag() + this->imag() * rhs.real());
  }

  /// Scalar Multiplication
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> operator*(A const &s) const {
    return complex<T>(this->real() * s, this->imag() * s);
  }

  /// Division
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> operator/(complex<A> const &rhs) const {
    T d = T(rhs.real() * rhs.real() + rhs.imag() * rhs.imag());

    return complex<T>(
      (real() * rhs.real() + imag() * rhs.imag()) / d,
      (imag() * rhs.real() - real() * rhs.imag()) / d
    );
  }

  /// Scalar Division
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> operator/(A const &s) const {
    return complex<T>(this->real() / s, this->imag() / s);
  }

  /// Addition
    template <typename A>
  MUTLASS_HOST_DEVICE complex<T> &operator+=(complex<A> const &rhs) {
      *this = *this + rhs;
      return *this;
  }

  /// Subtraction
  template <typename A>
  MUTLASS_HOST_DEVICE complex<T> &operator-=(complex<A> const &rhs) {
      *this = *this - rhs;
      return *this;
  }

  /// Multiplication
  template <typename A>
  MUTLASS_HOST_DEVICE complex<T> &operator*=(complex<A> const &rhs) {
      *this = *this * rhs;
      return *this;
  }

  /// Scalar multiplication
  template <typename A>
  MUTLASS_HOST_DEVICE complex<T> &operator*=(A s) {
      *this = *this * s;
      return *this;
  }

  /// Division
  template <typename A>
  MUTLASS_HOST_DEVICE complex<T> &operator/=(complex<A> const &rhs) {
      *this = *this / rhs;
      return *this;
  }

  /// Accesses the real part of the complex number
  MUTLASS_HOST_DEVICE
  T const &real() const { return _real; }

  /// Accesses the real part of the complex number
  MUTLASS_HOST_DEVICE
  T &real() { return _real; }

  /// Accesses the imaginary part of the complex number
  MUTLASS_HOST_DEVICE
  T const &imag() const { return _imag; }

  /// Accesses the imaginary part of the complex number
  MUTLASS_HOST_DEVICE
  T &imag() { return _imag; }

  /// Set the real part of the complex number
  MUTLASS_HOST_DEVICE
  void real(T real) { _real = real; }

  /// Set the imaginary part of the complex number
  MUTLASS_HOST_DEVICE
  void imag(T imag) { _imag = imag; }

  #if !defined(__MUSACC_RTC__)
  /// Converts to muFloatComplex
  MUTLASS_HOST_DEVICE
  explicit operator muFloatComplex() const { return make_muFloatComplex(float(real()), float(imag())); }

  /// Converts to muDoubleComplex
  MUTLASS_HOST_DEVICE
  explicit operator muDoubleComplex() const { return make_muDoubleComplex(real(), imag()); }
  #endif
};

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// Accessors for complex template
//

/// Returns the real part of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T const &real(complex<T> const &z) {
  return z.real();
}

/// Returns the real part of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T &real(complex<T> &z) {
  return z.real();
}

/// Returns the imaginary part of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T const &imag(complex<T> const &z) {
  return z.imag();
}

/// Returns the imaginary part of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T &imag(complex<T> &z) {
  return z.imag();
}

/// Returns the real part of the real number
template <typename T>
MUTLASS_HOST_DEVICE T const &real(T const &r) {
  return r;
}

/// Returns the real part of the real number
template <typename T>
MUTLASS_HOST_DEVICE T &real(T &r) {
  return r;
}

/// Returns the imaginary part of the real number
template <typename T>
MUTLASS_HOST_DEVICE T const &imag(T const &r) {
  return T();
}

/// Returns the imaginary part of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T &imag(T &r) {
  return T();
}

//
// Output operators
//

#if !defined(__MUSACC_RTC__)
template <typename T>
std::ostream &operator<<(std::ostream &out, complex<T> const &z) {
  T _r = real(z);
  T _i = imag(z);

  if (bool(_i)) {
    return out << _r << "+i" << _i;
  }
  return out << _r;
}
#endif

//
// Non-member operators defined for complex types
//


//
// Non-member functions defined for complex numbers
//

/// Returns the magnitude of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T abs(complex<T> const &z) {
  return sqrt(norm(z));
}

/// Returns the magnitude of the complex number
template <typename T>
MUTLASS_HOST_DEVICE T arg(complex<T> const &z) {
  return atan2(imag(z), real(z));
}

/// Returns the squared magnitude of a real number
template <typename T>
MUTLASS_HOST_DEVICE T norm(T const &z) {
    return z * z;
}

/// Returns the squared magnitude of a real number
template <>
MUTLASS_HOST_DEVICE int8_t norm(int8_t const &z) {
    return static_cast<int8_t>(z * z);
}

/// Returns the squared magnitude of a complex number
template <typename T>
MUTLASS_HOST_DEVICE double norm(complex<T> const &z) {
  return real(z) * real(z) + imag(z) * imag(z);
}

/// Norm-accumulate calculation
template <typename T, typename R>
MUTLASS_HOST_DEVICE R norm_accumulate(T const &x, R const & accumulator) {
  return accumulator + static_cast<R>(x) * static_cast<R>(x);
}

/// Norm accumulate specialized for complex types
template <typename T, typename R>
MUTLASS_HOST_DEVICE R norm_accumulate(complex<T> const &z, R const &accumulator) {
  return accumulator + static_cast<R>(real(z)) * static_cast<R>(real(z)) +
    static_cast<R>(imag(z)) * static_cast<R>(imag(z));
}

MUTLASS_HOST_DEVICE float conj(float const &z) {
  return z;
}

MUTLASS_HOST_DEVICE double conj(double const &z) {
  return z;
}

MUTLASS_HOST_DEVICE half_t conj(half_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE int32_t conj(int32_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE uint32_t conj(uint32_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE int64_t conj(int64_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE uint64_t conj(uint64_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE int4b_t conj(int4b_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE uint4b_t conj(uint4b_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE bfloat16_t conj(bfloat16_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE uint1b_t conj(uint1b_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE tfloat32_t conj(tfloat32_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE float_e4m3_t conj(float_e4m3_t const& z) {
  return z;
}

MUTLASS_HOST_DEVICE float_e5m2_t conj(float_e5m2_t const& z) {
  return z;
}


/// Returns the complex conjugate
template <typename T>
MUTLASS_HOST_DEVICE complex<T> conj(complex<T> const &z) {
  return complex<T>(real(z), -imag(z));
}

/// Projects the complex number z onto the Riemann sphere
template <typename T>
MUTLASS_HOST_DEVICE complex<T> proj(complex<T> const &z) {
  T d = real(z) * real(z) + imag(z) * imag(z) + T(1);
  return complex<T>((T(2) * real(z)) / d, (T(2) * imag(z)) / d);
}

/// Returns a complex number with magnitude r and phase theta
template <typename T>
MUTLASS_HOST_DEVICE complex<T> polar(T const &r, T const &theta = T()) {
  return complex<T>(r * cos(theta), r * sin(theta));
}

/// Computes the complex exponential of z.
template <typename T>
MUTLASS_HOST_DEVICE complex<T> exp(complex<T> const &z) {
  return complex<T>(fast_exp(real(z)) * fast_cos(imag(z)), fast_exp(real(z)) * fast_sin(imag(z)));
}

/// Computes the log of z
template <typename T>
MUTLASS_HOST_DEVICE complex<T> log(complex<T> const &z) {
  return complex<T>(log(abs(z)), arg(z));
}

/// Computes the log base 10 of z
template <typename T>
MUTLASS_HOST_DEVICE complex<T> log10(complex<T> const &z) {
  return log(z) / T(log(T(10)));
}

/// Computes the square root of complex number z
template <typename T>
MUTLASS_HOST_DEVICE complex<T> sqrt(complex<T> const &z) {
  return sqrt(T(2)) / T(2) *
         complex<T>(sqrt(sqrt(norm(z)) + real(z)),
                    (imag(z) < 0 ? T(-1) : T(1)) * sqrt(sqrt(norm(z)) - real(z)));
}

/// Computes the cosine of complex z.
template <typename T>
MUTLASS_HOST_DEVICE complex<T> cos(complex<T> const &z) {
  return (exp(z) + exp(-z)) / T(2);
}

/// Computes the sin of complex z.
template <typename T>
MUTLASS_HOST_DEVICE complex<T> sin(complex<T> const &z) {
  return (exp(-z) - exp(z)) * complex<T>(T(0), T(1) / T(2));
}

/// Comparison
template <typename T>
MUTLASS_HOST_DEVICE bool operator<(complex<T> const &lhs, complex<T> const &rhs) {
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex-valued type.
template <typename T>
struct RealType< complex<T> >
{
  using Type = T;

  /// Number of elements
  static int const kExtent = 2;

  MUTLASS_HOST_DEVICE
  static complex<T> from_real(double x) {
    return complex<T>(static_cast<T>(x));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
MUTLASS_HOST_DEVICE
mutlass::complex<half_t> from_real<mutlass::complex<half_t> >(double r) {
  return mutlass::complex<half_t>(half_t(r));
}

template <>
MUTLASS_HOST_DEVICE
mutlass::complex<float> from_real<mutlass::complex<float> >(double r) {
  return mutlass::complex<float>(float(r));
}

template <>
MUTLASS_HOST_DEVICE
mutlass::complex<double> from_real<mutlass::complex<double> >(double r) {
  return mutlass::complex<double>(r);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct is_complex {
  static bool const value = false;
};

template <typename T>
struct is_complex<complex<T>> {
  static bool const value = true;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// functional.h numeric specializations
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Squares with optional conversion
template <typename T, typename Output>
struct magnitude_squared<complex<T>, Output> {
  MUTLASS_HOST_DEVICE
  Output operator()(complex<T> lhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real());
    Output y_i = Output(lhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<complex<T>, complex<T>, complex<T>> {
  MUTLASS_HOST_DEVICE
  complex<T> operator()(
    complex<T> const &a,
    complex<T> const &b,
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b.real();
    real += -a.imag() * b.imag();
    imag += a.real() * b.imag();
    imag += a.imag () * b.real();

    return complex<T>{
      real,
      imag
    };
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<complex<T>, T, complex<T>> {
  MUTLASS_HOST_DEVICE
  complex<T> operator()(
    complex<T> const &a,
    T const &b,
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a.real() * b;
    imag += a.imag () * b;

    return complex<T>{
      real,
      imag
    };
  }
};

/// Fused multiply-add
template <typename T>
struct multiply_add<T, complex<T>, complex<T>> {
  MUTLASS_HOST_DEVICE
  complex<T> operator()(
    T const &a,
    complex<T> const &b,
    complex<T> const &c) const {

    T real = c.real();
    T imag = c.imag();

    real += a * b.real();
    imag += a * b.imag();

    return complex<T>{
      real,
      imag
    };
  }
};

/// Conjugate
template <typename T>
struct conjugate<complex<T>>  {
  MUTLASS_HOST_DEVICE
  complex<T> operator()(complex<T> const &a) const {
    return conj(a);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output>
struct magnitude_squared_difference<complex<T>, Output> {
  MUTLASS_HOST_DEVICE
  Output operator()(complex<T> lhs, complex<T> rhs) const {
    multiplies<Output> mul_op;

    Output y_r = Output(lhs.real()) - Output(rhs.real());
    Output y_i = Output(lhs.imag()) - Output(rhs.imag());

    return mul_op(y_r, y_r) + mul_op(y_i, y_i);
  }
};

/// Reduces value into the data pointed to by ptr (complex<T> specialization)
template <typename T>
struct atomic_add<complex<T>> {
  MUTLASS_DEVICE
  void operator()(complex<T> *ptr, const complex<T> &data)
  {
    data.red(ptr);
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mutlass

//////////////////////////////////////////////////////////////////////////////////////////////////
