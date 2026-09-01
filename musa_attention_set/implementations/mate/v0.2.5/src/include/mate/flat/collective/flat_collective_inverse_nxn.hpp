#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/simd_helper.hpp"

namespace mate::flat::collective {

using namespace mute;

template <class Element_, int Block_>
struct CollectiveLowerTriangularInverseNxN {
  using Element = Element_;

  // The current implementation maps one column solve to one thread, so Block
  // must fit within a single warp.
  static constexpr int Block   = Block_;
  static constexpr int VecSize = 4;

  using SmemLayout = decltype(make_layout(make_shape(Int<Block>{}, Int<Block>{}), make_stride(Int<Block>{}, _1{})));

  struct SharedStorage {
    alignas(16) float matrix[cosize_v<SmemLayout>];
  };

  static_assert(Block > 0);
  static_assert(Block <= mutlass::NumThreadsPerWarp);
  static_assert(Block % VecSize == 0);

  template <class TensorMatrix, class TensorBeta>
  MUTLASS_DEVICE static void solve_column(TensorMatrix const& matrix,
                                          TensorBeta const&   beta,
                                          int                 col,
                                          float*              solution) {
    MUTE_UNROLL
    for (int row = 0; row < Block; ++row) {
      solution[row] = 0.0f;
    }

    float rhs = beta(col);

    MUTE_UNROLL
    for (int row = 0; row < Block; ++row) {
      float value = row == col ? -rhs : 0.0f;

      MUTE_UNROLL
      for (int k = 0; k < row; k += VecSize) {
        float4 coeffs = simd::load_float4(&matrix(row, k));
        float4 sol    = *reinterpret_cast<float4 const*>(solution + k);
        simd::dot4(coeffs, sol, value);
      }

      solution[row] = -value;
    }
  }

  template <class TensorAccum, class ThreadMma, class TensorOutput, class TensorBeta, class SyncBarrier>
  MUTLASS_DEVICE void operator()(SharedStorage&     shared_storage,
                                 TensorAccum const& accum,
                                 ThreadMma&         thread_mma,
                                 TensorOutput&&     inv_output,
                                 TensorBeta const&  beta,
                                 SyncBarrier&&      sync_barrier,
                                 int                local_tid) const {
    auto matrix = make_tensor(make_smem_ptr(shared_storage.matrix), SmemLayout{});
    auto cId    = make_identity_tensor(make_shape(Int<Block>{}, Int<Block>{}));
    auto tCc    = thread_mma.partition_C(cId);
    static_assert(decltype(size(accum))::value % VecSize == 0);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum) / VecSize; ++i) {
      int base = i * VecSize;
      int row  = int(get<0>(tCc(base + 0)));

      float4 values  = *reinterpret_cast<float4 const*>(&accum(base));
      values         = simd::vmul(values, beta(row));
      auto value_ptr = reinterpret_cast<float const*>(&values);

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < VecSize; ++j) {
        int col          = int(get<1>(tCc(base + j)));
        matrix(row, col) = row > col ? value_ptr[j] : 0.0f;
      }
    }

    sync_barrier();

    if (local_tid < Block) {
      int               col = local_tid;
      alignas(16) float solution[Block];
      using SolutionVec = mutlass::Array<float, VecSize>;
      using ElementVec  = mutlass::Array<Element, VecSize>;
      mutlass::NumericArrayConverter<Element, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest>
          convert_solution;
      solve_column(matrix, beta, col, solution);

      MUTE_UNROLL
      for (int row = 0; row < Block; row += VecSize) {
        SolutionVec values                   = *reinterpret_cast<SolutionVec const*>(solution + row);
        ElementVec  packed                   = convert_solution(values);
        inv_output(make_coord(row + 0, col)) = packed[0];
        inv_output(make_coord(row + 1, col)) = packed[1];
        inv_output(make_coord(row + 2, col)) = packed[2];
        inv_output(make_coord(row + 3, col)) = packed[3];
      }
    }
  }
};

}  // namespace mate::flat::collective
