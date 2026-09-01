/*!
 * \file tl/musa/op/gemm/mp31_sqmma.cc
 * \brief MP31 SQMMA implementation hooks for tl.gemm.
 */

#include "musa/op/gemm/mp31_sqmma.h"

#include "musa/target_utils.h"
#include "op/utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace musa {
namespace mp31 {
namespace {

constexpr const char *kSQMMA = "musa.sqmma";

enum class SQMMATypeClass : uint8_t {
  kFP16,
  kBF16,
  kTF32,
  kFP8,
  kInt8,
  kUInt8,
  kFloatInt8,
  kInt8Float,
};

struct SQMMAShapeMN {
  int m;
  int n;
};

struct SQMMASquadTile {
  int64_t m;
  int64_t n;
};

bool IsMP31FP8(DataType dtype) {
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

bool IsMP31TF32(DataType dtype) { return dtype.is_tfloat32(); }

bool IsMP31Float16(DataType dtype) {
  return dtype == DataType::Float(16) || dtype == DataType::BFloat(16);
}

std::optional<SQMMATypeClass> GetSQMMATypeClass(const GemmNode &op) {
  const DataType a_dtype = op.a_->dtype;
  const DataType b_dtype = op.b_->dtype;
  const DataType c_dtype = op.c_->dtype;
  if (a_dtype == DataType::Float(16) && b_dtype == DataType::Float(16) &&
      c_dtype == DataType::Float(32)) {
    return SQMMATypeClass::kFP16;
  }
  if (a_dtype == DataType::BFloat(16) && b_dtype == DataType::BFloat(16) &&
      c_dtype == DataType::Float(32)) {
    return SQMMATypeClass::kBF16;
  }
  if (IsMP31TF32(a_dtype) && IsMP31TF32(b_dtype) &&
      c_dtype == DataType::Float(32)) {
    return SQMMATypeClass::kTF32;
  }
  if (IsMP31FP8(a_dtype) && IsMP31FP8(b_dtype) &&
      c_dtype == DataType::Float(32)) {
    return SQMMATypeClass::kFP8;
  }
  if (a_dtype == DataType::Int(8) && b_dtype == DataType::Int(8) &&
      c_dtype == DataType::Int(32)) {
    return SQMMATypeClass::kInt8;
  }
  if (a_dtype == DataType::UInt(8) && b_dtype == DataType::UInt(8) &&
      c_dtype == DataType::Int(32)) {
    return SQMMATypeClass::kUInt8;
  }
  if (IsMP31Float16(a_dtype) && b_dtype == DataType::Int(8) &&
      c_dtype == DataType::Float(32)) {
    return SQMMATypeClass::kFloatInt8;
  }
  if (a_dtype == DataType::Int(8) && IsMP31Float16(b_dtype) &&
      c_dtype == DataType::Float(32)) {
    return SQMMATypeClass::kInt8Float;
  }
  return std::nullopt;
}

const std::vector<SQMMAShapeMN> &GetMNCandidates(SQMMATypeClass type_class) {
  static const std::vector<SQMMAShapeMN> kStandard = {
      {128, 128}, {128, 64}, {128, 32}, {64, 128}, {64, 64}, {64, 32},
      {64, 16},   {32, 128}, {32, 64},  {32, 32},  {16, 64},
  };
  static const std::vector<SQMMAShapeMN> kTF32 = {
      {128, 128}, {128, 64}, {64, 64}, {64, 32},
      {64, 16},   {32, 64},  {32, 32}, {16, 64},
  };
  static const std::vector<SQMMAShapeMN> kFloatInt = {
      {128, 128}, {128, 64}, {128, 32}, {64, 128}, {64, 64},
      {64, 32},   {32, 128}, {32, 64},  {32, 32},  {16, 64},
  };
  static const std::vector<SQMMAShapeMN> kIntFloat = {
      {128, 128}, {128, 64}, {128, 32}, {64, 128}, {64, 64},
      {64, 32},   {64, 16},  {32, 128}, {32, 64},  {32, 32},
  };

  switch (type_class) {
  case SQMMATypeClass::kTF32:
    return kTF32;
  case SQMMATypeClass::kFloatInt8:
    return kFloatInt;
  case SQMMATypeClass::kInt8Float:
    return kIntFloat;
  default:
    return kStandard;
  }
}

const std::vector<int> &GetKCandidates(SQMMATypeClass type_class) {
  static const std::vector<int> kFP16 = {64, 32, 16};
  static const std::vector<int> kTF32 = {32, 16, 8};
  static const std::vector<int> kEightBit = {128, 64, 32};
  static const std::vector<int> kMixed = {64, 32};

  if (type_class == SQMMATypeClass::kFP16 ||
      type_class == SQMMATypeClass::kBF16) {
    return kFP16;
  }
  if (type_class == SQMMATypeClass::kTF32) {
    return kTF32;
  }
  if (type_class == SQMMATypeClass::kFloatInt8 ||
      type_class == SQMMATypeClass::kInt8Float) {
    return kMixed;
  }
  return kEightBit;
}

SQMMASquadTile GetSquadTile(const GemmNode &op, int block_size, int m_warp,
                            int n_warp, const Target &target) {
  ICHECK(TargetIsMP31(target)) << "SQMMA is only supported on MP31";
  ICHECK(IsSharedBuffer(op.a_))
      << "MP31 SQMMA requires A in shared scope, got " << op.a_.scope();
  ICHECK(IsSharedBuffer(op.b_))
      << "MP31 SQMMA requires B in shared scope, got " << op.b_.scope();
  ICHECK(IsFragmentBuffer(op.c_))
      << "MP31 SQMMA requires C in local.fragment scope, got " << op.c_.scope();

  const int warp_size = TargetMUSAGetWarpSize(target);
  ICHECK_EQ(block_size % warp_size, 0);
  const int num_warps = block_size / warp_size;
  ICHECK_EQ(m_warp * n_warp, num_warps)
      << "m_warp * n_warp must equal the block warp count";
  ICHECK_EQ(m_warp % 4, 0)
      << "MP31 SQMMA requires m_warp to be a multiple of 4";
  ICHECK_EQ(op.m_ % (m_warp * 4), 0)
      << "MP31 SQMMA M tile is incompatible with m_warp";
  ICHECK_EQ(op.n_ % (n_warp * 8), 0)
      << "MP31 SQMMA N tile is incompatible with n_warp";
  return {op.m_ / (m_warp / 4), op.n_ / n_warp};
}

std::optional<std::array<int, 3>> SelectNativeShape(SQMMATypeClass type_class,
                                                    const SQMMASquadTile &tile,
                                                    int64_t k, bool a_k_major,
                                                    bool b_k_major) {
  for (const SQMMAShapeMN &shape : GetMNCandidates(type_class)) {
    if (tile.m % shape.m != 0 || tile.n % shape.n != 0)
      continue;
    if (type_class == SQMMATypeClass::kTF32 &&
        ((!a_k_major && shape.m >= 128) || (!b_k_major && shape.n >= 128))) {
      continue;
    }
    for (int inst_k : GetKCandidates(type_class)) {
      if (k % inst_k == 0) {
        return std::array<int, 3>{shape.m, shape.n, inst_k};
      }
    }
  }
  return std::nullopt;
}

} // namespace

bool SQMMA::IsInstruction(const String &gemm_inst) {
  return gemm_inst == kSQMMA;
}

String SQMMA::SelectInst(const GemmNode &op, int block_size, Target target) {
  ICHECK(TargetIsMP31(target)) << "SQMMA is only supported on MP31";
  auto [m_warp, n_warp] = ComputeWarpPartition(*op.policy_.get(), op.m_, op.n_,
                                               block_size, target, kSQMMA);
  (void)GetInstShape(op, block_size, m_warp, n_warp, target);
  return kSQMMA;
}

std::pair<int, int>
SQMMA::ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int block_size, Target target, String gemm_inst) {
  ICHECK(IsInstruction(gemm_inst))
      << "Unsupported MP31 SQMMA instruction: " << gemm_inst;
  ICHECK(TargetIsMP31(target)) << "SQMMA is only supported on MP31";
  const int warp_size = TargetMUSAGetWarpSize(target);
  ICHECK_EQ(block_size % warp_size, 0)
      << "MP31 SQMMA block threads must be divisible by warp size " << warp_size
      << ", got " << block_size;
  const int num_warps = block_size / warp_size;
  constexpr int kWarpGroupSize = 4;
  constexpr int kMPerWarp = 4;
  constexpr int kNPerWarp = 8;
  ICHECK_EQ(num_warps % kWarpGroupSize, 0)
      << "MP31 SQMMA requires 128*k threads, got " << block_size;
  ICHECK_EQ(M % kMPerWarp, 0)
      << "MP31 SQMMA M must be divisible by " << kMPerWarp << ", got " << M;
  ICHECK_EQ(N % kNPerWarp, 0)
      << "MP31 SQMMA N must be divisible by " << kNPerWarp << ", got " << N;

  auto is_valid = [&](int m, int n) {
    return m * n == num_warps && m % kWarpGroupSize == 0 &&
           M % (m * kMPerWarp) == 0 && N % (n * kNPerWarp) == 0;
  };

  int m_warp = 0;
  int n_warp = 0;
  if (policy.IsFullRow()) {
    for (int m = num_warps; m >= kWarpGroupSize; m -= kWarpGroupSize) {
      if (num_warps % m != 0 || !is_valid(m, num_warps / m))
        continue;
      m_warp = m;
      n_warp = num_warps / m;
      break;
    }
  } else if (policy.IsFullCol()) {
    for (int n = num_warps / kWarpGroupSize; n >= 1; --n) {
      if (num_warps % n != 0 || !is_valid(num_warps / n, n))
        continue;
      m_warp = num_warps / n;
      n_warp = n;
      break;
    }
  } else if (policy.IsSquare()) {
    const float ideal_ratio =
        N > 0 ? static_cast<float>(M) / static_cast<float>(N) : 1.0f;
    float best_score = std::numeric_limits<float>::max();
    for (int m = kWarpGroupSize; m <= num_warps; m += kWarpGroupSize) {
      if (num_warps % m != 0)
        continue;
      const int n = num_warps / m;
      if (!is_valid(m, n))
        continue;
      const float m_per_warp =
          static_cast<float>(M) / static_cast<float>(m * kMPerWarp);
      const float n_per_warp =
          static_cast<float>(N) / static_cast<float>(n * kNPerWarp);
      const float score = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (score < best_score) {
        best_score = score;
        m_warp = m;
        n_warp = n;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy for MP31 SQMMA";
  }

  ICHECK_GT(m_warp, 0) << "No valid MP31 SQMMA warp partition for M=" << M
                       << ", N=" << N << ", threads=" << block_size;
  ICHECK_GT(n_warp, 0);
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

Array<Integer> SQMMA::GetInstShape(const GemmNode &op, int block_size,
                                   int m_warp, int n_warp,
                                   const Target &target) {
  const SQMMASquadTile tile =
      GetSquadTile(op, block_size, m_warp, n_warp, target);
  const auto type_class = GetSQMMATypeClass(op);
  ICHECK(type_class.has_value())
      << "Unsupported MP31 SQMMA dtype combination: A=" << op.a_->dtype
      << ", B=" << op.b_->dtype << ", C=" << op.c_->dtype;

  const auto shape =
      SelectNativeShape(*type_class, tile, op.k_, !op.transA_, op.transB_);
  ICHECK(shape.has_value())
      << "No native MP31 SQMMA instruction can cover M=" << op.m_
      << ", N=" << op.n_ << ", K=" << op.k_ << " with m_warp=" << m_warp
      << ", n_warp=" << n_warp;
  return {Integer((*shape)[0]), Integer((*shape)[1]), Integer((*shape)[2])};
}

bool SQMMA::ReuseExistingSharedLayout(String gemm_inst) {
  ICHECK(IsInstruction(gemm_inst))
      << "Unsupported MP31 SQMMA instruction: " << gemm_inst;
  return false;
}

} // namespace mp31
} // namespace musa

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.get_mp31_sqmma_inst_shape",
      [](Gemm gemm, int block_size, int m_warp, int n_warp, Target target) {
        return musa::mp31::SQMMA::GetInstShape(*gemm.operator->(), block_size,
                                               m_warp, n_warp, target);
      });
}

} // namespace tl
} // namespace tvm
