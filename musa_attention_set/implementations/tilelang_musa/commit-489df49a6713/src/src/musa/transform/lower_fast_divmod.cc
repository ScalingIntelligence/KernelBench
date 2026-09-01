/*!
 * \file lower_fast_divmod.cc
 * \brief Prepare explicit Common MUSA fast divmod on the host or device.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "musa/op/fast_divmod.h"

#include "support/check.h"

namespace tvm {
namespace tl {
namespace musa {

using namespace tirx;

namespace {

class VarCollector : public ExprVisitor {
public:
  std::vector<Var> Collect(const PrimExpr &expr) {
    VisitExpr(expr);
    return vars_;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    if (seen_.insert(op).second) {
      vars_.push_back(ffi::GetRef<Var>(op));
    }
  }

  std::unordered_set<const VarNode *> seen_;
  std::vector<Var> vars_;
};

class HostEvaluableExprChecker : public ExprVisitor {
public:
  explicit HostEvaluableExprChecker(
      const std::unordered_set<const VarNode *> &host_vars)
      : host_vars_(host_vars) {}

  bool Check(const PrimExpr &expr) {
    VisitExpr(expr);
    return host_evaluable_;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    if (!host_vars_.count(op)) {
      host_evaluable_ = false;
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    host_evaluable_ = false;
  }

  void VisitExpr_(const CallNode *op) final {
    // Calls may be device-only even when all their operands are host Vars.
    host_evaluable_ = false;
  }

  const std::unordered_set<const VarNode *> &host_vars_;
  bool host_evaluable_{true};
};

struct MagicNumbers {
  uint32_t multiplier;
  uint32_t shift_right;
};

MagicNumbers ComputeMagicNumbers(int32_t divisor) {
  ICHECK_GT(divisor, 0)
      << "T.fast_div, T.fast_mod, and T.fast_divmod require a strictly "
         "positive divisor";
  if (divisor == 1) {
    return {0u, 0u};
  }

  uint32_t denominator = static_cast<uint32_t>(divisor);
  uint32_t ceil_log2 = 0;
  for (uint32_t value = denominator - 1; value != 0; value >>= 1) {
    ++ceil_log2;
  }
  uint32_t exponent = 31 + ceil_log2;
  uint64_t multiplier =
      ((uint64_t{1} << exponent) + denominator - 1) / denominator;
  return {static_cast<uint32_t>(multiplier), exponent - 32};
}

class FastDivmodLowerer : public StmtExprMutator {
public:
  static PrimFunc Lower(PrimFunc func) {
    std::unordered_set<const VarNode *> host_vars;
    auto collect_expr_vars = [&host_vars](const PrimExpr &expr) {
      for (const Var &var : VarCollector().Collect(expr)) {
        host_vars.insert(var.get());
      }
    };

    for (const Var &param : func->params) {
      host_vars.insert(param.get());
    }
    for (const auto &[param, buffer] : func->buffer_map) {
      host_vars.insert(param.get());
      host_vars.insert(buffer->data.get());
      for (const PrimExpr &shape : buffer->shape) {
        collect_expr_vars(shape);
      }
      for (const PrimExpr &stride : buffer->strides) {
        collect_expr_vars(stride);
      }
      collect_expr_vars(buffer->elem_offset);
    }

    FastDivmodLowerer lowerer(std::move(host_vars));
    Stmt body = lowerer.VisitStmt(func->body);
    body = lowerer.PrependHostSetup(std::move(body));

    auto *node = func.CopyOnWrite();
    node->body = std::move(body);
    return func;
  }

private:
  struct DivisorInfo {
    PrimExpr divisor;
    PrimExpr multiplier;
    PrimExpr shift_right;
    bool needs_host_setup{false};
    bool needs_device_setup{false};
  };

  explicit FastDivmodLowerer(
      std::unordered_set<const VarNode *> host_vars)
      : host_vars_(std::move(host_vars)) {}

  static void CheckOperandTypes(const PrimExpr &dividend,
                                const PrimExpr &divisor) {
    ICHECK_EQ(dividend.dtype(), DataType::Int(32))
        << "T.fast_div, T.fast_mod, and T.fast_divmod require an int32 "
           "dividend, but got "
        << dividend.dtype();
    ICHECK_EQ(divisor.dtype(), DataType::Int(32))
        << "T.fast_div, T.fast_mod, and T.fast_divmod require an int32 "
           "divisor, but got "
        << divisor.dtype();
  }

  DivisorInfo &GetOrCreateDivisorInfo(const PrimExpr &divisor) {
    ExprDeepEqual equal;
    for (DivisorInfo &info : divisors_) {
      if (equal(info.divisor, divisor)) {
        return info;
      }
    }

    DivisorInfo info;
    info.divisor = divisor;
    if (const auto *immediate = divisor.as<IntImmNode>()) {
      MagicNumbers magic =
          ComputeMagicNumbers(static_cast<int32_t>(immediate->value));
      info.multiplier = IntImm(DataType::UInt(32), magic.multiplier);
      info.shift_right = IntImm(DataType::UInt(32), magic.shift_right);
    } else if (HostEvaluableExprChecker(host_vars_).Check(divisor)) {
      size_t index = divisors_.size();
      info.multiplier = Var("musa_fast_div_multiplier_" +
                                std::to_string(index),
                            DataType::UInt(32));
      info.shift_right = Var("musa_fast_div_shift_" + std::to_string(index),
                             DataType::UInt(32));
      info.needs_host_setup = true;
    } else {
      info.needs_device_setup = true;
    }

    divisors_.push_back(std::move(info));
    return divisors_.back();
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    bool is_fast_div = op->op.same_as(fast_div());
    bool is_fast_mod = op->op.same_as(fast_mod());
    if (!is_fast_div && !is_fast_mod) {
      return StmtExprMutator::VisitExpr_(op);
    }

    ICHECK_EQ(op->args.size(), 2U)
        << (is_fast_div ? "tl.musa.fast_div" : "tl.musa.fast_mod")
        << " expects dividend and divisor";
    PrimExpr dividend = VisitExpr(op->args[0]);
    PrimExpr divisor = VisitExpr(op->args[1]);
    CheckOperandTypes(dividend, divisor);

    DivisorInfo &info = GetOrCreateDivisorInfo(divisor);
    if (info.needs_device_setup) {
      return Call(op->dtype,
                  is_fast_div ? fast_div_device() : fast_mod_device(),
                  {dividend, divisor}, op->annotations, op->span);
    }
    return Call(op->dtype,
                is_fast_div ? fast_div_precomputed()
                            : fast_mod_precomputed(),
                {dividend, divisor, info.multiplier, info.shift_right},
                op->annotations, op->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    ffi::Array<Stmt> rewritten;
    rewritten.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      rewritten.push_back(VisitStmt(stmt));
    }

    ffi::Array<Stmt> fused;
    for (size_t index = 0; index < rewritten.size(); ++index) {
      const auto *quotient_bind = rewritten[index].as<BindNode>();
      const auto *quotient = quotient_bind == nullptr
                                 ? nullptr
                                 : quotient_bind->value.as<CallNode>();
      if (quotient == nullptr || index + 1 >= rewritten.size()) {
        fused.push_back(rewritten[index]);
        continue;
      }

      const auto *remainder_bind = rewritten[index + 1].as<BindNode>();
      const auto *remainder = remainder_bind == nullptr
                                  ? nullptr
                                  : remainder_bind->value.as<CallNode>();
      bool matching_kinds =
          remainder != nullptr &&
          ((quotient->op.same_as(fast_div_precomputed()) &&
            remainder->op.same_as(fast_mod_precomputed())) ||
           (quotient->op.same_as(fast_div_device()) &&
            remainder->op.same_as(fast_mod_device())));
      if (!matching_kinds || !SameFastDivArgs(*quotient, *remainder)) {
        fused.push_back(rewritten[index]);
        continue;
      }

      PrimExpr fused_remainder =
          quotient->args[0] - quotient_bind->var * quotient->args[1];
      fused.push_back(rewritten[index]);
      fused.push_back(Bind(remainder_bind->var, fused_remainder,
                           remainder_bind->span));
      ++index;
    }
    return SeqStmt::Flatten(fused);
  }

  static bool SameFastDivArgs(const CallNode &lhs, const CallNode &rhs) {
    if (lhs.args.size() != rhs.args.size() ||
        (lhs.args.size() != 2U && lhs.args.size() != 4U)) {
      return false;
    }
    ExprDeepEqual equal;
    for (size_t index = 0; index < lhs.args.size(); ++index) {
      if (!equal(lhs.args[index], rhs.args[index])) {
        return false;
      }
    }
    return true;
  }

  static PrimExpr MakeCeilLog2(const PrimExpr &divisor) {
    DataType u32 = DataType::UInt(32);
    PrimExpr divisor_u32 = Cast(u32, divisor);
    PrimExpr divisor_minus_one =
        max(divisor_u32 - make_const(u32, 1), make_const(u32, 1));
    // The host wrapper has no corresponding MUSA C API; use the compiler builtin.
    PrimExpr leading_zeros = Cast(
        u32, Call(DataType::Int(32), builtin::call_pure_extern(),
                  {StringImm("__builtin_clz"), divisor_minus_one}));
    PrimExpr ceil_log2 = make_const(u32, 32) - leading_zeros;
    return Select(divisor == make_const(divisor.dtype(), 1),
                  make_const(u32, 0), ceil_log2);
  }

  static PrimExpr MakeMultiplier(const PrimExpr &divisor) {
    DataType u32 = DataType::UInt(32);
    DataType i64 = DataType::Int(64);
    PrimExpr exponent = make_const(u32, 31) + MakeCeilLog2(divisor);
    PrimExpr numerator = make_const(i64, 1) << Cast(i64, exponent);
    PrimExpr divisor_i64 = Cast(i64, divisor);
    PrimExpr multiplier =
        (numerator + divisor_i64 - make_const(i64, 1)) / divisor_i64;
    return Select(divisor == make_const(divisor.dtype(), 1),
                  make_const(u32, 0), Cast(u32, multiplier));
  }

  static PrimExpr MakeShiftRight(const PrimExpr &divisor) {
    DataType u32 = DataType::UInt(32);
    PrimExpr exponent = make_const(u32, 31) + MakeCeilLog2(divisor);
    PrimExpr shift_right = exponent - make_const(u32, 32);
    return Select(divisor == make_const(divisor.dtype(), 1),
                  make_const(u32, 0), shift_right);
  }

  Stmt PrependHostSetup(Stmt body) const {
    ffi::Array<Stmt> setup;
    for (const DivisorInfo &info : divisors_) {
      if (!info.needs_host_setup) {
        continue;
      }
      setup.push_back(AssertStmt(
          info.divisor > make_const(info.divisor.dtype(), 0),
          StringImm("ValueError"),
          {StringImm("T.fast_div, T.fast_mod, and T.fast_divmod require a "
                     "strictly positive divisor")}));
      setup.push_back(Bind(Downcast<Var>(info.multiplier),
                           MakeMultiplier(info.divisor)));
      setup.push_back(Bind(Downcast<Var>(info.shift_right),
                           MakeShiftRight(info.divisor)));
    }
    return SeqStmt::Flatten(setup, body);
  }

  std::unordered_set<const VarNode *> host_vars_;
  std::vector<DivisorInfo> divisors_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass LowerFastDivmod() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const PassContext &ctx) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || target.value()->kind->name != "musa") {
      return func;
    }
    return FastDivmodLowerer::Lower(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.musa.LowerFastDivmod", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.musa.transform.LowerFastDivmod",
                        LowerFastDivmod);
}

} // namespace musa
} // namespace tl
} // namespace tvm
