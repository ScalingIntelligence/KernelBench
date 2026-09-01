/* Lower MP31 TME descriptors and barrier buffers to device/host ABI values. */

#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

#include "musa/runtime.h"
#include "musa/target_utils.h"
#include "op/builtin.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace musa {

using namespace tirx;
using namespace ffi;

class LowerTMEIntrinPass : public StmtExprMutator {
public:
  explicit LowerTMEIntrinPass(bool enable_prefetch)
      : enable_prefetch_(enable_prefetch) {}

  static PrimFunc Substitute(PrimFunc f, bool enable_prefetch) {
    LowerTMEIntrinPass pass(enable_prefetch);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = pass.VisitStmt(f->body);

    if (!pass.desc_init_args_.empty()) {
      f = WithAttr(std::move(f), "tma_descriptor_args", pass.desc_init_args_);
      Array<Stmt> prologue;
      for (const auto &entry : pass.desc_inits_) {
        Call alloc = Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                          {StringImm("tvm_ffi_any"), 16});
        Call init = Call(DataType::Handle(), builtin::tvm_call_packed(),
                         entry.second);
        prologue.push_back(tirx::Bind(entry.first, alloc));
        prologue.push_back(Evaluate(init));
      }
      fptr = f.CopyOnWrite();
      prologue.push_back(fptr->body);
      fptr->body = SeqStmt(prologue);
    }
    return f;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool is_thread_extent = op->attr_key == tirx::attr::thread_extent;
    const auto *iter_var = op->node.as<IterVarNode>();
    bool is_thread_idx_x =
        iter_var != nullptr && iter_var->thread_tag == "threadIdx.x";
    if (!is_thread_extent || !is_thread_idx_x) {
      return StmtExprMutator::VisitStmt_(op);
    }

    AttrStmt result = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
    if (!enable_prefetch_ || prefetch_calls_.empty()) {
      return result;
    }

    Array<Stmt> body;
    Stmt prefetch = prefetch_calls_.size() == 1
                        ? prefetch_calls_[0]
                        : Stmt(SeqStmt(prefetch_calls_));
    body.push_back(IfThenElse(
        Call(DataType::Bool(), Op::Get("tl.tl_shuffle_elect"), {32}),
        prefetch));
    body.push_back(result->body);
    result.CopyOnWrite()->body = SeqStmt(body);
    prefetch_calls_.clear();
    return result;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(create_tma_descriptor())) {
      return StmtExprMutator::VisitExpr_(op);
    }

    Call call = GetRef<Call>(op);
    auto it = descriptor_map_.find(call);
    if (it != descriptor_map_.end()) {
      return it->second;
    }

    ICHECK_GE(op->args.size(), 3U);
    const auto *base = op->args[2].as<VarNode>();
    ICHECK(base) << "MP31 TME descriptor global address must be a buffer var";
    Var descriptor(base->name_hint + "_desc",
                   PointerType(TensorMapType(), "grid_constant"));
    descriptor_map_.emplace(call, descriptor);

    Array<PrimExpr> init_args;
    init_args.push_back(StringImm(tvm_musa_tensordesc_create_tiled));
    init_args.push_back(descriptor);
    init_args.insert(init_args.end(), op->args.begin(), op->args.end());
    desc_init_args_.Set(descriptor, init_args);
    desc_inits_.push_back({descriptor, init_args});
    prefetch_calls_.push_back(Evaluate(Call(
        DataType::Handle(), prefetch_tma_descriptor(), {descriptor})));
    return descriptor;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Array<Buffer> barrier_buffers;
    for (auto buffer : op->alloc_buffers) {
      if (buffer.scope() == "shared.barrier" ||
          buffer.scope() == "shared.cluster_barrier") {
        barrier_buffers.push_back(buffer);
      }
    }
    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // MTCC MP31 reserves async barrier id 0.
    int base = 1;
    for (auto buffer : barrier_buffers) {
      ICHECK(buffer->shape.size() == 1);
      auto extent = buffer->shape[0].as<IntImmNode>();
      ICHECK(extent) << "MP31 TME barriers require a static allocation size";
      barrier_base_.emplace(buffer->data.get(), base);
      base += static_cast<int>(extent->value);
    }

    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    Array<Buffer> kept;
    for (auto buffer : block->alloc_buffers) {
      if (barrier_base_.count(buffer->data.get()) == 0) {
        kept.push_back(buffer);
      }
    }
    block_ptr->alloc_buffers = kept;

    Array<Stmt> init;
    init.push_back(Evaluate(Call(
        DataType::Handle(), builtin::create_barriers(), {base - 1})));
    for (auto buffer : barrier_buffers) {
      auto it = barrier_base_.find(buffer->data.get());
      auto ann = block->annotations.Get("barrier_init");
      if (!ann) {
        continue;
      }
      auto init_map = ann->as<Map<Var, Array<PrimExpr>>>();
      if (!init_map || !init_map.value().count(buffer->data)) {
        continue;
      }
      auto counts = init_map.value().at(buffer->data);
      for (size_t i = 0; i < counts.size(); ++i) {
        init.push_back(Evaluate(Call(
            DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
            {IntImm(DataType::Int(32), it->second + static_cast<int>(i)),
             counts[i]})));
      }
    }
    // Match CUDA LowerSharedBarrier: no TME transaction may observe the
    // barrier before its leader-thread initialization is visible CTA-wide.
    init.push_back(Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                                 {StringImm("shared")})));
    block_ptr->body = SeqStmt({SeqStmt(init), block->body});
    return block;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto it = barrier_base_.find(op->buffer->data.get());
    if (it == barrier_base_.end()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    ICHECK_EQ(op->indices.size(), 1U);
    PrimExpr index = op->indices[0];
    if (it->second != 0) {
      index += it->second;
    }
    return index;
  }

  std::unordered_map<Call, Var, StructuralHash, tirx::ExprDeepEqual>
      descriptor_map_;
  Map<Var, Array<PrimExpr>> desc_init_args_;
  std::vector<std::pair<Var, Array<PrimExpr>>> desc_inits_;
  std::unordered_map<const VarNode *, int> barrier_base_;
  Array<Stmt> prefetch_calls_;
  bool enable_prefetch_{false};
};

namespace transform {
using namespace tirx::transform;

tvm::transform::Pass LowerTMEIntrin() {
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsMP31(target.value())) {
      return f;
    }
    bool enable_prefetch =
        ctx->GetConfig<Bool>(kEnableMusaTmaPrefetch, Bool(false)).value();
    return LowerTMEIntrinPass::Substitute(std::move(f), enable_prefetch);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.musa.LowerTMEIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.musa.transform.LowerTMEIntrin", LowerTMEIntrin);
}

} // namespace transform
} // namespace musa
} // namespace tl
} // namespace tvm
