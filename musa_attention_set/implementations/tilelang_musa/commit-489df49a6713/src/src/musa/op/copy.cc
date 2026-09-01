/*!
 * \file tl/musa/op/copy.cc
 * \brief MUSA implementation for tl.copy lowering.
 */

#include "musa/op/copy.h"

#include "backend/common/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "musa/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include "support/check.h"

#include <optional>

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

namespace {

bool IsExplicitAsyncCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("is_async_copy")) {
    if (const auto *flag = value.value().as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

bool IsExplicitTmaCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("is_tma_copy")) {
    if (const auto *flag = value->as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

// Values are part of the stable MUSA descriptor ABI. Including musa.h in
// this host compiler translation unit conflicts with TileLang's CUDA stubs;
// the MP31 device template includes the authoritative toolkit header instead.
constexpr int kDescInt8 = 0;
constexpr int kDescUInt8 = 1;
constexpr int kDescInt16 = 2;
constexpr int kDescUInt16 = 3;
constexpr int kDescFloat16 = 4;
constexpr int kDescBFloat16 = 5;
constexpr int kDescInt32 = 6;
constexpr int kDescUInt32 = 7;
constexpr int kDescFloat32 = 8;
constexpr int kDescInt64 = 10;
constexpr int kDescUInt64 = 11;
constexpr int kDescFloat64 = 12;

PrimExpr GetTmaBarrier(const CopyNode &op) {
  auto value = op.annotations.Get("barrier");
  ICHECK(value) << "T.tma_copy() load requires a barrier argument on MP31";
  auto barrier = value->as<PrimExprNode>();
  ICHECK(barrier) << "T.tma_copy() barrier annotation must be a PrimExpr";
  return GetRef<PrimExpr>(barrier);
}

int GetTmaDescriptorDataType(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return kDescBFloat16;
  }
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 16:
      return kDescFloat16;
    case 32:
      return kDescFloat32;
    case 64:
      return kDescFloat64;
    default:
      break;
    }
  }
  if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 8:
      return kDescInt8;
    case 16:
      return kDescInt16;
    case 32:
      return kDescInt32;
    case 64:
      return kDescInt64;
    default:
      break;
    }
  }
  if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 8:
      return kDescUInt8;
    case 16:
      return kDescUInt16;
    case 32:
      return kDescUInt32;
    case 64:
      return kDescUInt64;
    default:
      break;
    }
  }
  LOG(FATAL) << "MP31 TME does not support descriptor dtype " << dtype;
  return 0;
}

Array<PrimExpr> Reverse(const Array<PrimExpr> &values) {
  Array<PrimExpr> result;
  result.reserve(values.size());
  for (auto it = values.rbegin(); it != values.rend(); ++it) {
    result.push_back(*it);
  }
  return result;
}

Array<PrimExpr> ReverseRanges(const Array<Range> &ranges, bool extent) {
  Array<PrimExpr> values;
  values.reserve(ranges.size());
  for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
    values.push_back(extent ? (*it)->extent : (*it)->min);
  }
  return values;
}

struct LoweredTMEDesc {
  PrimExpr descriptor;
  Array<PrimExpr> global_coords;
  Array<PrimExpr> box_dims;
};

bool TmaDescriptorSupportsDataType(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return true;
  }
  if (dtype.is_float()) {
    return dtype.bits() == 16 || dtype.bits() == 32 || dtype.bits() == 64;
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
           dtype.bits() == 64;
  }
  return false;
}

std::optional<Array<PrimExpr>>
GetTmaGlobalStrideBytes(const Buffer &global_buffer,
                        arith::Analyzer *analyzer) {
  const size_t rank = global_buffer->shape.size();
  Array<PrimExpr> global_shape = Reverse(global_buffer->shape);
  Array<PrimExpr> global_stride;
  if (!global_buffer->strides.empty()) {
    global_stride = Reverse(global_buffer->strides);
  } else {
    PrimExpr stride = Integer(1);
    for (const PrimExpr &shape : global_shape) {
      global_stride.push_back(stride);
      stride = analyzer->Simplify(stride * shape);
    }
  }
  if (global_stride.size() != rank ||
      !analyzer->CanProveEqual(global_stride[0], Integer(1))) {
    return std::nullopt;
  }

  Array<PrimExpr> result;
  result.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    PrimExpr stride =
        analyzer->Simplify(global_stride[i] * global_buffer->dtype.bytes());
    if (i != 0) {
      if (const auto *imm = stride.as<IntImmNode>()) {
        if (imm->value % 16 != 0) {
          return std::nullopt;
        }
      }
    }
    result.push_back(cast(DataType::UInt(64), stride));
  }
  return result;
}

PrimExpr MakeIm2ColTmaDescriptor(const Buffer &global_buffer,
                                 const Array<PrimExpr> &box_dims,
                                 const Array<PrimExpr> &global_stride_bytes) {
  size_t rank = global_buffer->shape.size();
  ICHECK_EQ(rank, 4) << "MP31 TME im2col requires an NHWC 4D input";
  Array<PrimExpr> global_shape = Reverse(global_buffer->shape);
  TMEDesc desc{/*rank=*/rank,
               /*data_type=*/GetTmaDescriptorDataType(global_buffer->dtype),
               /*global_shape=*/global_shape,
               /*global_stride=*/global_stride_bytes,
               /*smem_box=*/box_dims,
               /*smem_stride=*/Array<PrimExpr>(rank, Integer(1)),
               /*global_addr=*/global_buffer->data,
               /*interleave=*/0,
               /*swizzle=*/0,
               /*l2_promotion=*/0,
               /*oob_fill=*/0};
  return Call(DataType::Handle(), create_tma_descriptor(),
              desc.EncodeCallArgs());
}

LoweredTMEDesc MakeTmaDescriptor(const Buffer &global_buffer,
                                 const Array<Range> &global_range,
                                 const Array<Range> &shared_range,
                                 arith::Analyzer *analyzer) {
  const size_t rank = global_range.size();
  ICHECK_GE(rank, 1U);
  ICHECK_LE(rank, 5U);
  ICHECK_EQ(shared_range.size(), rank);

  // TME descriptors are expressed in bytes and use the innermost dimension
  // first. Restrict the first vertical slice to a row-major contiguous tile;
  // swizzle and split-box support will be added in later MP31 commits.
  Array<PrimExpr> global_shape = Reverse(global_buffer->shape);
  Array<PrimExpr> global_coords = ReverseRanges(global_range, false);
  // Keep the descriptor's true global shape while the TME box follows the
  // normalized shared tile. Coordinates plus box dimensions may cross the
  // descriptor boundary; MP31 then fills those load elements with zero.
  Array<PrimExpr> box_dims = ReverseRanges(shared_range, true);
  Array<PrimExpr> global_stride;
  if (!global_buffer->strides.empty()) {
    global_stride = Reverse(global_buffer->strides);
  } else {
    PrimExpr stride = 1;
    Array<PrimExpr> row_major;
    for (auto shape : global_shape) {
      row_major.push_back(stride);
      stride *= shape;
    }
    global_stride = row_major;
  }
  ICHECK_EQ(global_stride.size(), rank);
  ICHECK(is_one(global_stride[0]))
      << "MP31 TME load requires a contiguous innermost global dimension";

  Array<PrimExpr> global_stride_bytes;
  for (size_t i = 0; i < rank; ++i) {
    PrimExpr stride =
        analyzer->Simplify(global_stride[i] * global_buffer->dtype.bytes());
    if (i != 0) {
      if (auto imm = stride.as<IntImmNode>()) {
        ICHECK_EQ(imm->value % 16, 0)
            << "MP31 TME global stride must be 16-byte aligned, got "
            << stride;
      }
    }
    global_stride_bytes.push_back(cast(DataType::UInt(64), stride));
  }

  for (size_t i = 0; i < rank; ++i) {
    ICHECK(analyzer->CanProveEqual(global_range[rank - i - 1]->extent,
                                   shared_range[rank - i - 1]->extent))
        << "MP31 TME requires matching global/shared tile extents";
  }

  Array<PrimExpr> smem_stride;
  for (size_t i = 0; i < rank; ++i) {
    smem_stride.push_back(1);
  }
  TMEDesc desc{
      rank,
      GetTmaDescriptorDataType(global_buffer->dtype),
      global_shape,
      global_stride_bytes,
      box_dims,
      smem_stride,
      global_buffer->data,
      /*interleave=*/0,
      /*swizzle=*/0,
      /*l2_promotion=*/0,
      /*oob_fill=*/0,
  };
  PrimExpr descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());
  return {descriptor, global_coords, box_dims};
}

PrimExpr MakeTmaSharedPtr(const Buffer &shared_buffer,
                          const Array<Range> &shared_range, int rw_mask) {
  PrimExpr shared_offset = 0;
  PrimExpr stride = 1;
  for (auto it = shared_range.rbegin(); it != shared_range.rend(); ++it) {
    shared_offset += (*it)->min * stride;
    stride *= (*it)->extent;
  }
  return shared_buffer.access_ptr(rw_mask, DataType::Handle(), 1,
                                  shared_offset, stride);
}

Stmt LowerTmaLoad(const CopyNode &op, const LowerArgs &args,
                  arith::Analyzer *analyzer) {
  ICHECK(TargetIsMP31(args.target))
      << "MP31 TME load requires an MP31 MUSA target, got " << args.target;
  ICHECK(IsExplicitTmaCopy(op))
      << "MUSA MP31 TME lowering requires T.tma_copy";
  ICHECK(IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst))
      << "MP31 TME load only supports global-to-shared copies, got src="
      << op.src.scope() << ", dst=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "MP31 TME load requires matching source and destination dtypes";

  LoweredTMEDesc lowered = MakeTmaDescriptor(
      op.src, op.src_range, op.dst_range, analyzer);
  PrimExpr shared_ptr = MakeTmaSharedPtr(op.dst, op.dst_range, /*rw_mask=*/2);

  Array<PrimExpr> tma_args;
  tma_args.push_back(lowered.descriptor);
  tma_args.push_back(GetTmaBarrier(op));
  tma_args.push_back(shared_ptr);
  tma_args.insert(tma_args.end(), lowered.global_coords.begin(),
                  lowered.global_coords.end());
  tma_args.insert(tma_args.end(), lowered.box_dims.begin(),
                  lowered.box_dims.end());
  Stmt load = Evaluate(Call(DataType::Handle(), tma_load(), tma_args));

  PrimExpr bytes = 1;
  for (auto dim : lowered.box_dims) {
    bytes *= dim;
  }
  bytes = analyzer->Simplify(bytes * op.dst->dtype.bytes());
  Stmt expect = Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                              {GetTmaBarrier(op), bytes}));
  load = SeqStmt({expect, load});
  return IfThenElse(EQ(args.thread_index, args.thread_bounds->min), load);
}

Stmt LowerTmaStore(const CopyNode &op, const LowerArgs &args,
                   arith::Analyzer *analyzer) {
  ICHECK(TargetIsMP31(args.target))
      << "MP31 TME store requires an MP31 MUSA target, got " << args.target;
  ICHECK(IsExplicitTmaCopy(op))
      << "MUSA MP31 TME lowering requires T.tma_copy";
  ICHECK(IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst))
      << "MP31 TME store only supports shared-to-global copies, got src="
      << op.src.scope() << ", dst=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "MP31 TME store requires matching source and destination dtypes";

  LoweredTMEDesc lowered = MakeTmaDescriptor(
      op.dst, op.dst_range, op.src_range, analyzer);
  PrimExpr shared_ptr = MakeTmaSharedPtr(op.src, op.src_range, /*rw_mask=*/1);

  Array<PrimExpr> tma_args;
  tma_args.push_back(lowered.descriptor);
  tma_args.push_back(shared_ptr);
  tma_args.insert(tma_args.end(), lowered.global_coords.begin(),
                  lowered.global_coords.end());
  tma_args.insert(tma_args.end(), lowered.box_dims.begin(),
                  lowered.box_dims.end());
  Stmt store = Evaluate(Call(DataType::Handle(), tma_store(), tma_args));
  Stmt commit = Evaluate(Call(DataType::Handle(), tma_store_arrive(), {}));
  return IfThenElse(EQ(args.thread_index, args.thread_bounds->min),
                    SeqStmt({store, commit}));
}

Stmt LowerAsyncCopy(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
  ICHECK(TargetMUSAHasAsyncCopy(lower_args.target))
      << "T.async_copy requires a MUSA target with async-copy support "
         "(mp_21 or newer). Got target="
      << lower_args.target;
  ICHECK(IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst))
      << "T.async_copy only supports global-to-shared copies. Got src scope="
      << op.src.scope() << ", dst scope=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "T.async_copy requires matching source and destination dtypes.";

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);
  for (auto level :
       {InferLevel::kCommon, InferLevel::kStrict, InferLevel::kFree}) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }

  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop = LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true, /*should_vectorize=*/true,
      par_op->LoopLayoutRequiresPaddingGuard());

  auto injected = InjectPTXAsyncCopy(lowered_loop,
                                     /*async_without_async_commit_wait=*/true);
  ICHECK(injected.injected_ptx_async_copy)
      << "T.async_copy requires an eligible global-to-shared vectorized copy.";

  Stmt commit_group =
      Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  return SeqStmt({injected.stmt, commit_group});
}

} // namespace

const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "normal";
  case CopyInst::kAsync:
    return "async";
  case CopyInst::kTMELoad:
    return "tme_load";
  case CopyInst::kTMEStore:
    return "tme_store";
  case CopyInst::kInvalid:
    return "invalid";
  }
  return "unknown";
}

bool CopyInstIsTME(CopyInst inst) {
  return inst == CopyInst::kTMELoad || inst == CopyInst::kTMEStore;
}

bool CopyInstIsTMEStore(CopyInst inst) {
  return inst == CopyInst::kTMEStore;
}

bool CopyInstIsAsync(CopyInst inst) { return inst == CopyInst::kAsync; }

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  if (IsExplicitTmaCopy(op)) {
    if (!TargetIsMP31(ctx.target)) {
      return {CopyInst::kInvalid, false,
              "T.tma_copy() is currently supported only on MP31 MUSA "
              "targets"};
    }
    if (IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst)) {
      return {CopyInst::kTMELoad, true, {}};
    }
    if (IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst)) {
      return {CopyInst::kTMEStore, true, {}};
    }
    return {CopyInst::kInvalid, false,
            "T.tma_copy() only supports global-to-shared loads or "
            "shared-to-global stores on MP31"};
  }
  if (IsExplicitAsyncCopy(op)) {
    return {CopyInst::kAsync, true, {}};
  }
  return {CopyInst::kNormal, true, {}};
}

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    CopyAnalysisContext ctx{layout_args.target, &layout_args.layout_map,
                            layout_args.analyzer, true};
    CopyInstSelection selection = SelectCopyInstForLowering(op, ctx);
    ICHECK(selection.supported) << selection.reason;
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    CopyAnalysisContext ctx{lower_args.target, &lower_args.layout_map, analyzer,
                            true};
    CopyInstSelection selection = SelectCopyInstForLowering(op, ctx);
    ICHECK(selection.supported) << selection.reason;
    if (CopyInstIsTMEStore(selection.inst)) {
      return LowerTmaStore(op, lower_args, analyzer);
    }
    if (CopyInstIsTME(selection.inst)) {
      return LowerTmaLoad(op, lower_args, analyzer);
    }
    if (CopyInstIsAsync(selection.inst)) {
      return LowerAsyncCopy(op, lower_args, analyzer);
    }
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

struct Im2Col {
  static Stmt Lower(const Im2ColOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    ICHECK(TargetIsMP31(lower_args.target))
        << "T.im2col is supported only on MP31 MUSA targets";
    const Buffer &src = op.src_;
    const Buffer &dst_unmapped = op.dst_;
    const BufferRegion &dst_region = op.dstRegion_;
    ICHECK(src.scope() == "global")
        << "MP31 TME im2col requires a global input buffer";
    ICHECK(dst_unmapped.scope() == "shared" ||
           dst_unmapped.scope() == "shared.dyn")
        << "MP31 TME im2col requires a shared-memory destination";
    ICHECK_EQ(src->shape.size(), 4)
        << "MP31 TME im2col requires a 4D NHWC input";
    ICHECK_GE(dst_region->region.size(), 2)
        << "MP31 TME im2col destination must have at least two dimensions";
    ICHECK_EQ(dst_region->region.size(), dst_unmapped->shape.size())
        << "MP31 TME im2col requires a complete destination region rank";
    ICHECK(src->dtype == dst_unmapped->dtype)
        << "MP31 TME im2col requires matching source/destination dtypes";
    ICHECK(TmaDescriptorSupportsDataType(src->dtype))
        << "MP31 TME im2col does not support dtype " << src->dtype;

    int64_t kernel = op.kernel_;
    int64_t stride = op.stride_;
    int64_t dilation = op.dilation_;
    int64_t padding = op.padding_;
    ICHECK_GT(kernel, 0) << "MP31 TME im2col requires kernel > 0";
    ICHECK_GT(stride, 0) << "MP31 TME im2col requires stride > 0";
    ICHECK_GT(dilation, 0) << "MP31 TME im2col requires dilation > 0";
    ICHECK_GE(padding, 0) << "MP31 TME im2col requires padding >= 0";

    const size_t dst_rank = dst_region->region.size();
    PrimExpr block_m = dst_region->region[dst_rank - 2]->extent;
    PrimExpr block_k = dst_region->region[dst_rank - 1]->extent;
    PrimExpr transaction_bytes =
        analyzer->Simplify(block_m * block_k * dst_unmapped->dtype.bytes());
    PrimExpr channel_bytes =
        analyzer->Simplify(block_k * dst_unmapped->dtype.bytes());
    ICHECK(analyzer->CanProveEqual(floormod(channel_bytes, Integer(16)),
                                   Integer(0)))
        << "MP31 TME im2col requires block_K * dtype_bytes to be 16-byte "
           "aligned, got "
        << channel_bytes << " bytes";

    PrimExpr h = src->shape[1];
    PrimExpr w = src->shape[2];
    PrimExpr c = src->shape[3];
    // One MP31 im2col transaction cannot cross a kernel-position channel
    // slice. TileLang's c_step is a tile index, so C divisible by block_k
    // guarantees every legal tile starts and ends within one such slice.
    ICHECK(analyzer->CanProveEqual(floormod(c, block_k), Integer(0)))
        << "MP31 TME im2col requires input channels to be divisible by "
           "block_K";
    auto global_stride_bytes = GetTmaGlobalStrideBytes(src, analyzer);
    ICHECK(global_stride_bytes.has_value())
        << "MP31 TME im2col requires a contiguous innermost dimension and "
           "16-byte aligned outer strides";

    PrimExpr pad_term = Integer(2 * padding - dilation * (kernel - 1) - 1);
    PrimExpr p = analyzer->Simplify(floordiv(h + pad_term, stride) + 1);
    PrimExpr q = analyzer->Simplify(floordiv(w + pad_term, stride) + 1);
    PrimExpr m_start = analyzer->Simplify(op.nhw_step_ * block_m);
    PrimExpr k_start = analyzer->Simplify(op.c_step_ * block_k);
    PrimExpr rs = analyzer->Simplify(floordiv(k_start, c));
    PrimExpr kernel_r = analyzer->Simplify(floordiv(rs, kernel));
    PrimExpr kernel_s = analyzer->Simplify(floormod(rs, kernel));
    // MP31 packs the W/S kernel position in the low byte and H/R in the next
    // byte, matching the old architecture instruction contract.
    PrimExpr weight_pos =
        analyzer->Simplify(kernel_s + kernel_r * Integer(256));

    Array<PrimExpr> box_dims = {block_k, block_m, Integer(1), Integer(1)};
    PrimExpr descriptor =
        MakeIm2ColTmaDescriptor(src, box_dims, global_stride_bytes.value());
    Buffer shared = dst_unmapped;
    if (lower_args.buffer_remap.count(shared)) {
      shared = lower_args.buffer_remap.at(shared);
    }
    if (lower_args.require_smem_alignment) {
      lower_args.require_smem_alignment(shared->data, 128);
    }

    PrimExpr shared_offset = Integer(0);
    PrimExpr shared_stride = Integer(1);
    for (size_t i = 0; i < dst_unmapped->shape.size(); ++i) {
      size_t axis = dst_unmapped->shape.size() - i - 1;
      shared_offset += dst_region->region[axis]->min * shared_stride;
      shared_stride *= dst_unmapped->shape[axis];
    }
    shared_offset = analyzer->Simplify(shared_offset);
    PrimExpr shared_ptr = shared.access_ptr(2, DataType::Handle(), 1,
                                            shared_offset, block_m * block_k);

    PrimExpr output_volume = analyzer->Simplify(p * q);
    Array<PrimExpr> block_pos = {
        analyzer->Simplify(floormod(k_start, c)),
        analyzer->Simplify(floormod(floormod(m_start, output_volume), q)),
        analyzer->Simplify(floordiv(floormod(m_start, output_volume), q)),
        analyzer->Simplify(floordiv(m_start, output_volume))};

    PrimExpr barrier;
    bool user_managed_barrier = false;
    if (auto user_barrier = op.annotations_.Get("barrier")) {
      barrier = Downcast<PrimExpr>(user_barrier.value());
      user_managed_barrier = true;
    } else {
      ICHECK(lower_args.alloc_mbarrier && lower_args.mbarrier_buffer != nullptr)
          << "MP31 TME im2col requires mbarrier allocation support";
      int barrier_slot =
          lower_args.alloc_mbarrier(1, std::string("im2col_mbarrier"));
      ICHECK(lower_args.mbarrier_buffer->defined());
      barrier = BufferLoad(lower_args.mbarrier_buffer->value(),
                           {IntImm(DataType::Int(32), barrier_slot)});
    }

    Array<PrimExpr> args = {descriptor,
                            barrier,
                            shared_ptr,
                            block_k,
                            block_m,
                            block_pos[0],
                            block_pos[1],
                            block_pos[2],
                            block_pos[3],
                            weight_pos,
                            p,
                            q,
                            Integer(padding * 257),
                            Integer(65536 + stride * 257),
                            Integer(65536 + dilation * 257)};
    Stmt load =
        Evaluate(Call(DataType::Handle(), tma_load_im2col(), std::move(args)));
    Stmt expect = Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                                {barrier, transaction_bytes}));

    Array<Stmt> producer_seq{expect, load};
    if (user_managed_barrier) {
      if (auto emit_arrive = op.annotations_.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive.value())->value != 0) {
          producer_seq.push_back(Evaluate(Call(
              DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier})));
        }
      }
    } else {
      producer_seq.push_back(Evaluate(
          Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier})));
    }

    Stmt producer = IfThenElse(Call(DataType::Bool(), tl_shuffle_elect(),
                                    {lower_args.thread_bounds->extent}),
                               SeqStmt(producer_seq));
    if (user_managed_barrier) {
      return producer;
    }
    PrimExpr phase = lower_args.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(op.annotations_)) {
      phase = explicit_phase.value();
    }
    Stmt wait = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(), {barrier, phase}));
    return SeqStmt({producer, wait});
  }
};

} // namespace musa

namespace {

bool MatchMUSACopyTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSACopy() {
  RegisterCopyImpl(CopyImpl{
      "musa.Copy",
      MatchMUSACopyTarget,
      100,
      musa::Copy::InferLayout,
      musa::Copy::Lower,
  });
  return true;
}

const bool musa_copy_registered = RegisterMUSACopy();

bool RegisterMUSAIm2Col() {
  RegisterIm2ColImpl(Im2ColImpl{
      "musa.Im2Col",
      MatchMUSACopyTarget,
      100,
      musa::Im2Col::Lower,
  });
  return true;
}

const bool musa_im2col_registered = RegisterMUSAIm2Col();

} // namespace

} // namespace tl
} // namespace tvm
