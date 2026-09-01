/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_musa.h
 * \brief Utility to generate musa code
 */
#ifndef TVM_TL_MUSA_CODEGEN_CODEGEN_MUSA_H_
#define TVM_TL_MUSA_CODEGEN_CODEGEN_MUSA_H_

#include <tvm/target/codegen.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenMUSA final : public CodeGenC {
 public:
  CodeGenMUSA();
  void Init(bool output_ssa);
  std::string Finish();
  bool need_include_path() {
    return (enable_fp16_ || enable_bf16_ || enable_int8_ || enable_fp8_ || enable_fp6_ ||
            enable_fp4_ || need_math_constants_h_ || need_math_h_ || need_mma_h_ ||
            need_atomic_h_ || need_debug_h_ || need_reduce_h_ || need_scan_h_ || need_copy_h_ ||
            need_cvt_h_ || need_threadblock_swizzle_h_ || need_dp4a_h_ ||
            need_mp31_tme_h_ || need_mp31_sqmma_h_);
  }
  // override behavior
  void PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                              std::ostream& os) final;
  void PrintExtraAttrs(const PrimFunc& f, std::ostream& os) final;  // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const EvaluateNode* op) final;
  void PrintStorageSync(const CallNode* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                        std::ostream& os) final;       // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintVecConstructor(DataType t, std::ostream& os) final;
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, DataType t, int i, const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) final;
  std::string CastFromTo(std::string value, DataType from, DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;
  void VisitExpr_(const CallNode* op, std::ostream& os) final;
  void VisitExpr_(const CastNode* op, std::ostream& os) final;
  void VisitStmt_(const AllocBufferNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

 protected:
  void PrintCallExtern(Type ret_type, ffi::String global_symbol, const ffi::Array<PrimExpr>& args,
                       bool skip_first_arg, std::ostream& os) final;  // NOLINT(*)

 private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                           std::ostream& os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable fp6
  bool enable_fp6_{false};
  // whether enable fp4
  bool enable_fp4_{false};
  bool need_cvt_h_{false};
  bool need_threadblock_swizzle_h_{false};
  bool need_dp4a_h_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need tl MUSA math helpers
  bool need_math_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need tl atomic helpers
  bool need_atomic_h_{false};
  // whether need tl debug helpers
  bool need_debug_h_{false};
  // whether need tl reduce helpers
  bool need_reduce_h_{false};
  // whether need tl scan helpers
  bool need_scan_h_{false};
  // whether need tl async-copy helpers
  bool need_copy_h_{false};
  // Whether an MP31 TME load helper is referenced by the generated kernel.
  bool need_mp31_tme_h_{false};
  // Whether an MP31 SQMMA helper is referenced by the generated kernel.
  bool need_mp31_sqmma_h_{false};
  // whether Common MUSA fast-divmod helpers are required
  bool need_fast_divmod_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ = Op::GetAttrMap<bool>("musa.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode*, std::string> fragment_shapes;
  std::unordered_map<const VarNode*, std::string> fragment_layouts;
  friend void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenMUSA* p);
  void PrintWmmaScope(const std::string& scope, DataType t, const VarNode* variable,
                      std::ostream& os);
  int32_t GetWmmaFragmentSize(const std::string& scope, const VarNode* variable, int32_t size);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TL_MUSA_CODEGEN_CODEGEN_MUSA_H_
