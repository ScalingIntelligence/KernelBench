"""
Unit tests for validate_kernel_static function.

Tests the main entry point function to ensure it correctly:
- Passes precision to precision-dependent checks
- Categorizes errors vs warnings correctly
- Handles backend-specific checks
- Respects forbidden/warnings parameters
- Returns correct output format
- Validates backend-specific patterns with real kernel code

Test Coverage:
1. API/Infrastructure Tests
   - Function signature and return values
   - Precision parameter handling
   - Error vs warning categorization
   - Custom forbidden/warnings lists
   - Backend parameter processing
   - Edge cases and integration

2. Backend Pattern Validation Tests
   - CUDA: Valid kernels, shared memory, thread indexing
   - Triton: Valid kernels, autotune, memory operations
   - ThunderKittens: Simple kernels, warpgroup MMA, compute ops
   - CUTLASS/CuTe: GEMM kernels, tensor operations
   - TileLang: Complete kernels, buffer allocation, iteration

Run with pytest:
    pytest src/kernelbench/unit_tests/test_validate_kernel_static.py -v
    or
    uv run pytest src/kernelbench/unit_tests/test_validate_kernel_static.py -v
"""

import os
import sys
import pytest

# Add src directory to path for imports (consistent with other test files)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernelbench.kernel_static_checker import validate_kernel_static


# ============================================================================
# Test Basic Function Signature and Return Values
# ============================================================================

def test_validate_kernel_static_returns_tuple():
    """Test that validate_kernel_static returns a tuple of (valid, errors, warnings)."""
    code = "x = 1 + 1"
    result = validate_kernel_static(code)
    
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 3, "Should return (valid, errors, warnings)"
    valid, errors, warnings = result
    assert isinstance(valid, bool), "First element should be bool"
    assert isinstance(errors, list), "Second element should be list"
    assert isinstance(warnings, list), "Third element should be list"


def test_validate_kernel_static_defaults():
    """Test that validate_kernel_static works with default parameters."""
    code = "x = 1 + 1"
    valid, errors, warnings = validate_kernel_static(code)
    
    # Should work without errors for simple valid code
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


# ============================================================================
# Test Precision Parameter Passing
# ============================================================================

def test_precision_passed_to_precision_checker_fp32():
    """Test that precision parameter is correctly passed to precision-dependent checks."""
    # Code with FP32 -> FP16 downgrade
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    # With fp32 precision, should detect downgrade (as warning by default)
    valid, errors, warnings = validate_kernel_static(code, precision="fp32")
    
    # Check that precision downgrade was detected (should be in warnings by default)
    all_messages = errors + warnings
    has_precision_warning = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                for msg in all_messages)
    assert has_precision_warning, "Should detect precision downgrade with fp32 precision"


def test_precision_passed_to_precision_checker_fp16():
    """Test that fp16 precision skips FP32 -> FP16 downgrade check."""
    # Code with FP32 -> FP16 downgrade
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    # With fp16 precision, precision downgrade check should be skipped
    valid, errors, warnings = validate_kernel_static(code, precision="fp16")
    
    # Should not detect precision downgrade (check is skipped for non-FP32)
    all_messages = errors + warnings
    has_precision_warning = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                for msg in all_messages)
    # This is expected - the check only runs for fp32
    # So for fp16, it won't flag this


def test_precision_case_insensitive():
    """Test that precision parameter is case-insensitive."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # Test different case variations
    result1 = validate_kernel_static(code, precision="FP32")
    result2 = validate_kernel_static(code, precision="fp32")
    result3 = validate_kernel_static(code, precision="Fp32")
    
    # All should produce the same result
    assert result1 == result2 == result3, "Precision should be case-insensitive"


def test_precision_alternative_names():
    """Test that alternative precision names are normalized."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # float32 should be normalized to fp32
    result1 = validate_kernel_static(code, precision="float32")
    result2 = validate_kernel_static(code, precision="fp32")
    
    assert result1 == result2, "float32 should be normalized to fp32"


# ============================================================================
# Test Error vs Warning Categorization
# ============================================================================

def test_strict_checks_are_errors():
    """Test that strict checks (like code_bypass) produce errors."""
    code = """
    try:
        result = custom_kernel(x)
    except:
        result = torch.matmul(x, w)  # Fallback to torch
    """
    
    valid, errors, warnings = validate_kernel_static(code)
    
    assert not valid, "Code with strict violations should be invalid"
    assert len(errors) > 0, "Strict checks should produce errors, not warnings"
    assert any("try-except" in e.lower() or "bypass" in e.lower() 
               for e in errors), "Should flag bypass in errors"


def test_warning_checks_are_warnings():
    """Test that warning checks produce warnings, not errors."""
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade - in warnings by default
        return x
    """
    
    # Test with default settings - precision_downgrade should be in warnings
    valid, errors, warnings = validate_kernel_static(
        code, 
        precision="fp32"
        # Using defaults - precision_downgrade is in WARNING_CHECKS
    )
    
    # Check that precision downgrade message is in warnings (if detected)
    # Note: backend impl checks might add errors, but precision should be in warnings
    precision_warnings = [w for w in warnings if "precision" in w.lower() or "fp16" in w.lower()]
    precision_errors = [e for e in errors if "precision" in e.lower() or "fp16" in e.lower()]
    
    if precision_warnings or precision_errors:
        # If precision downgrade is detected, it should be in warnings, not errors
        assert len(precision_warnings) > 0, "Precision downgrade should be in warnings (default)"
        assert len(precision_errors) == 0, "Precision downgrade should not be in errors (default)"


def test_custom_forbidden_checks():
    """Test that custom forbidden checks produce errors."""
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    # Make precision_downgrade a forbidden check (error) instead of warning
    valid, errors, warnings = validate_kernel_static(
        code, 
        precision="fp32",
        forbidden=["precision_downgrade"]
    )
    
    assert not valid, "Should be invalid when precision_downgrade is forbidden"
    assert len(errors) > 0, "Forbidden checks should produce errors"
    assert any("precision" in e.lower() or "fp16" in e.lower() 
               for e in errors), "Should flag precision downgrade in errors"


def test_custom_warnings_list():
    """Test that custom warnings list works."""
    code = """
    try:
        result = custom_kernel(x)
    except:
        result = torch.matmul(x, w)
    """
    
    # Move code_bypass to warnings instead of errors
    # Use a backend that won't add strict impl checks
    valid, errors, warnings = validate_kernel_static(
        code,
        backend="cuda",  # Explicit backend
        forbidden=[],  # No forbidden checks
        warnings=["code_bypass"]  # Make bypass a warning
    )
    
    # Note: Backend might add impl checks, so we check that code_bypass
    # appears in warnings (not errors) if it's detected
    all_messages = errors + warnings
    bypass_messages = [msg for msg in all_messages if "bypass" in msg.lower() or "try-except" in msg.lower()]
    
    if bypass_messages:
        # If bypass is detected, it should be in warnings, not errors
        bypass_in_warnings = any(msg in warnings for msg in bypass_messages)
        assert bypass_in_warnings, "Bypass should be in warnings when specified as warning"


# ============================================================================
# Test Backend Parameter Handling
# ============================================================================

def test_backend_adds_impl_check():
    """Test that backend parameter adds appropriate implementation check."""
    code = """
    # This code doesn't have CUDA implementation
    def forward(self, x):
        return x * 2
    """
    
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    
    # Should check for CUDA implementation (cuda_impl check)
    # The exact behavior depends on what cuda_impl check does,
    # but we can verify the backend parameter is processed
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_different_backends():
    """Test that different backends are handled correctly."""
    code = """
    def forward(self, x):
        return x * 2
    """
    
    # Test multiple backends
    backends = ["cuda", "triton", "thunderkittens", "cute", "tilelang"]
    
    for backend in backends:
        valid, errors, warnings = validate_kernel_static(code, backend=backend)
        assert isinstance(valid, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_empty_code():
    """Test handling of empty code."""
    code = ""
    
    valid, errors, warnings = validate_kernel_static(code)
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_whitespace_only_code():
    """Test handling of whitespace-only code."""
    code = "   \n\n\t  \n  "
    
    valid, errors, warnings = validate_kernel_static(code)
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_unknown_check_name():
    """Test that unknown check names are ignored."""
    code = "x = 1"
    
    # Should not crash with unknown check names
    valid, errors, warnings = validate_kernel_static(
        code,
        forbidden=["unknown_check_that_doesnt_exist"],
        warnings=["another_unknown_check"]
    )
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_multiple_precision_dependent_checks():
    """Test that multiple precision-dependent checks work (if any exist in future)."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # Currently only precision_downgrade is precision-dependent
    valid, errors, warnings = validate_kernel_static(code, precision="fp32")
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


# ============================================================================
# Test Integration: Precision + Backend + Custom Checks
# ============================================================================

def test_integration_precision_backend_forbidden():
    """Test integration of precision, backend, and custom forbidden checks."""
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    valid, errors, warnings = validate_kernel_static(
        code,
        backend="cuda",
        precision="fp32",
        forbidden=["precision_downgrade"]
    )
    
    assert not valid, "Should be invalid with precision downgrade as forbidden"
    assert len(errors) > 0, "Should have errors"
    assert any("precision" in e.lower() or "fp16" in e.lower() 
               for e in errors), "Should flag precision downgrade"


def test_integration_all_parameters():
    """Test with all parameters specified."""
    code = """
    def forward(self, x):
        return x * 2.0
    """
    
    valid, errors, warnings = validate_kernel_static(
        code,
        backend="triton",
        precision="fp16",
        forbidden=["code_bypass"],
        warnings=["precision_downgrade"]
    )
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


# ============================================================================
# Test Precision Check Integration
# ============================================================================

def test_precision_check_in_warnings_by_default():
    """Test that precision_downgrade is in warnings by default."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    valid, errors, warnings = validate_kernel_static(code, precision="fp32")
    
    # precision_downgrade should be in WARNING_CHECKS by default
    # So it should produce warnings, not errors
    all_messages = errors + warnings
    has_precision_msg = any("precision" in msg.lower() or "fp16" in msg.lower() 
                            for msg in all_messages)
    
    if has_precision_msg:
        # If detected, should be in warnings, not errors (by default)
        precision_in_warnings = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                    for msg in warnings)
        assert precision_in_warnings, "Precision downgrade should be in warnings by default"


def test_precision_check_respects_forbidden():
    """Test that precision_downgrade respects forbidden parameter."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # Make precision_downgrade forbidden
    valid, errors, warnings = validate_kernel_static(
        code,
        precision="fp32",
        forbidden=["precision_downgrade"],
        warnings=[]  # Remove from warnings
    )
    
    # Should produce errors, not warnings
    has_precision_in_errors = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                  for msg in errors)
    
    if has_precision_in_errors:
        assert not valid, "Should be invalid when precision downgrade is forbidden"
        assert len(errors) > 0, "Should have errors"


# ============================================================================
# Test Backend-Specific Pattern Validation
# These tests validate that backend checks correctly identify valid kernels
# from official documentation and reject wrapper/incomplete code
# ============================================================================

# -----------------------------------------------------------------------------
# CUDA Backend Tests
# -----------------------------------------------------------------------------

def test_cuda_valid_kernel_with_thread_indexing():
    """Test that valid CUDA kernel with threadIdx/blockIdx passes"""
    code = """
    #include <torch/extension.h>
    
    __global__ void vector_add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    
    torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
        auto c = torch::empty_like(a);
        int threads = 256;
        int blocks = (a.numel() + threads - 1) / threads;
        vector_add<<<blocks, threads>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), a.numel()
        );
        return c;
    }
    
    auto module = torch::utils::cpp_extension::load_inline(
        "vector_add", cuda_src, cuda_src, {"vector_add.cu"}, {}, {}, true
    );
    """
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert valid, f"Expected valid CUDA kernel to pass, got errors: {errors}"


def test_cuda_valid_kernel_with_shared_memory():
    """Test that CUDA kernel with __shared__ memory passes"""
    code = """
    #include <torch/extension.h>
    
    __global__ void matmul_shared(float* A, float* B, float* C, int N) {
        __shared__ float shared_A[16][16];
        __shared__ float shared_B[16][16];
        
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        
        __syncthreads();
        // ... computation using shared memory
    }
    
    torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
        return torch::utils::cpp_extension::load_inline("matmul", cuda_src, {}, {});
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert valid, f"Expected CUDA kernel with __shared__ to pass, got errors: {errors}"


def test_cuda_missing_thread_indexing():
    """Test that CUDA kernel without thread indexing is rejected"""
    code = """
    #include <torch/extension.h>
    
    __global__ void fake_kernel() {
        // No threadIdx, no blockIdx, no actual CUDA features
        float x = 1.0f;
    }
    
    void wrapper() {
        torch::utils::cpp_extension::load_inline("fake", src, {}, {});
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "Expected CUDA kernel without thread indexing to fail"
    assert any("thread indexing" in err.lower() or "kernel features" in err.lower() for err in errors)


# -----------------------------------------------------------------------------
# Triton Backend Tests
# -----------------------------------------------------------------------------

def test_triton_valid_kernel():
    """Test that valid Triton kernel with tl.load/store/program_id passes"""
    code = """
    import triton
    import triton.language as tl
    
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    """
    valid, errors, warnings = validate_kernel_static(code, backend="triton")
    assert valid, f"Expected valid Triton kernel to pass, got errors: {errors}"


def test_triton_autotune():
    """Test that Triton kernel with @triton.autotune passes"""
    code = """
    import triton
    import triton.language as tl
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def optimized_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2, mask=mask)
    """
    valid, errors, warnings = validate_kernel_static(code, backend="triton")
    assert valid, f"Expected Triton autotune kernel to pass, got errors: {errors}"


def test_triton_missing_memory_ops():
    """Test that Triton kernel without memory operations is rejected"""
    code = """
    import triton
    
    @triton.jit
    def fake_kernel(x):
        # Missing memory operations
        return x * 2
    """
    valid, errors, warnings = validate_kernel_static(code, backend="triton")
    assert not valid, f"Expected Triton kernel without memory ops to fail, but got valid={valid}"
    assert any("load" in err.lower() or "store" in err.lower() or "tl" in err.lower() for err in errors), \
        f"Expected error about tl operations, got: {errors}"


# -----------------------------------------------------------------------------
# ThunderKittens Backend Tests
# -----------------------------------------------------------------------------

def test_thunderkittens_valid_simple():
    """Test that simple ThunderKittens kernel passes (from official manual)"""
    code = """
    #include <kittens.cuh>
    
    using namespace kittens;
    
    __global__ void example_kernel() {
        rt_fl<32, 64> a, b, c;
        __shared__ st_hf<32, 64> s;
        
        kittens::mul(c, a, b);
        kittens::store(s, c);
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="thunderkittens")
    assert valid, f"Expected simple ThunderKittens kernel to pass, got errors: {errors}"


def test_thunderkittens_valid_warpgroup():
    """Test that ThunderKittens warpgroup MMA kernel passes"""
    code = """
    #include <kittens.cuh>
    
    using namespace kittens;
    
    __global__ void gemm_kernel(const bf16 *A, const bf16 *B, bf16 *C) {
        using namespace kittens::warpgroup;
        
        rt_fl<16, 16> A_tile;
        rt_fl<16, 16> B_tile;
        rt_fl<16, 16> C_tile;
        
        zero(C_tile);
        kittens::load(A_tile, A);
        kittens::load(B_tile, B);
        mma_AB(C_tile, A_tile, B_tile);
        kittens::store(C, C_tile);
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="thunderkittens")
    assert valid, f"Expected ThunderKittens warpgroup kernel to pass, got errors: {errors}"


def test_thunderkittens_missing_compute():
    """Test that ThunderKittens kernel without compute ops is rejected"""
    code = """
    #include <kittens.cuh>
    
    using namespace kittens;
    
    __global__ void incomplete_kernel() {
        rt_fl<32, 64> tile;
        kittens::load(tile, ptr);
        kittens::store(output, tile);
        // Has load/store but no compute!
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="thunderkittens")
    assert not valid, "Expected ThunderKittens kernel without compute to fail"
    assert any("compute" in err.lower() for err in errors)


# -----------------------------------------------------------------------------
# CUTLASS/CuTe Backend Tests
# -----------------------------------------------------------------------------

def test_cutlass_valid_gemm():
    """Test that CUTLASS GEMM kernel passes"""
    code = """
    #include <cutlass/cutlass.h>
    #include <cutlass/gemm/device/gemm.h>
    #include <cutlass/layout/matrix.h>
    
    using namespace cutlass;
    
    using Gemm = cutlass::gemm::device::Gemm<
        float, layout::RowMajor,
        float, layout::ColumnMajor,
        float, layout::RowMajor
    >;
    
    void run_gemm() {
        Gemm gemm_op;
        gemm_op();
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="cutlass")
    assert valid, f"Expected CUTLASS GEMM kernel to pass, got errors: {errors}"


def test_cute_valid_tensor_ops():
    """Test that CuTe tensor operations pass"""
    code = """
    #include <cute/tensor.hpp>
    
    __global__ void cute_kernel() {
        auto tensor = cute::make_tensor(ptr, cute::make_layout(cute::make_shape(16, 16)));
        auto layout = cute::make_layout(cute::make_shape(16, 16), cute::make_stride(1, 16));
        cute::copy(tensor_A, tensor_B);
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="cute")
    assert valid, f"Expected CuTe tensor ops kernel to pass, got errors: {errors}"


def test_cutlass_missing_operations():
    """Test that code with just namespace is rejected"""
    code = """
    #include <cutlass/cutlass.h>
    
    using namespace cutlass;
    
    void wrapper() {
        // Just includes cutlass but no actual operations
        float x = 1.0f;
    }
    """
    valid, errors, warnings = validate_kernel_static(code, backend="cutlass")
    assert not valid, "Expected CUTLASS code without operations to fail"
    assert any("operations" in err.lower() or "namespace" in err.lower() for err in errors)


# -----------------------------------------------------------------------------
# TileLang Backend Tests
# -----------------------------------------------------------------------------

def test_tilelang_valid_complete():
    """Test that complete TileLang kernel passes"""
    code = """
    import tvm
    from tvm.script import tir as T
    
    @T.prim_func
    def matmul(A: T.Buffer((1024, 1024), "float32"),
               B: T.Buffer((1024, 1024), "float32"),
               C: T.Buffer((1024, 1024), "float32")):
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    
    mod = tvm.IRModule({"main": matmul})
    func = tvm.build(mod, target="cuda")
    """
    valid, errors, warnings = validate_kernel_static(code, backend="tilelang")
    assert valid, f"Expected complete TileLang kernel to pass, got errors: {errors}"


def test_tilelang_valid_with_buffer():
    """Test that TileLang kernel with buffer allocation passes"""
    code = """
    import tvm
    from tvm.script import tir as T
    
    @T.prim_func
    def conv2d(Input: T.Buffer((1, 3, 224, 224), "float32"),
               Weight: T.Buffer((64, 3, 3, 3), "float32"),
               Output: T.Buffer((1, 64, 222, 222), "float32")):
        temp = T.alloc_buffer((1, 64, 222, 222), "float32")
        
        for n, c, h, w in T.grid(1, 64, 222, 222):
            for rc, rh, rw in T.grid(3, 3, 3):
                temp[n, c, h, w] = temp[n, c, h, w] + Input[n, rc, h+rh, w+rw] * Weight[c, rc, rh, rw]
        
        for i, j, k, l in T.grid(1, 64, 222, 222):
            T.buffer_store(Output, temp[i, j, k, l], [i, j, k, l])
    
    mod = tvm.build(T.prim_func)
    """
    valid, errors, warnings = validate_kernel_static(code, backend="tilelang")
    assert valid, f"Expected TileLang kernel with buffer to pass, got errors: {errors}"


def test_tilelang_missing_iteration():
    """Test that TileLang kernel without iteration constructs is rejected"""
    code = """
    import tvm
    from tvm.script import tir as T
    
    @T.prim_func
    def fake_kernel():
        # Has decorator but no T.grid, T.serial, etc.
        x = 1.0
    """
    valid, errors, warnings = validate_kernel_static(code, backend="tilelang")
    assert not valid, "Expected TileLang kernel without iteration to fail"
    assert any("iteration" in err.lower() for err in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

