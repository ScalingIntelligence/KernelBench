"""
Tests for kernel_static_checker.py
[WIP] will use more realisitic adverserial kernels
Run with: pytest src/unit_tests/test_static_checker.py -v
"""

import pytest
from src.kernel_static_checker import (
    validate_kernel_static,
    check_code_bypass,
    check_pytorch_wrap,
    check_torch_computation_ops,
    check_cuda_impl,
    check_triton_impl,
    check_tk_impl,
    check_cute_impl,
    check_tilelang_impl,
)


# =============================================================================
# Test Code Bypass Detection (try-except + pass)
# =============================================================================

def test_bypass_try_except():
    code = "try:\n    result = kernel(x)\nexcept:\n    result = torch.matmul(x, w)"
    has_issue, msg = check_code_bypass(code)
    assert has_issue, "Should detect try-except"
    assert "try-except" in msg


def test_bypass_pass_statement():
    code = "class MyKernel:\n    def forward(self, x):\n        pass"
    has_issue, msg = check_code_bypass(code)
    assert has_issue, "Should detect pass statement"
    assert "pass" in msg


def test_bypass_pass_in_word():
    code = "# This check has passed all tests"
    has_issue, msg = check_code_bypass(code)
    assert not has_issue, "Should not match 'passed'"


def test_no_bypass():
    code = "result = kernel(x)"
    has_issue, msg = check_code_bypass(code)
    assert not has_issue, "Clean code should pass"


# =============================================================================
# Test PyTorch Wrapping Detection
# =============================================================================

def test_pytorch_wrap_functional():
    code = "import torch.nn.functional as F\nresult = F.relu(x)"
    has_issue, msg = check_pytorch_wrap(code)
    assert has_issue, "Should detect F.relu"


def test_pytorch_wrap_nn_functional():
    code = "result = torch.nn.functional.conv2d(x, w)"
    has_issue, msg = check_pytorch_wrap(code)
    assert has_issue, "Should detect torch.nn.functional"


def test_pytorch_wrap_allows_module():
    code = "class Model(torch.nn.Module):\n    def forward(self, x): return x"
    has_issue, msg = check_pytorch_wrap(code)
    assert not has_issue, "Should allow torch.nn.Module"


def test_pytorch_wrap_allows_parameter():
    code = "self.weight = torch.nn.Parameter(torch.randn(10))"
    has_issue, msg = check_pytorch_wrap(code)
    assert not has_issue, "Should allow torch.nn.Parameter"


# =============================================================================
# Test Torch Computation Ops Detection
# =============================================================================

def test_torch_ops_conv2d():
    code = "result = torch.conv2d(x, w)"
    has_issue, msg = check_torch_computation_ops(code)
    assert has_issue, "Should detect torch.conv2d"


def test_torch_ops_matmul():
    code = "result = torch.matmul(x, w)"
    has_issue, msg = check_torch_computation_ops(code)
    assert has_issue, "Should detect torch.matmul"


def test_torch_ops_allowed():
    code = "result = torch.zeros(10)"  # tensor creation is fine
    has_issue, msg = check_torch_computation_ops(code)
    assert not has_issue, "torch.zeros should be allowed"


# =============================================================================
# Test CUDA Implementation Check
# =============================================================================

def test_cuda_valid():
    code = '''
cuda_src = """
__global__ void my_kernel(float* out) {
    out[0] = 1.0f;
}
"""
module = load_inline(name="my_kernel", cuda_sources=[cuda_src])
'''
    has_issue, msg = check_cuda_impl(code)
    assert not has_issue, "Valid CUDA should pass"


def test_cuda_missing_global():
    code = "module = load_inline(name='test', cuda_sources=[src])"
    has_issue, msg = check_cuda_impl(code)
    assert has_issue, "Should detect missing __global__"


def test_cuda_missing_load_inline():
    code = "__global__ void kernel() {}"
    has_issue, msg = check_cuda_impl(code)
    assert has_issue, "Should detect missing load_inline"


def test_cuda_cpp_extension():
    code = "__global__ void kernel() {}\nfrom torch.utils.cpp_extension import load"
    has_issue, msg = check_cuda_impl(code)
    assert not has_issue, "cpp_extension should also be valid"


# =============================================================================
# Test Triton Implementation Check
# =============================================================================

def test_triton_valid():
    code = '''
@triton.jit
def my_kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
'''
    has_issue, msg = check_triton_impl(code)
    assert not has_issue, "Valid Triton should pass"


def test_triton_missing_jit():
    code = "def my_kernel(x_ptr):\n    x = tl.load(x_ptr)"
    has_issue, msg = check_triton_impl(code)
    assert has_issue, "Should detect missing @triton.jit"


def test_triton_missing_tl_ops():
    code = "@triton.jit\ndef my_kernel():\n    return 1"
    has_issue, msg = check_triton_impl(code)
    assert has_issue, "Should detect missing tl.* ops"


# =============================================================================
# Test ThunderKittens Implementation Check
# =============================================================================

def test_tk_valid():
    code = '''
using namespace kittens;
__global__ void my_kernel() {
    warpgroup::sync();
    rt_bf<16, 16> reg_tile;
    st_bf<16, 16> shared_tile;
}
'''
    has_issue, msg = check_tk_impl(code)
    assert not has_issue, "Valid TK should pass"


def test_tk_missing_warp():
    code = "rt_bf<16, 16> tile;"
    has_issue, msg = check_tk_impl(code)
    assert has_issue, "Should detect missing warp patterns"


def test_tk_missing_tiles():
    code = "using namespace kittens;\nwarpgroup::sync();"
    has_issue, msg = check_tk_impl(code)
    assert has_issue, "Should detect missing tile declarations"


# =============================================================================
# Test CuTe/CUTLASS Implementation Check
# =============================================================================

def test_cute_valid():
    code = "cute::copy(src, dst);"
    has_issue, msg = check_cute_impl(code)
    assert not has_issue, "Valid CuTe should pass"


def test_cute_cutlass_namespace():
    code = "cutlass::gemm::GemmCoord coord;"
    has_issue, msg = check_cute_impl(code)
    assert not has_issue, "CUTLASS namespace should pass"


def test_cute_missing():
    code = "__global__ void kernel() {}"
    has_issue, msg = check_cute_impl(code)
    assert has_issue, "Should detect missing cute::"


# =============================================================================
# Test TileLang Implementation Check
# =============================================================================

def test_tilelang_valid():
    code = "@T.prim_func\ndef kernel(): pass"
    has_issue, msg = check_tilelang_impl(code)
    assert not has_issue, "Valid TileLang should pass"


def test_tilelang_missing():
    code = "def kernel(): pass"
    has_issue, msg = check_tilelang_impl(code)
    assert has_issue, "Should detect missing @T.prim_func"


# =============================================================================
# Test validate_kernel_static Integration
# =============================================================================

def test_validate_cuda_valid():
    code = '''
cuda_src = """
__global__ void kernel(float* out) { out[0] = 1.0f; }
"""
module = load_inline(name="test", cuda_sources=[cuda_src])

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return module.kernel(x)
'''
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert valid, f"Valid CUDA should pass: {errors}"


def test_validate_with_bypass():
    code = "try:\n    x = kernel()\nexcept:\n    x = torch.matmul(a, b)"
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    assert not valid, "Should detect bypass"
    assert len(errors) > 0


def test_validate_custom_checks():
    code = "torch.matmul(x, w)"  # torch op, no bypass
    
    # Default: torch_computation_ops is in warnings
    valid, errors, warnings = validate_kernel_static(
        code, backend="cuda", 
        forbidden=["code_bypass"],  # only check bypass
        warnings=["torch_computation_ops"]
    )
    # This will fail on cuda_impl (no __global__), but let's check warnings work
    assert len(warnings) > 0, "Should have torch op warning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
