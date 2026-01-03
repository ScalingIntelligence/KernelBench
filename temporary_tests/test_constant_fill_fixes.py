"""
Test that constant-fill problems produce constant outputs (OLD)
and varying outputs after fix.

Run with: pytest tests/test_constant_fill_fixes.py -v
Or directly: python tests/test_constant_fill_fixes.py
"""
import os
import sys
import importlib.util
import torch

KERNEL_BENCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../KernelBench"))


def load_model_from_file(filepath):
    """Load Model class and input functions from a KernelBench file."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model, module.get_inputs, module.get_init_inputs


def check_constant_vs_varying(old_path, new_path, atol=1e-5):
    """
    Verify OLD model produces constant output, NEW model produces varying output.
    Returns (old_is_constant, new_varies) booleans.
    """
    OldModel, get_inputs, get_init_inputs = load_model_from_file(old_path)
    NewModel, _, _ = load_model_from_file(new_path)
    
    torch.manual_seed(42)
    init_inputs = get_init_inputs()
    
    old_model = OldModel(*init_inputs).eval()
    new_model = NewModel(*init_inputs).eval()
    
    with torch.no_grad():
        torch.manual_seed(1)
        x1 = get_inputs()[0]
        torch.manual_seed(2)
        x2 = get_inputs()[0]
        
        old_out1, old_out2 = old_model(x1), old_model(x2)
        new_out1, new_out2 = new_model(x1), new_model(x2)
    
    # OLD should be constant (approximately zero or same for different inputs)
    old_is_constant = torch.allclose(old_out1, old_out2, atol=atol)
    
    # NEW should vary with input
    new_varies = not torch.allclose(new_out1, new_out2, atol=atol)
    
    return old_is_constant, new_varies


def test_80_gemm_max_subtract_gelu():
    """mean(dim=1) on (B,1) → value itself → x - mean = 0"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/80_Gemm_Max_Subtract_GELU_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/80_Gemm_Max_Subtract_GELU.py")
    
    old_const, new_varies = check_constant_vs_varying(old_path, new_path)
    assert old_const, "OLD should produce constant output"
    assert new_varies, "NEW should produce varying output"
    print("✓ test_80_gemm_max_subtract_gelu passed")


def test_83_conv3d_groupnorm_min_clamp_dropout():
    """min(x,0) + clamp(min=0) → all zeros"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/83_Conv3d_GroupNorm_Min_Clamp_Dropout.py")
    
    old_const, new_varies = check_constant_vs_varying(old_path, new_path)
    assert old_const, "OLD should produce constant output"
    assert new_varies, "NEW should produce varying output"
    print("✓ test_83_conv3d_groupnorm_min_clamp_dropout passed")


def test_23_conv3d_groupnorm_mean():
    """GroupNorm zero-mean → global mean ≈ 0"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/23_Conv3d_GroupNorm_Mean_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/23_Conv3d_GroupNorm_Mean.py")
    
    old_const, new_varies = check_constant_vs_varying(old_path, new_path)
    assert old_const, "OLD should produce constant output"
    assert new_varies, "NEW should produce varying output"
    print("✓ test_23_conv3d_groupnorm_mean passed")


if __name__ == "__main__":
    test_80_gemm_max_subtract_gelu()
    test_83_conv3d_groupnorm_min_clamp_dropout()
    test_23_conv3d_groupnorm_mean()
    print("\nAll tests passed!")

