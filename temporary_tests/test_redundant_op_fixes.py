"""
Test that removing redundant operations produces equivalent outputs.

Run with: pytest tests/test_redundant_op_fixes.py -v
Or directly: python tests/test_redundant_op_fixes.py
"""
import os
import importlib.util
import torch

KERNEL_BENCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../KernelBench"))


def load_model_from_file(filepath):
    """Load Model class and input functions from a KernelBench file."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model, module.get_inputs, module.get_init_inputs


def check_equivalence(old_path, new_path, atol=1e-5):
    """
    Verify OLD and NEW models produce equivalent outputs.
    Returns True if outputs match within tolerance.
    """
    OldModel, get_inputs, get_init_inputs = load_model_from_file(old_path)
    NewModel, _, _ = load_model_from_file(new_path)
    
    torch.manual_seed(42)
    init_inputs = get_init_inputs()
    
    old_model = OldModel(*init_inputs).eval()
    new_model = NewModel(*init_inputs).eval()
    
    # Copy weights from old to new (they may have different params due to removed layers)
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    # Only copy matching keys
    for key in new_state:
        if key in old_state:
            new_state[key] = old_state[key]
    new_model.load_state_dict(new_state)
    
    with torch.no_grad():
        torch.manual_seed(123)
        inputs = get_inputs()
        
        old_out = old_model(*inputs)
        new_out = new_model(*inputs)
    
    return torch.allclose(old_out, new_out, atol=atol)


def test_44_double_global_avg_pool():
    """Second global avg pool is no-op (tensor already 1x1 after first)"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_44_double_global_avg_pool passed")


def test_95_hardtanh_after_tanh_gelu():
    """Hardtanh redundant: tanh→GELU output is already in [-1,1]"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/95_Matmul_Add_Swish_Tanh_GELU_Hardtanh_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_95_hardtanh_after_tanh_gelu passed")


def test_81_clamp_after_tanh():
    """Clamp [-1,1] after tanh is redundant (tanh already outputs [-1,1])"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_81_clamp_after_tanh passed")


def test_7_leakyrelu_after_relu():
    """LeakyReLU after ReLU is identity (all values already ≥0)"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level2/7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_7_leakyrelu_after_relu passed")


def test_36_lstm_hn_dead_fc():
    """fc layer is dead code (computes but returns h_n instead)"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level3/36_LSTMHn_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level3/36_LSTMHn.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_36_lstm_hn_dead_fc passed")


def test_37_lstm_cn_dead_fc():
    """fc layer is dead code (computes but returns c_n instead)"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level3/37_LSTMCn_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level3/37_LSTMCn.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_37_lstm_cn_dead_fc passed")


def test_49_mamba2_dead_y_diag():
    """Y_diag and L are computed but never used in return value"""
    old_path = os.path.join(KERNEL_BENCH_PATH, "level3/49_Mamba2ReturnFinalState_OLD.py")
    new_path = os.path.join(KERNEL_BENCH_PATH, "level3/49_Mamba2ReturnFinalState.py")
    
    assert check_equivalence(old_path, new_path), "Outputs should be equivalent"
    print("✓ test_49_mamba2_dead_y_diag passed")


if __name__ == "__main__":
    test_44_double_global_avg_pool()
    test_95_hardtanh_after_tanh_gelu()
    test_81_clamp_after_tanh()
    test_7_leakyrelu_after_relu()
    test_36_lstm_hn_dead_fc()
    test_37_lstm_cn_dead_fc()
    test_49_mamba2_dead_y_diag()
    print("\nAll equivalence tests passed!")

