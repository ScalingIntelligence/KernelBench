#!/usr/bin/env python3
"""
Quick test script to verify the AIDE + KernelBench integration
Tests basic functionality without running a full search
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from kernel_interpreter import KernelInterpreter
        from kernel_agent import KernelAgent
        from journal import Journal
        from backend import query, compile_prompt_to_md
        from src.eval import eval_kernel_against_ref
        from src.utils import read_file
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_kernel_interpreter():
    """Test that KernelInterpreter can be instantiated."""
    print("\nTesting KernelInterpreter...")
    try:
        from kernel_interpreter import KernelInterpreter
        
        # Simple reference architecture
        ref_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
    
    def forward(self, x):
        return x + 1.0

def get_inputs():
    return [torch.randn(10, 10).cuda()]

def get_init_inputs():
    return [10]
"""
        
        interpreter = KernelInterpreter(
            ref_arch_src=ref_code,
            working_dir="./test_workspace",
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=10,
            measure_performance=False,
            verbose=False,
        )
        
        print(f"✓ KernelInterpreter created successfully")
        print(f"  Device: {interpreter.device}")
        print(f"  Backend: {interpreter.backend}")
        return True
    except Exception as e:
        print(f"✗ KernelInterpreter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_agent():
    """Test that KernelAgent can be instantiated."""
    print("\nTesting KernelAgent...")
    try:
        from kernel_agent import KernelAgent
        from journal import Journal
        from omegaconf import OmegaConf
        
        # Create minimal config
        cfg = OmegaConf.create({
            "agent": {
                "steps": 10,
                "code": {"model": "test", "temp": 0.7},
                "feedback": {"model": "test", "temp": 0.3},
                "search": {
                    "num_drafts": 2,
                    "debug_prob": 0.3,
                    "max_debug_depth": 2,
                },
                "k_fold_validation": 1,
                "expose_prediction": False,
                "data_preview": False,
            },
            "log_dir": "./test_logs",
            "workspace_dir": "./test_workspace",
        })
        
        ref_code = "# Simple reference"
        task_desc = "Test task"
        journal = Journal()
        
        agent = KernelAgent(
            ref_arch_src=ref_code,
            task_desc=task_desc,
            cfg=cfg,
            journal=journal,
            backend="cuda",
            precision="fp32",
        )
        
        print(f"✓ KernelAgent created successfully")
        print(f"  Backend: {agent.backend}")
        print(f"  Precision: {agent.precision}")
        return True
    except Exception as e:
        print(f"✗ KernelAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend():
    """Test the backend query interface."""
    print("\nTesting backend...")
    try:
        from backend import compile_prompt_to_md
        
        # Test prompt compilation
        prompt = {
            "Section 1": "Content 1",
            "Section 2": ["Item 1", "Item 2"],
        }
        md = compile_prompt_to_md(prompt)
        
        assert "## Section 1" in md
        assert "## Section 2" in md
        assert "Item 1" in md
        
        print("✓ Backend compile_prompt_to_md works")
        return True
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from omegaconf import OmegaConf
        
        config_path = Path(__file__).parent / "kernel_config.yaml"
        if not config_path.exists():
            print(f"⚠️  Config file not found at {config_path}")
            return False
        
        cfg = OmegaConf.load(config_path)
        
        # Check required sections
        assert "kernel" in cfg
        assert "gpu" in cfg
        assert "agent" in cfg
        assert "evaluation" in cfg
        
        print("✓ Configuration loads successfully")
        print(f"  Default level: {cfg.kernel.level}")
        print(f"  Default backend: {cfg.gpu.backend}")
        print(f"  Default steps: {cfg.agent.steps}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def main():
    print("="*80)
    print("AIDE + KernelBench Integration Test Suite")
    print("="*80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("\n⚠️  CUDA not available - some tests may be limited")
    
    # Run tests
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Backend", test_backend()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("KernelInterpreter", test_kernel_interpreter()))
    results.append(("KernelAgent", test_kernel_agent()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print("="*80)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is ready to use.")
        print("\nTo run a kernel search, use:")
        print("  python run_kernel_search_simple.py --level 1 --problem_id 1 --steps 5")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
