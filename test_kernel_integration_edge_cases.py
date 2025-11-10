#!/usr/bin/env python3
"""
Simplified edge case tests for KernelInterpreter and Kernel Agent
Tests the mapping functions without full initialization
"""

import sys
import tempfile
from pathlib import Path
import torch
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from kernel_interpreter import KernelExecResult
from journal import Journal, Node
from interpreter import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue


class TestKernelExecResultMapping(unittest.TestCase):
    """Test mapping KernelExecResult to ExecutionResult"""
    
    def test_map_compilation_error(self):
        """Test mapping compilation error result"""
        from kernel_interpreter import KernelInterpreter
        
        kernel_result = KernelExecResult(
            stage="compilation",
            success=False,
            error_message="SyntaxError: invalid syntax",
            compilation_error="Expected ';' at line 10",
            correctness_error=None,
            performance_metric=None,
        )
        
        # Create a minimal interpreter just to test mapping
        temp_dir = tempfile.mkdtemp()
        interpreter = KernelInterpreter(
            ref_arch_src="def forward(): pass",
            working_dir=Path(temp_dir)
        )
        
        exec_result = interpreter._map_kernel_result_to_exec_result(kernel_result)
        
        self.assertIsInstance(exec_result, ExecutionResult)
        self.assertIn("COMPILATION", exec_result.term_out[0])
        self.assertEqual(exec_result.exc_type, "CompilationError")
    
    def test_map_correctness_error(self):
        """Test mapping correctness error result"""
        from kernel_interpreter import KernelInterpreter
        
        kernel_result = KernelExecResult(
            stage="correctness",
            success=False,
            error_message="Output mismatch",
            compilation_error=None,
            correctness_error="Max absolute error: 0.5",
            performance_metric=None,
        )
        
        temp_dir = tempfile.mkdtemp()
        interpreter = KernelInterpreter(
            ref_arch_src="def forward(): pass",
            working_dir=Path(temp_dir)
        )
        
        exec_result = interpreter._map_kernel_result_to_exec_result(kernel_result)
        
        self.assertIn("CORRECTNESS", exec_result.term_out[0])
        self.assertEqual(exec_result.exc_type, "CorrectnessError")
    
    def test_map_success_result(self):
        """Test mapping successful result"""
        from kernel_interpreter import KernelInterpreter
        
        kernel_result = KernelExecResult(
            stage="performance",
            success=True,
            error_message=None,
            compilation_error=None,
            correctness_error=None,
            performance_metric=1.234,
        )
        
        temp_dir = tempfile.mkdtemp()
        interpreter = KernelInterpreter(
            ref_arch_src="def forward(): pass",
            working_dir=Path(temp_dir)
        )
        
        exec_result = interpreter._map_kernel_result_to_exec_result(kernel_result)
        
        self.assertIn("SUCCESS", exec_result.term_out[0])
        self.assertIsNone(exec_result.exc_type)
        self.assertIsNotNone(exec_result.exec_time)


class TestAgentMetricParsing(unittest.TestCase):
    """Test agent metric parsing from execution results"""
    
    def test_parse_compilation_error(self):
        """Test parsing compilation error result"""
        from kernel_agent import KernelAgent
        
        exec_result = ExecutionResult(
            term_out=[
                "COMPILATION ERROR",
                "Expected ';' at line 10",
            ],
            exec_time=None,
            exc_type="CompilationError",
            exc_info=None,
            exc_stack=None,
        )
        
        # Test the parse method directly (static method behavior)
        # Create a minimal mock agent
        mock_agent = Mock(spec=KernelAgent)
        mock_agent.ref_arch_src = "def forward(): pass"
        
        metrics = KernelAgent.parse_exec_result(mock_agent, exec_result)
        
        self.assertIsInstance(metrics["runtime"], WorstMetricValue)
        self.assertTrue(metrics["is_buggy"])
        self.assertIn("COMPILATION", metrics["analysis"])
    
    def test_parse_correctness_error(self):
        """Test parsing correctness error result"""
        from kernel_agent import KernelAgent
        
        exec_result = ExecutionResult(
            term_out=[
                "CORRECTNESS ERROR",
                "Output mismatch",
            ],
            exec_time=None,
            exc_type="CorrectnessError",
            exc_info=None,
            exc_stack=None,
        )
        
        mock_agent = Mock(spec=KernelAgent)
        mock_agent.ref_arch_src = "def forward(): pass"
        
        metrics = KernelAgent.parse_exec_result(mock_agent, exec_result)
        
        self.assertIsInstance(metrics["runtime"], WorstMetricValue)
        self.assertTrue(metrics["is_buggy"])
        self.assertIn("CORRECTNESS", metrics["analysis"])
    
    def test_parse_success_result(self):
        """Test parsing successful result"""
        from kernel_agent import KernelAgent
        
        exec_result = ExecutionResult(
            term_out=[
                "SUCCESS",
                "All correctness checks passed",
                "Runtime: 1.234 ms",
            ],
            exec_time=1.234,
            exc_type=None,
            exc_info=None,
            exc_stack=None,
        )
        
        mock_agent = Mock(spec=KernelAgent)
        mock_agent.ref_arch_src = "def forward(): pass"
        
        metrics = KernelAgent.parse_exec_result(mock_agent, exec_result)
        
        self.assertIsInstance(metrics["runtime"], MetricValue)
        self.assertEqual(metrics["runtime"].value, 1.234)
        self.assertFalse(metrics["is_buggy"])
        self.assertIn("SUCCESS", metrics["analysis"])


def run_tests(verbose=True):
    """Run all tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKernelExecResultMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentMetricParsing))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("="*80)
    print("Kernel Integration Edge Case Tests")
    print("="*80)
    print()
    
    result = run_tests(verbose=True)
    
    print()
    print("="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print()
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print()
        print("❌ Some tests failed")
        sys.exit(1)
