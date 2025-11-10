"""
Kernel Interpreter for AIDE Integration
Wraps KernelBench evaluation pipeline to be compatible with AIDE's Interpreter interface
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch

# Use absolute imports for compatibility
import sys
sys.path.insert(0, str(Path(__file__).parent))

from interpreter import ExecutionResult
from src.eval import eval_kernel_against_ref, KernelExecResult, get_torch_dtype_from_string

logger = logging.getLogger("aide")


class KernelInterpreter:
    """
    Interpreter for CUDA kernel evaluation that wraps KernelBench's evaluation pipeline.
    Compatible with AIDE's Interpreter interface.
    """
    
    def __init__(
        self,
        ref_arch_src: str,
        working_dir: Path | str,
        device: torch.device | int = None,
        backend: str = "cuda",
        precision: str = "fp32",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        measure_performance: bool = True,
        timeout: int = 3600,
        verbose: bool = False,
    ):
        """
        Initialize kernel interpreter for CUDA kernel evaluation.
        
        Args:
            ref_arch_src: Reference PyTorch architecture source code
            working_dir: Working directory for compilation artifacts
            device: GPU device to use for evaluation
            backend: Backend type ('cuda', 'triton', 'tilelang', 'cute')
            precision: Precision for computation ('fp32', 'fp16', 'bf16')
            num_correct_trials: Number of correctness trials with different random inputs
            num_perf_trials: Number of performance measurement trials
            measure_performance: Whether to measure performance (only if correct)
            timeout: Timeout for evaluation (not currently enforced in eval)
            verbose: Whether to print verbose output
        """
        self.ref_arch_src = ref_arch_src
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU setup
        if device is None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = device
            
        self.backend = backend
        self.precision_str = precision
        self.precision = get_torch_dtype_from_string(precision)
        
        # Evaluation parameters
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.measure_performance = measure_performance
        self.timeout = timeout
        self.verbose = verbose
        
        # Build directory for CUDA compilation artifacts
        self.build_dir = self.working_dir / "cuda_build"
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Kernel Interpreter initialized with device: {self.device}, backend: {self.backend}")
    
    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """
        Execute CUDA kernel code and evaluate against reference architecture.
        
        Args:
            code: Generated CUDA kernel code (should define ModelNew class)
            reset_session: Compatibility parameter (ignored, each eval is independent)
            
        Returns:
            ExecutionResult with evaluation results mapped to AIDE format
        """
        logger.debug(f"Kernel Interpreter evaluating code (backend={self.backend})")
        
        if self.verbose:
            print(f"[KernelInterpreter] Starting evaluation")
            print(f"[KernelInterpreter] Backend: {self.backend}, Precision: {self.precision_str}")
        
        # Run evaluation through KernelBench pipeline
        try:
            kernel_result = eval_kernel_against_ref(
                original_model_src=self.ref_arch_src,
                custom_model_src=code,
                seed_num=42,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                verbose=self.verbose,
                measure_performance=self.measure_performance,
                build_dir=str(self.build_dir),
                device=self.device,
                backend=self.backend,
                precision=self.precision,
            )
        except Exception as e:
            logger.error(f"Unexpected error during kernel evaluation: {e}")
            return ExecutionResult(
                term_out=[f"Unexpected evaluation error: {str(e)}"],
                exec_time=0.0,
                exc_type="EvaluationError",
                exc_info={"error": str(e)},
                exc_stack=None,
            )
        
        # Handle case where eval returns None (e.g., lock file error - should retry)
        if kernel_result is None:
            logger.warning("Kernel evaluation returned None (likely lock file error)")
            return ExecutionResult(
                term_out=["Evaluation returned None - likely concurrent compilation issue. Please retry."],
                exec_time=0.0,
                exc_type="LockFileError",
                exc_info={"error": "Concurrent compilation lock file error"},
                exc_stack=None,
            )
        
        # Map KernelExecResult to ExecutionResult
        return self._map_kernel_result_to_exec_result(kernel_result)
    
    def _map_kernel_result_to_exec_result(self, kernel_result: KernelExecResult) -> ExecutionResult:
        """
        Map KernelBench's KernelExecResult to AIDE's ExecutionResult format.
        
        Stages of evaluation:
        1. Compilation: Did the CUDA code compile?
        2. Correctness: Does output match reference?
        3. Performance: What is the runtime? (only measured if correct)
        """
        term_out = []
        exec_time = 0.0
        exc_type = None
        exc_info = None
        exc_stack = None
        
        # Convert metadata to be JSON serializable (convert exception objects to strings)
        serializable_metadata = {}
        if kernel_result.metadata:
            for key, value in kernel_result.metadata.items():
                if isinstance(value, Exception):
                    serializable_metadata[key] = str(value)
                elif isinstance(value, dict):
                    # Recursively handle nested dicts
                    serializable_metadata[key] = {k: str(v) if isinstance(v, Exception) else v for k, v in value.items()}
                else:
                    serializable_metadata[key] = value
        
        # Stage 1: Compilation
        if not kernel_result.compiled:
            exc_type = "CompilationError"
            compilation_error = serializable_metadata.get('compilation_error', 'Unknown compilation error')
            compilation_error_name = serializable_metadata.get('compilation_error_name', 'CompilationError')
            compilation_traceback = serializable_metadata.get('compilation_traceback', '')
            
            term_out.append("=" * 80)
            term_out.append("COMPILATION FAILED")
            term_out.append("=" * 80)
            term_out.append(f"Error: {compilation_error_name}")
            term_out.append(f"\n{str(compilation_error)}")
            
            # Add traceback if available for better debugging
            if compilation_traceback:
                term_out.append("\n" + "=" * 80)
                term_out.append("FULL COMPILATION TRACEBACK:")
                term_out.append("=" * 80)
                term_out.append(compilation_traceback)
            else:
                term_out.append("\nThe generated CUDA kernel failed to compile.")
                term_out.append("Please check the kernel implementation for syntax errors,")
                term_out.append("incorrect PyTorch CUDA extension API usage, or compilation issues.")
            
            exc_info = {
                'compilation_error': str(compilation_error),
                'compilation_error_name': compilation_error_name,
                'compilation_traceback': compilation_traceback,
                'metadata': serializable_metadata,
            }
            
            return ExecutionResult(
                term_out=term_out,
                exec_time=exec_time,
                exc_type=exc_type,
                exc_info=exc_info,
                exc_stack=exc_stack,
            )
        
        # Stage 2: Correctness
        if not kernel_result.correctness:
            exc_type = "CorrectnessError"
            
            term_out.append("=" * 80)
            term_out.append("CORRECTNESS CHECK FAILED")
            term_out.append("=" * 80)
            
            # Check for runtime errors during correctness testing
            if 'runtime_error' in serializable_metadata:
                runtime_error = serializable_metadata.get('runtime_error', 'Unknown runtime error')
                runtime_error_name = serializable_metadata.get('runtime_error_name', 'RuntimeError')
                
                term_out.append(f"Runtime Error: {runtime_error_name}")
                term_out.append(f"\n{str(runtime_error)}")
                
                # Include traceback if available
                if 'runtime_error_traceback' in serializable_metadata:
                    term_out.append("\nFull Traceback:")
                    term_out.append(serializable_metadata['runtime_error_traceback'])
                
                exc_info = {
                    'runtime_error': str(runtime_error),
                    'runtime_error_name': runtime_error_name,
                    'metadata': serializable_metadata,
                }
            else:
                # Correctness issue (output mismatch)
                correctness_issue = serializable_metadata.get('correctness_issue', 'Output mismatch')
                trials_info = serializable_metadata.get('correctness_trials', 'Unknown')
                
                term_out.append(f"Correctness Issue: {correctness_issue}")
                term_out.append(f"Trials passed: {trials_info}")
                
                if 'max_difference' in serializable_metadata:
                    term_out.append(f"\nMaximum difference: {serializable_metadata['max_difference']}")
                if 'avg_difference' in serializable_metadata:
                    term_out.append(f"Average difference: {serializable_metadata['avg_difference']}")
                
                term_out.append("\nThe kernel output does not match the reference implementation.")
                term_out.append("Please verify the kernel logic, memory access patterns,")
                term_out.append("and ensure all operations match the reference semantics.")
                
                exc_info = {
                    'correctness_issue': correctness_issue,
                    'trials_info': trials_info,
                    'metadata': serializable_metadata,
                }
            
            return ExecutionResult(
                term_out=term_out,
                exec_time=exec_time,
                exc_type=exc_type,
                exc_info=exc_info,
                exc_stack=exc_stack,
            )
        
        # Stage 3: Success - Kernel is correct
        exec_time = kernel_result.runtime / 1000.0  # Convert ms to seconds
        
        term_out.append("=" * 80)
        term_out.append("KERNEL EVALUATION SUCCESS")
        term_out.append("=" * 80)
        term_out.append(f"✓ Compilation: PASSED")
        term_out.append(f"✓ Correctness: PASSED")
        
        trials_info = serializable_metadata.get('correctness_trials', 'Unknown')
        term_out.append(f"  Correctness trials: {trials_info}")
        
        if kernel_result.runtime > 0:
            term_out.append(f"\n✓ Performance: {kernel_result.runtime:.3f} ms")
            
            if kernel_result.runtime_stats:
                stats = kernel_result.runtime_stats
                term_out.append(f"  Mean: {stats.get('mean', 'N/A'):.3f} ms")
                term_out.append(f"  Std:  {stats.get('std', 'N/A'):.3f} ms")
                term_out.append(f"  Min:  {stats.get('min', 'N/A'):.3f} ms")
                term_out.append(f"  Max:  {stats.get('max', 'N/A'):.3f} ms")
                term_out.append(f"  Trials: {stats.get('num_trials', 'N/A')}")
        
        hardware = serializable_metadata.get('hardware', 'Unknown GPU')
        term_out.append(f"\nHardware: {hardware}")
        term_out.append(f"Runtime: {exec_time:.3f} seconds")
        
        return ExecutionResult(
            term_out=term_out,
            exec_time=exec_time,
            exc_type=None,
            exc_info=None,
            exc_stack=None,
        )
    
    def cleanup_session(self):
        """Clean up any resources (compatibility with AIDE Interpreter interface)."""
        # KernelBench evaluation cleans up after itself
        logger.debug("Kernel Interpreter cleanup called")
        pass
