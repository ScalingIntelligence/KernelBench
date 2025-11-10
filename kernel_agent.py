"""
Kernel Agent for AIDE Integration
Extends AIDE's Agent class with CUDA kernel-specific prompts and logic
"""

import logging
import random
from typing import Any, Callable, cast
from pathlib import Path

# Use absolute imports for compatibility
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agent import Agent, ExecCallbackType
from backend import query
from journal import Journal, Node
from utils.config import Config
from utils.metric import MetricValue, WorstMetricValue
from utils.response import extract_code, extract_text_up_to_code, wrap_code

from src.prompt_constructor import (
    prompt_generate_custom_cuda_from_prompt_template,
    prompt_fix_compile,
    prompt_fix_correctness,
)

logger = logging.getLogger("aide")


class KernelAgent(Agent):
    """
    Agent specialized for CUDA kernel optimization using AIDE's tree search.
    Extends Agent with kernel-specific prompts and evaluation logic.
    """
    
    def __init__(
        self,
        ref_arch_src: str,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        backend: str = "cuda",
        precision: str = "fp32",
    ):
        """
        Initialize kernel optimization agent.
        
        Args:
            ref_arch_src: Reference PyTorch architecture source code
            task_desc: Task description (problem name, etc.)
            cfg: AIDE configuration
            journal: Solution tree journal
            backend: Backend type ('cuda', 'triton', 'tilelang', 'cute')
            precision: Computation precision ('fp32', 'fp16', 'bf16')
        """
        super().__init__(task_desc=task_desc, cfg=cfg, journal=journal)
        self.ref_arch_src = ref_arch_src
        self.backend = backend
        self.precision = precision
        self.debug = cfg.get('debug', False)
        
        logger.info(f"Kernel Agent initialized for backend: {backend}, precision: {precision}")
    
    def update_data_preview(self):
        """Override to skip data preview generation for kernel optimization."""
        # Kernel optimization doesn't require data preview
        self.data_preview = None
    
    def step(self, exec_callback: ExecCallbackType):
        """
        Execute one step of kernel optimization.
        Overrides parent to skip data preview requirement.
        """
        if self.debug:
            print(f"[DEBUG KernelAgent.step] Starting step")
            print(f"[DEBUG KernelAgent.step] Journal has {len(self.journal)} nodes")
        
        parent_node = self.search_policy()
        if self.debug:
            print(f"[DEBUG KernelAgent.step] search_policy() returned: {parent_node}")
        logger.debug(f"Kernel Agent generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            if self.debug:
                print(f"[DEBUG KernelAgent.step] Calling _draft() (no parent)")
            result_node = self._draft()
        elif parent_node.is_buggy:
            if self.debug:
                print(f"[DEBUG KernelAgent.step] Calling _debug() (parent is buggy)")
            result_node = self._debug(parent_node)
        else:
            if self.debug:
                print(f"[DEBUG KernelAgent.step] Calling _improve() (parent is good)")
            result_node = self._improve(parent_node)

        if self.debug:
            print(f"[DEBUG KernelAgent.step] Generated node with code length: {len(result_node.code)}")
            print(f"[DEBUG KernelAgent.step] Calling exec_callback()...")
        
        exec_result = exec_callback(result_node.code, True)
        if self.debug:
            print(f"[DEBUG KernelAgent.step] exec_callback returned: {type(exec_result)}")
            print(f"[DEBUG KernelAgent.step] Parsing execution result...")
        
        self.parse_exec_result(
            node=result_node,
            exec_result=exec_result,
        )
        
        if self.debug:
            print(f"[DEBUG KernelAgent.step] Appending node to journal...")
        self.journal.append(result_node)
        if self.debug:
            print(f"[DEBUG KernelAgent.step] Step complete. Journal now has {len(self.journal)} nodes")
    
    @property
    def _prompt_environment(self):
        """Environment setup for CUDA kernel development."""
        env_prompt = {
            "Environment": [
                f"You are developing CUDA kernels using the {self.backend} backend.",
                f"Target precision: {self.precision}",
                "You have access to PyTorch's CUDA extension API for inline kernel compilation.",
                "Use torch.utils.cpp_extension.load_inline() to compile and load CUDA kernels.",
                "All necessary CUDA development tools and libraries are available.",
            ]
        }
        return env_prompt
    
    @property
    def _prompt_impl_guideline(self):
        """Implementation guidelines for CUDA kernel development."""
        impl_guideline = {
            "Implementation Guidelines": [
                "Your code should define a class named 'ModelNew' that inherits from torch.nn.Module.",
                "The ModelNew class should have the same interface as the original Model class.",
                "Use custom CUDA kernels via torch.utils.cpp_extension.load_inline().",
                "CRITICAL: load_inline() requires BOTH 'cpp_sources' (function declaration) and 'cuda_sources' (kernel implementation).",
                "Format: load_inline(name='...', cpp_sources='torch::Tensor func(...);', cuda_sources='...', functions=[...])",
                "The CUDA source should include the kernel definition (__global__) and a C++ wrapper function.",
                "The C++ wrapper should call the kernel with proper grid/block dimensions.",
                "Consider operator fusion opportunities (e.g., matmul+relu, conv+batchnorm).",
                "Consider algorithmic improvements (e.g., tiled matrix multiplication, online softmax).",
                "Ensure memory access patterns are coalesced for optimal GPU performance.",
                "Use shared memory effectively to reduce global memory traffic.",
                "The code must be self-contained and compile successfully.",
                "Do NOT include test code or execution code - only the ModelNew class definition.",
                "Your response should contain only a single Python code block.",
            ]
        }
        return impl_guideline
    
    @property
    def _prompt_resp_fmt(self):
        """Response format instructions."""
        return {
            "Response Format": (
                "Your response should consist of:\n"
                "1. A brief optimization plan (3-5 sentences) describing your approach\n"
                "2. A single Python code block (wrapped in ```) implementing the optimized ModelNew class\n"
                "Do not include any additional headings, explanations after the code, or test code."
            )
        }
    
    def _get_kernel_best_practices(self) -> dict:
        """Best practices for CUDA kernel optimization."""
        return {
            "CUDA Optimization Best Practices": [
                "Maximize parallelism by keeping all GPU cores busy",
                "Minimize memory transfers between host and device",
                "Coalesce global memory accesses for optimal bandwidth",
                "Use shared memory to cache frequently accessed data",
                "Avoid thread divergence within warps",
                "Optimize thread block sizes (typically multiples of 32)",
                "Consider register usage to maximize occupancy",
                "Use async operations when possible to overlap computation and memory transfers",
            ]
        }
    
    def _draft(self) -> Node:
        """Generate initial CUDA kernel implementation."""
        if self.debug:
            print(f"[DEBUG KernelAgent._draft] Starting draft generation")
        
        # Add few-shot example
        example_template = """
EXAMPLE: Here's how to properly use torch.utils.cpp_extension.load_inline() for custom CUDA kernels:

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source (includes both kernel and C++ wrapper)
cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    elementwise_add_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
\"\"\"

# C++ function declaration
cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code
elementwise_add = load_inline(
    name='elementwise_add',
    cpp_sources=cpp_source,  # REQUIRED: function declaration
    cuda_sources=cuda_source,  # REQUIRED: kernel implementation
    functions=['elementwise_add_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.elementwise_add = elementwise_add
    
    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
```

KEY POINTS:
1. load_inline() REQUIRES both cpp_sources (declaration) and cuda_sources (implementation)
2. CUDA source includes: #include <torch/extension.h>, kernel definition, and C++ wrapper
3. C++ wrapper handles memory allocation and kernel launch
4. Kernel is called via: module.function_name(args)
"""
        
        prompt: Any = {
            "Introduction": (
                "You are an expert CUDA kernel developer specializing in GPU optimization. "
                "Your task is to optimize a PyTorch neural network architecture by implementing "
                "custom CUDA kernels to replace standard PyTorch operators. "
                "This is your first implementation - focus on correctness and basic optimization."
            ),
            "Example": example_template,
            "Reference Architecture": f"Here is the original PyTorch architecture:\n```python\n{self.ref_arch_src}\n```",
            "Task": self.task_desc,
            "Memory": self.journal.generate_summary() if self.journal.nodes else "No previous attempts.",
            "Instructions": {},
        }
        
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Initial Implementation Strategy": [
                "Start with a correct implementation that compiles and runs successfully.",
                "Identify 1-2 key operators that would benefit most from custom CUDA kernels.",
                "For your first attempt, prioritize correctness over aggressive optimization.",
                "Consider the Memory section to avoid repeating failed approaches.",
                "Think about which operations are compute-bound vs memory-bound.",
            ]
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment
        prompt["Instructions"] |= self._get_kernel_best_practices()
        
        if self.debug:
            print(f"[DEBUG KernelAgent._draft] Calling plan_and_code_query()...")
        plan, code = self.plan_and_code_query(prompt)
        if self.debug:
            print(f"[DEBUG KernelAgent._draft] Got plan (len={len(plan)}) and code (len={len(code)})")
        return Node(plan=plan, code=code)
    
    def _improve(self, parent_node: Node) -> Node:
        """Generate improved CUDA kernel based on working parent implementation."""
        prompt: Any = {
            "Introduction": (
                "You are an expert CUDA kernel developer. "
                "You have a working CUDA kernel implementation that is correct but can be optimized further. "
                "Your goal is to improve its runtime performance through a single, focused optimization."
            ),
            "Reference Architecture": f"Here is the original PyTorch architecture:\n```python\n{self.ref_arch_src}\n```",
            "Previous Implementation": {
                "Plan": parent_node.plan,
                "Code": wrap_code(parent_node.code),
                "Performance": f"Runtime: {parent_node.metric.value:.3f} ms" if parent_node.metric else "Unknown",
                "Analysis": parent_node.analysis if parent_node.analysis else "No analysis available.",
            },
            "Task": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Optimization Strategy": [
                "Propose a SINGLE, SPECIFIC optimization to improve the previous implementation's runtime.",
                "This should be an atomic change that can be experimentally evaluated.",
                "Consider the Memory section to learn from past attempts and avoid redundant optimizations.",
                "Possible optimization directions:",
                "  - Fuse more operators into a single kernel",
                "  - Improve memory access patterns (coalescing, shared memory usage)",
                "  - Optimize thread block configuration",
                "  - Apply algorithmic improvements (e.g., better tiling strategy)",
                "  - Reduce kernel launch overhead",
                "  - Optimize register usage for better occupancy",
                "Your optimization plan should be 3-5 sentences describing the specific change.",
            ]
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)
    
    def _debug(self, parent_node: Node) -> Node:
        """Fix buggy CUDA kernel implementation."""
        # Determine error type from parent node
        is_compilation_error = parent_node.exc_type == "CompilationError"
        is_correctness_error = parent_node.exc_type == "CorrectnessError"
        
        error_type_str = "compilation error" if is_compilation_error else "correctness error"
        
        # Add helpful example for common issues
        example_template = """
REMINDER: Proper load_inline() usage requires BOTH cpp_sources and cuda_sources:

```python
cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(...) { /* kernel code */ }

torch::Tensor my_function(torch::Tensor input) {
    // Allocate output
    auto output = torch::zeros_like(input);
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(...);
    
    return output;
}
\"\"\"

cpp_source = "torch::Tensor my_function(torch::Tensor input);"

module = load_inline(
    name='my_module',
    cpp_sources=cpp_source,  # REQUIRED
    cuda_sources=cuda_source,  # REQUIRED
    functions=['my_function'],
    verbose=True,
)
```
"""
        
        prompt: Any = {
            "Introduction": (
                f"You are an expert CUDA kernel developer. "
                f"Your previous implementation had a {error_type_str}. "
                f"Analyze the error carefully and fix the issue while maintaining the optimization goals."
            ),
            "Example": example_template,
            "Reference Architecture": f"Here is the original PyTorch architecture:\n```python\n{self.ref_arch_src}\n```",
            "Previous (Buggy) Implementation": {
                "Plan": parent_node.plan,
                "Code": wrap_code(parent_node.code),
            },
            "Error Information": {
                "Error Type": parent_node.exc_type or "Unknown",
                "Error Details": wrap_code(parent_node.term_out, lang=""),
            },
            "Task": self.task_desc,
            "Instructions": {},
        }
        
        prompt["Instructions"] |= self._prompt_resp_fmt
        
        if is_compilation_error:
            prompt["Instructions"] |= {
                "Debugging Strategy for Compilation Errors": [
                    "Carefully review the compilation error messages.",
                    "Common issues:",
                    "  - MISSING cpp_sources or cuda_sources in load_inline() - BOTH are REQUIRED",
                    "  - Missing #include <torch/extension.h> or #include <cuda_runtime.h>",
                    "  - Function declared in cpp_sources not defined in cuda_sources",
                    "  - Incorrect CUDA kernel syntax (__global__ void kernel_name(...))",
                    "  - Type mismatches between Python and CUDA code",
                    "  - Missing template parameters or incorrect kernel launch syntax",
                    "  - Incompatible CUDA operations for the target architecture",
                    "Fix the specific compilation issue while preserving the optimization intent.",
                    "Ensure all CUDA code follows correct syntax and API usage.",
                    "VERIFY: load_inline() has both cpp_sources AND cuda_sources parameters.",
                ]
            }
        else:
            prompt["Instructions"] |= {
                "Debugging Strategy for Correctness Errors": [
                    "Carefully review the correctness error information.",
                    "Common issues:",
                    "  - Incorrect kernel logic or algorithm",
                    "  - Out-of-bounds memory accesses",
                    "  - Race conditions or synchronization issues",
                    "  - Incorrect handling of edge cases (boundaries, small sizes)",
                    "  - Numerical precision issues",
                    "  - Wrong reduction or aggregation logic",
                    "Compare your kernel's logic carefully with the reference implementation.",
                    "Ensure thread indexing, memory access patterns, and computations are correct.",
                    "Add necessary __syncthreads() barriers if using shared memory.",
                ]
            }
        
        prompt["Instructions"] |= self._prompt_impl_guideline
        
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)
    
    def parse_exec_result(self, node: Node, exec_result):
        """
        Parse kernel evaluation results and update node with metrics.
        Overrides parent method to use kernel-specific evaluation.
        """
        logger.info(f"Kernel Agent parsing execution results for node {node.id}")
        
        node.absorb_exec_result(exec_result)
        
        # Determine if the kernel is buggy based on execution result
        node.is_buggy = exec_result.exc_type is not None
        
        if node.is_buggy:
            # Kernel failed (compilation or correctness error)
            node.metric = WorstMetricValue()
            
            # Generate analysis based on error type
            if exec_result.exc_type == "CompilationError":
                node.analysis = (
                    f"Compilation failed. The CUDA kernel code has syntax or API errors. "
                    f"Error: {exec_result.exc_info.get('compilation_error_name', 'Unknown')}. "
                    f"Review the error messages and fix the compilation issues."
                )
            elif exec_result.exc_type == "CorrectnessError":
                if 'runtime_error' in exec_result.exc_info.get('metadata', {}):
                    node.analysis = (
                        f"Runtime error during correctness testing. "
                        f"Error: {exec_result.exc_info.get('runtime_error_name', 'Unknown')}. "
                        f"The kernel compiled but failed during execution. "
                        f"Check for memory access violations, synchronization issues, or invalid operations."
                    )
                else:
                    trials_info = exec_result.exc_info.get('trials_info', 'Unknown')
                    node.analysis = (
                        f"Correctness check failed ({trials_info}). "
                        f"The kernel output does not match the reference implementation. "
                        f"Review the kernel logic, memory access patterns, and numerical computations."
                    )
            else:
                node.analysis = f"Evaluation failed with error: {exec_result.exc_type}"
        else:
            # Kernel succeeded - extract performance metric
            runtime_ms = exec_result.exec_time * 1000.0  # Convert seconds to ms
            node.metric = MetricValue(runtime_ms, maximize=False)  # Lower runtime is better
            
            # Generate success analysis
            node.analysis = (
                f"Kernel implementation successful! "
                f"Runtime: {runtime_ms:.3f} ms. "
                f"The kernel compiled correctly, passed all correctness checks, and was successfully profiled."
            )
        
        logger.info(
            f"Node {node.id}: is_buggy={node.is_buggy}, "
            f"metric={node.metric.value if node.metric else 'N/A'}, "
            f"stage={node.stage_name}"
        )
