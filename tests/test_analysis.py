import pytest
import numpy as np
from kernelbench.analysis import pass_at_k, extract_all_cuda_sources

"""
Usage:
pytest test_analysis.py
"""


def test_pass_at_k():
    """Test the pass@k metric calculation"""
    # Common use cases
    assert pass_at_k(10, 5, 1) == 0.5  # 5/10 correct = 50% pass@1
    assert pass_at_k(10, 0, 5) == 0.0  # None correct = 0%
    assert pass_at_k(10, 10, 1) == 1.0  # All correct = 100%

    # Pass@k should be higher for larger k values
    # (more chances to pass when drawing more samples)
    assert pass_at_k(10, 5, 5) > pass_at_k(10, 5, 1)


def test_extract_all_cuda_sources():
    """Test extraction of CUDA code from triple-quoted strings"""
    # Test with a single CUDA kernel
    code_single = '''
    kernel = """
    __global__ void add(int *a, int *b, int *c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        c[i] = a[i] + b[i];
    }
    """
    '''
    extracted = extract_all_cuda_sources(code_single)
    assert len(extracted) == 1
    assert "__global__" in extracted[0]
    assert "c[i] = a[i] + b[i]" in extracted[0]

    # Test with multiple CUDA kernels
    code_multiple = '''
    kernel1 = """
    __global__ void add(int *a, int *b, int *c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        c[i] = a[i] + b[i];
    }
    """

    kernel2 = """
    __global__ void multiply(int *a, int *b, int *c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        c[i] = a[i] * b[i];
    }
    """
    '''
    extracted = extract_all_cuda_sources(code_multiple)
    assert len(extracted) == 2
    assert "add" in extracted[0]
    assert "multiply" in extracted[1]

    # Test with no CUDA code
    assert len(extract_all_cuda_sources("def python_function(): pass")) == 0
