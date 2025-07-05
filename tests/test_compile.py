import pytest
from unittest.mock import patch, MagicMock

"""
Usage:
pytest test_compile.py
"""


@patch("torch.cuda.is_available", return_value=False)
def test_cuda_detection(mock_cuda_available):
    """Test CUDA availability detection"""
    import torch

    # Should detect that CUDA is not available
    assert torch.cuda.is_available() == False
    mock_cuda_available.assert_called_once()


@patch("kernelbench.compile.compile_and_benchmark_kernel")
def test_compile_code(mock_compile_benchmark):
    """Test the code compilation handling"""
    from kernelbench.compile import compile_code

    # Mock successful compilation
    mock_compile_benchmark.return_value = {
        "status": "success",
        "compile_time": 0.5,
        "benchmark_results": {"mean": 1.2, "std": 0.1, "min": 1.0, "max": 1.5},
    }

    # Test with valid kernel
    valid_code = '''
    valid_kernel = """
    __global__ void add(int *a, int *b, int *c) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        c[tid] = a[tid] + b[tid];
    }
    """
    '''

    result = compile_code(valid_code)
    assert result["status"] == "success"
    assert "benchmark_results" in result

    # Test handling of invalid code
    mock_compile_benchmark.side_effect = Exception("Compilation error")
    result = compile_code("invalid code", skip_on_error=True)
    assert result["status"] == "skipped"
    assert "error" in result


@patch("kernelbench.compile.torch.cuda.is_available", return_value=True)
def test_data_type_mappings(mock_cuda_available):
    """Test the data type mapping functionality"""
    from kernelbench.compile import get_data_type_mappings

    # Check that we get back a valid mapping dictionary
    dtype_map = get_data_type_mappings()
    assert isinstance(dtype_map, dict)

    # Should contain common CUDA types
    assert "float32" in dtype_map
    assert "int32" in dtype_map
