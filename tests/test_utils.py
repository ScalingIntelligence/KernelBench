import pytest
import os
import tempfile
from unittest.mock import patch
from kernelbench.utils import (
    read_file,
    set_gpu_arch,
    remove_code_block_header,
    maybe_multithread,
    extract_first_code,
    extract_code_blocks,
    extract_last_code,
    extract_python_code,
)

"""
Usage:
pytest test_utils.py
"""


def test_read_file():
    """Test the read_file function"""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write("Test content")
        temp_file_path = temp_file.name

    try:
        # Test reading an existing file
        content = read_file(temp_file_path)
        assert content == "Test content"

        # Test reading a non-existent file
        assert read_file(temp_file_path + "_nonexistent") == ""
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_set_gpu_arch():
    """Test GPU architecture setting"""
    # Test valid architectures
    set_gpu_arch(["Volta", "Ampere"])
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "Volta;Ampere"

    # Test invalid architecture
    with pytest.raises(ValueError):
        set_gpu_arch(["InvalidArch"])


def test_remove_code_block_header():
    """Test code block header removal"""
    # Test with language header
    assert (
        remove_code_block_header("python\ndef hello(): pass", "python")
        == "def hello(): pass"
    )

    # Test with no header
    assert (
        remove_code_block_header("def hello(): pass", "python") == "def hello(): pass"
    )

    # Test with different header
    code = "cpp\nint main() {}"
    assert remove_code_block_header(code, "python") == code  # Should not change
    assert remove_code_block_header(code, "cpp") == "int main() {}"


def test_maybe_multithread():
    """Test the multithreading utility"""

    # Define a simple test function
    def test_func(x, multiplier=2):
        return x * multiplier

    # Test with single thread
    results = maybe_multithread(test_func, [1, 2, 3], num_workers=1, multiplier=3)
    assert results == [3, 6, 9]

    # Test filtering behavior
    def filter_func(x):
        return x if x > 2 else None

    results = maybe_multithread(filter_func, [1, 2, 3, 4], num_workers=1)
    assert results == [3, 4]


def test_code_extraction():
    """Test the code extraction utilities"""

    # Test input with code blocks
    example = """Here's some code:
    ```python
    def hello():
        print("Hello")
    ```

    And another block:
    ```cpp
    int main() {
        return 0;
    }
    ```
    """

    # Test extract_first_code - should get the Python block
    first_code = extract_first_code(example, ["python", "cpp"])
    assert "def hello()" in first_code
    assert 'print("Hello")' in first_code

    # Test extract_last_code - should get the C++ block
    last_code = extract_last_code(example, ["python", "cpp"])
    assert "int main()" in last_code
    assert "return 0" in last_code

    # Test extract_code_blocks - should get both blocks
    all_code = extract_code_blocks(example, ["python", "cpp"])
    assert "def hello()" in all_code
    assert "int main()" in all_code

    # Test with no code blocks
    no_code = "This is text with no code blocks"
    assert extract_first_code(no_code, ["python"]) is None
    assert extract_last_code(no_code, ["python"]) is None
    assert extract_code_blocks(no_code, ["python"]) == ""

    # Test with empty code block
    empty_block = "```python\n```"
    assert extract_first_code(empty_block, ["python"]) == ""


def test_extract_python_code():
    """Test extracting Python code specifically"""
    # Input with Python code block
    text = """Here's some Python code:
    ```python
    def add(a, b):
        return a + b
    ```
    """

    # Should extract the Python code
    code = extract_python_code(text)
    assert "def add(a, b):" in code
    assert "return a + b" in code

    # Multiple Python blocks
    text_multiple = """
    ```python
    def add(a, b):
        return a + b
    ```

    And another:
    ```python
    def multiply(a, b):
        return a * b
    ```
    """

    code = extract_python_code(text_multiple)
    assert "def add" in code
    assert "def multiply" in code

    # No Python code
    assert extract_python_code("No code here") == ""
