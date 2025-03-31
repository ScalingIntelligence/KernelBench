import pytest
from kernelbench.utils import extract_last_code


def test_extract_last_code():
    """Test the extract_last_code function."""
    test_string = """
    Some text before code
    ```python
    def test_function():
        return "test"
    ```
    Some text after code
    """
    result = extract_last_code(test_string, ["python"])
    assert "def test_function():" in result
    assert 'return "test"' in result


def test_imports():
    """Test that the utils module can be imported."""
    from kernelbench import utils

    # Simple assertion to ensure imports work
    assert utils is not None
