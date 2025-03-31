import pytest
from kernelbench.dataset import get_code_hash

"""
Usage
pytest test_dataset.py
"""


def test_get_code_hash():
    """Test the code hashing functionality"""

    # Test semantic equivalence (different comments, whitespace, but same code)
    code1 = """
    import torch
    # This is for a single batch
    '''
    Some random multi-line comment
    '''
    B = 1
    """

    code2 = """
    import torch
    '''
    More problem descriptions (updated)
    '''
    # low batch setting

    B = 1
    """

    code3 = "import torch\nB = 1"

    # All three versions should hash to the same value
    assert get_code_hash(code1) == get_code_hash(code2) == get_code_hash(code3)

    # Test that meaningful code changes cause different hashes
    code_different = """
    import torch
    B = 64  # Different batch size
    """

    assert get_code_hash(code1) != get_code_hash(code_different)

    # Test case sensitivity
    code_case1 = "def test(): pass"
    code_case2 = "def TEST(): pass"

    assert get_code_hash(code_case1) != get_code_hash(code_case2)

    # Test stability (multiple calls should return the same hash)
    complex_code = """
    import torch
    def complex_function(x, y):
        return torch.matmul(x, y)
    """

    hash1 = get_code_hash(complex_code)
    hash2 = get_code_hash(complex_code)
    assert hash1 == hash2
