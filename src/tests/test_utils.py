import pytest # noqa    
 
from src.utils import extract_first_code, construct_problem_dataset_from_problem_dir



def test_extract_first_code():
    # Test with Python code block
    example_outpu = """The LLM wrote some code here
    ```python
    def hello():
        print("Hello")
    ```
    and it says more stuff afterwards"""
    
    code = extract_first_code(example_outpu, "python")
    assert code == 'def hello():\n    print("Hello")'

    # Test with no code block
    text = "Some code here"
    code = extract_first_code(text, "python") 
    assert code is None

    # Test with empty code block
    text = "```python\n```"
    code = extract_first_code(text, "python")
    assert code == ""


# Test python hash

