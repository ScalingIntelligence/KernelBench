import pytest # noqa    
 
from kernelbench.utils import extract_first_code, extract_code_blocks, extract_last_code

def check_code_assertions(code: str, expected_code: str):
    """
    Check code is equivalent (don't worry about whitespace)
    """
    if code is None:
        assert expected_code == ""
    else:
        assert code.replace("\n", "").replace(" ", "") == expected_code.replace("\n", "").replace(" ", "")


def test_extract_last_code():
    # Test with Python code block
    example_output = """The LLM wrote some code here
    ```python
    def hello():
        print("Hello")
    ```
    and it says more stuff afterwards"""
    code = extract_last_code(example_output, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")


    example_output = """The LLM wrote some code here
    ```cpp
    int main() {
        return 0;
    }
    ```

    and some other code block 
    ```python
    def hello():
        print("Hello")
    ``` 
    and it says more stuff afterwards"""
    code = extract_last_code(example_output, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")



def test_extract_first_code():
    # Test with Python code block
    example_output = """The LLM wrote some code here
    ```python
    def hello():
        print("Hello")
    ```
    and it says more stuff afterwards"""
    
    code = extract_first_code(example_output, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")

    # Test with no code block
    text = "Some code here"
    code = extract_first_code(text, ["python", "cpp"]) 
    check_code_assertions(code, "")

    # Test with empty code block
    text = "```python\n```"
    code = extract_first_code(text, ["python", "cpp"])
    check_code_assertions(code, "")


    # Test with multiple code blocks
    text = """```python
    def hello():
        print("Hello")
    ```

    ```cpp
    int main() {
        return 0;
    }
    ```
    """
    # NOTE: is this a problem 
    code = extract_first_code(text, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")
# Test python hash



def test_extract_code_blocks():
    text = """```python
    def hello():
        print("Hello")
    ```
    """
    code = extract_code_blocks(text, ["python", "rust"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")

    text = """```python
    def hello():
        print("Hello")
    ```

    ```cpp
    int main() {
        return 0;
    }
    ```
    """
    # NOTE: is this a problem 
    code = extract_code_blocks(text, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\") \n int main() { \n return 0; \n }")


def test_untagged_code_block_variable_preservation():
    """Untagged code blocks starting with language names (e.g. python_var = 123) do not have prefix stripped."""
    content = "```\npython_var = 123\ncpp_var = 456\n```"
    first = extract_first_code(content, ["python", "cpp"])
    assert first == "python_var = 123\ncpp_var = 456"

    last = extract_last_code(content, ["python", "cpp"])
    assert last == "python_var = 123\ncpp_var = 456"


def test_tagged_code_block_crlf_and_trailing_spaces():
    """Tagged code blocks with trailing spaces and CRLF \\r\\n parse correctly."""
    content = "```python   \r\ndef foo():\r\n    return 42\r\n```"
    first = extract_first_code(content, ["python"])
    assert "def foo():" in first
    assert "return 42" in first

    last = extract_last_code(content, ["python"])
    assert "def foo():" in last
    assert "return 42" in last


def test_unclosed_code_blocks_at_eof():
    """Unclosed code blocks at EOF are extracted properly by both extract_first_code and extract_last_code."""
    # Single unclosed block
    content = "```python\ndef bar():\n    return 'hello'"
    assert extract_first_code(content, ["python"]) == "def bar():\n    return 'hello'"
    assert extract_last_code(content, ["python"]) == "def bar():\n    return 'hello'"

    # Closed block followed by unclosed block at EOF
    mixed = (
        "```python\ndef first():\n    pass\n```\n"
        "Some text in between\n"
        "```python\ndef second():\n    pass"
    )
    assert extract_first_code(mixed, ["python"]) == "def first():\n    pass"
    assert extract_last_code(mixed, ["python"]) == "def second():\n    pass"


def test_multiple_code_blocks_different_languages():
    """Multiple code blocks with different languages return the correct language-matched block."""
    content = (
        "```bash\n"
        "echo 'running script'\n"
        "```\n\n"
        "```python\n"
        "print('hello from python')\n"
        "```\n"
    )
    # Extract python
    assert extract_first_code(content, ["python"]) == "print('hello from python')"
    assert extract_last_code(content, ["python"]) == "print('hello from python')"

    # Extract bash
    assert extract_first_code(content, ["bash"]) == "echo 'running script'"
    assert extract_last_code(content, ["bash"]) == "echo 'running script'"


def test_none_input_returns_none():
    """None input returns None."""
    assert extract_first_code(None, ["python"]) is None
    assert extract_last_code(None, ["python"]) is None


