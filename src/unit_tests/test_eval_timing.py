import os


"""
Test Timing

We want to systematically study different timing methodologies.

"""
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# use exampls in the few shot directory
EXAMPLES_PATH = os.path.join(REPO_PATH, "src", "prompts", "few_shot")

# Configure your test cases here
TEST_REF_FILE = "model_ex_tiled_matmul.py"
TEST_KERNEL_FILE = "model_new_ex_tiled_matmul.py"

assert os.path.exists(os.path.join(EXAMPLES_PATH, TEST_REF_FILE)), f"Reference file {TEST_REF_FILE} does not exist in {EXAMPLES_PATH}"
assert os.path.exists(os.path.join(EXAMPLES_PATH, TEST_KERNEL_FILE)), f"Kernel file {TEST_KERNEL_FILE} does not exist in {EXAMPLES_PATH}"


