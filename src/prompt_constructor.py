import os
from src.utils import read_file
'''
Construct Prompts
As basic as we can be, not to steer the LLM too much
'''

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)
    
def get_arch_definition(arch_src):
    '''
    Construct torch definition from original torch nn.Module definition 
    '''
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

def prompt_generate_custom_cuda(arc_src: str,
                                 example_arch_src: str,
                                 example_new_arch_src: str) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt

def prompt_generate_custom_cuda_oneshot_and_template(ref_arch_src: str) -> str:
    prompt = PROBLEM_STATEMENT

    example_vectoradd_model = read_file(os.path.join(REPO_TOP_PATH, "src/prompts/model_ex_1.py"))
    example_vectoradd_model_new = read_file(os.path.join(REPO_TOP_PATH, "src/prompts/model_new_ex_1.py"))
    example_fuse_pseudocode_model = read_file(os.path.join(REPO_TOP_PATH, "src/prompts/model_ex_2.py"))
    example_fuse_pseudocode_model_new = read_file(os.path.join(REPO_TOP_PATH, "src/prompts/model_new_ex_2.py"))

    prompt += f"""
    Here's an example to show you the syntax of inline embedding custom CUDA operators in torch. This given architecture is a pointwise addition example: \n
    ```
    {example_vectoradd_model}
    ``` \n
    The example generated architecture with custom CUDA kernels looks like this: 
    ```
    {example_vectoradd_model_new}
    ``` \n
    """

    prompt += f"""
    Here's a pseudocode example to show you operator fusion. The given architecture in torch: \n
    ```
    {example_fuse_pseudocode_model}
    ``` \n
    The example generated architecture with operator fusion looks like this: 
    ```
    {example_fuse_pseudocode_model_new}
    ``` \n
    """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt

def prompt_generate_custom_cuda_from_file_one_example(ref_arch_src, example_ind=1):
    """
    Check example_ind for prompt templates
    """
    # arch = get_arch_definition_from_file(arch_path)
    arch = ref_arch_src
    # These are strictly defined for now
    
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_{example_ind}.py")
    example_new_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_new_ex_{example_ind}.py")
    
    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(f"Example architecture file not found: {example_arch_path}")
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(f"Example new architecture file not found: {example_new_arch_path}")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)

def prompt_fix_compile(ref_arch_src, custom_cuda, metadata):
    prompt = PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed to compile:
    ```
    {custom_cuda}
    ```
    Here's the metadata of the compilation error:
    ```
    {metadata}
    ```
    Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt

def prompt_fix_correctness(ref_arch_src, custom_cuda, metadata):
    prompt = PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed correctness:
    ```
    {custom_cuda}
    ```
    Here's the metadata of the correctness error:
    ```
    {metadata}
    ```
    Please consider how your custom CUDA kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt