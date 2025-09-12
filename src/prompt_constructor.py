import os
import random
from .utils import read_file, read_json_file, WorkArgs
from .run_utils import fetch_kernel_from_disk, fetch_eval_results_for_problem, fetch_eval_result_from_disk
from .eval import KernelExecResult


"""
Construct Prompt

Design principles: 
- To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
- However, we do not do extensive prompt engineering or few-shot example in the LLM to steer behaviour. 
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)


def get_arch_definition(arch_src):
    """
    Construct torch definition from original torch nn.Module definition
    """
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
PROBLEM_INSTRUCTION_IMPROVE = """
Optimize the architecture named Model with custom CUDA operators! 
Improve upon your previous attempts by debugging any correctness issues or improving the efficiency if the kernel was correct.
Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
PROBLEM_INSTRUCTION_COT = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
Let's think step by step.\n
""" 

TRITON_PROBLEM_STATEMENT = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
TRITON_PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
TRITON_PROBLEM_INSTRUCTION_IMPROVE = """
Optimize the architecture named Model with custom Triton operators! 
Improve upon your previous attempts by debugging any correctness issues or improving the efficiency if the kernel was correct.
Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. \n
"""
TRITON_PROBLEM_INSTRUCTION_COT = """
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks in markdown format (i.e. ```python or ```cpp). Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
Let's think step by step.\n
""" 


def get_problem_statement(triton=False):
    if triton:
        return TRITON_PROBLEM_STATEMENT
    else:
        return PROBLEM_STATEMENT
    
def get_problem_instruction(triton=False):
    if triton:
        return TRITON_PROBLEM_INSTRUCTION
    else:
        return PROBLEM_INSTRUCTION

def get_problem_instruction_improve(triton=False):
    if triton:
        return TRITON_PROBLEM_INSTRUCTION_IMPROVE
    else:
        return PROBLEM_INSTRUCTION_IMPROVE

def get_problem_instruction_cot(triton=False):
    if triton:
        return TRITON_PROBLEM_INSTRUCTION_COT
    else:
        return PROBLEM_INSTRUCTION_COT


def prompt_bare(ref_arch_src: str, triton=False) -> str:
    prompt = get_problem_statement(triton)
    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
    prompt += get_problem_instruction(triton)
    return prompt


def prompt_with_one_example(
    arc_src: str, example_arch_src: str, example_new_arch_src: str, triton=False, rule_path=None
) -> str:
    prompt = get_problem_statement(triton)

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
```python
{example_arch_src}
``` \n
        The example new arch with custom CUDA kernels looks like this: 
```python
{example_new_arch_src}
``` \n
        """
    
    if rule_path is not None:
        rules = read_json_file(rule_path)
        rules_str = "\n".join(rules)

        prompt += f"""Here are guidelines for writing efficient CUDA kernels: \n
{rules_str}
"""

    prompt += f"""
    You are given the following architecture: \n
```python
{arc_src}
```
    """
    prompt += get_problem_instruction(triton)
    return prompt


def prompt_base(ref_arch_src: str, triton=False, rule_path=None) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    if triton:
        example_new_arch_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/model_new_ex_add_triton.py"
        )
    else:
        example_new_arch_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
        )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_with_one_example(arch, example_arch, example_new_arch, triton, rule_path)


def prompt_cot(ref_arch_src: str, cot_example: str = "ex_fuse_gelu", triton=False) -> str:
    """
    Generate a prompt with a CoT example following a template 
    Avaliable CoT examples: 
    - ex_fuse_gelu: fused gelu
    - ex_mnist2: fused convolutions and relus
    - ex_tiled_matmul: tiled matrix multiplication
    """

    prompt = get_problem_statement(triton)
    
    assert cot_example in ["ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul"]

    # k = 2
    example_fuse_gelu = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
    )
    example_fuse_gelu_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_fuse_gelu.py")
    )
    example_fuse_gelu_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
    )
    example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

    # k = 3
    example_mnist2 = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
    )
    example_mnist2_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_mnist2.py")
    )
    example_mnist2_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
    )
    exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

    # k = 4
    example_tiled_matmul = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
    )
    example_tiled_matmul_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_tiled_matmul.py")
    )
    example_tiled_matmul_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
    )
    example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "
    
    match cot_example:
        case "ex_fuse_gelu":
            base = example_fuse_gelu
            cot = example_fuse_gelu_cot
            kernel = example_fuse_gelu_new
            desc = example_fuse_gelu_desc
        case "ex_mnist2":
            base = example_mnist2
            cot = example_mnist2_cot
            kernel = example_mnist2_new
            desc = exmaple_mnist2_desc
        case "ex_tiled_matmul":
            base = example_tiled_matmul
            cot = example_tiled_matmul_cot
            kernel = example_tiled_matmul_new
            desc = example_tiled_matmul_desc
        case _:
            raise ValueError(f"Invalid CoT example: {cot_example} not found in CoT examples")

    prompt += f"""
Here is an example architecture:\n\n
```
{base}
```\n
{get_problem_instruction_cot(triton)} \n
{cot} \n
```
{kernel}
```\n\n
"""

# show task to solve
    prompt += f"""
Task:\n\n
Here is an example architecture:\n\n
```
{ref_arch_src}
```\n
"""
    prompt += get_problem_instruction_cot(triton)

    return prompt


def prompt_main(ref_arch_src: str, config, triton=False, rules=None) -> str:
    match config.prompt:
        case "regular":
            return prompt_base(ref_arch_src, triton, rules)
        case "cot":
            return prompt_cot(ref_arch_src, cot_example="ex_fuse_gelu", triton=triton)
        case _:
            raise ValueError(f"Invalid prompt type: {config.prompt}")


def exec_result_to_exeution_feedback(exec_result: dict) -> str:
    if isinstance(exec_result, KernelExecResult):
        metadata = exec_result.metadata
        correctness = exec_result.correctness
        runtime = exec_result.runtime
    else:
        metadata = exec_result['metadata']
        correctness = exec_result['correctness']
        runtime = exec_result['runtime']

    compilation_error = metadata['compilation_error'] if 'compilation_error' in metadata else None
    runtime_error = metadata['runtime_error'] if 'runtime_error' in metadata else None
    correctness_issue = metadata['correctness_issue'] if 'correctness_issue' in metadata else None
    other_error = metadata['other_error'] if 'other_error' in metadata else None
    correctness_feedback = compilation_error if compilation_error else runtime_error if runtime_error else correctness_issue if correctness_issue else other_error if other_error else "All trials passed" 

    evaluation_feedback = f"""
Here is your Evaluation Result:
```
{correctness_feedback}
```
"""

    if correctness:
        evaluation_feedback += f"""
Your kernel executed successfully and produced the correct output.
Here is your wall clock time: {runtime} milliseconds.

{metadata["profiler_info"]}
"""

    return evaluation_feedback
 

def prompt_refinement_from_last_kernel(ref_arch_src: str, config, last_kernel_src: str, last_exec_result: KernelExecResult, triton=False) -> str:
    prompt = prompt_main(ref_arch_src, config, triton)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""

    prompt += get_problem_instruction_improve(triton)
    return prompt


def prompt_refinement_from_history(ref_arch_src: str, history: list[tuple[str, KernelExecResult]], triton=False, rule_path=None) -> str:
    prompt = prompt_base(ref_arch_src, triton, rule_path)

    for kernel_src, exec_result in history:

        execution_feedback = exec_result_to_exeution_feedback(exec_result)

        prompt += f"""Your generated kernel:
```
{kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""
    
    prompt += get_problem_instruction_improve(triton)
    return prompt


def prompt_idea_generation(ref_arc_src: str, config, last_kernel_src: str, last_exec_result: KernelExecResult, triton=False) -> str:
    prompt = prompt_main(ref_arc_src, config, triton)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""

    prompt += "Generate an idea for how to improve the kernel. Please do not output code yet, just the idea."
    return prompt

def prompt_refinement_from_idea(ref_arc_src: str, config, last_kernel_src: str, last_exec_result: KernelExecResult, idea: str, triton=False) -> str:
    prompt = prompt_main(ref_arc_src, config, triton)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}

Here is your idea for how to improve the kernel:
```
{idea}
```
"""

    prompt += get_problem_instruction(triton)
    return prompt


def generate_prompt_iterative_refinement(work: WorkArgs, config, ref_arch_src: str, inference_server: callable, run_dir: str, triton=False, rule_path=None) -> str:
    if work.sample_id < config.num_parallel:
        return prompt_main(ref_arch_src, config, triton)
    
    # Fetch previous history of kernels
    history = []
    for sample_id in range(work.sample_id % config.num_parallel, work.sample_id):
        kernel_src, _ = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, sample_id)
        exec_result = fetch_eval_result_from_disk(run_dir, config.level, work.problem_id, sample_id)
        history.append((kernel_src, exec_result))
    
    # Construct prompt
    prompt = prompt_refinement_from_history(ref_arch_src, history, triton, rule_path)
    
    return prompt


def generate_prompt_metr(work: WorkArgs, config, ref_arch_src: str, inference_server: callable, run_dir: str, triton=False) -> str:
    if work.sample_id <= config.num_parallel:
        return prompt_main(ref_arch_src, config, triton)
    
    # Fetch evaluation results
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = fetch_eval_results_for_problem(work.level, work.problem_id, eval_file_path)

    ref_kernel_result = eval_results["0"]
    assert ref_kernel_result["correctness"], "Reference kernel is not correct"

    correct_kernels = [eval_result for eval_result in eval_results.values() if eval_result["correctness"]]
    
    # Sample from the correct kernels based on efficiency
    speedups = [ref_kernel_result["runtime"] / eval_result["runtime"] for eval_result in correct_kernels]
    sampled_kernel_eval_result = random.choices(correct_kernels, weights=speedups)[0]
    sampled_kernel_id = int(sampled_kernel_eval_result["sample_id"])
    if config.verbose:
        print(f"[METR] Sampled kernel {sampled_kernel_id} with speedup {ref_kernel_result['runtime'] / sampled_kernel_eval_result['runtime']}")

    sampled_kernel_src, _ = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, sampled_kernel_id)

    return prompt_refinement_from_last_kernel(ref_arch_src, config, sampled_kernel_src, sampled_kernel_eval_result, triton)


def generate_prompt_stanford(work: WorkArgs, config, ref_arch_src: str, inference_server: callable, run_dir: str, triton=False) -> str:
    if work.sample_id < config.num_parallel:
        return prompt_main(ref_arch_src, config, triton)
    
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = fetch_eval_results_for_problem(work.level, work.problem_id, eval_file_path)
    # Get best kernel(s) from last round
    last_iteration_start_id = (work.sample_id // config.num_parallel - 1) * config.num_parallel
    last_step_sample_id_range = range(last_iteration_start_id, last_iteration_start_id + config.num_parallel)
    last_step_eval_results = [eval_results[str(sample_id)] for sample_id in last_step_sample_id_range]
    last_step_correct_kernels = [eval_result for eval_result in last_step_eval_results if eval_result["correctness"]]
    last_step_incorrect_kernels = [eval_result for eval_result in last_step_eval_results if not eval_result["correctness"]]
    last_step_best_kernels = sorted(last_step_correct_kernels, key=lambda x: x["runtime"])
    if len(last_step_best_kernels) < config.num_best:
        # If not enough correct kernels, randomly sample incorrect kernels
        last_step_best_kernels = last_step_best_kernels + random.choices(last_step_incorrect_kernels, k=config.num_best - len(last_step_best_kernels))

    last_step_best_kernel = last_step_best_kernels[work.sample_id % config.num_best] # use top config.num_best kernels
    last_step_best_kernel_src, _ = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, int(last_step_best_kernel["sample_id"]))
    if config.verbose:
        print(f"[Stanford] Last step best kernel sample_id: {int(last_step_best_kernel['sample_id'])}")

    prompt = prompt_idea_generation(ref_arch_src, config, last_step_best_kernel_src, last_step_best_kernel, triton)

    idea = inference_server(prompt)

    prompt = prompt_refinement_from_idea(ref_arch_src, config, last_step_best_kernel_src, last_step_best_kernel, idea, triton)
    return prompt


def generate_prompt(work: WorkArgs, config, ref_arch_src: str, inference_server: callable, run_dir: str, rule_path=None) -> str:
    triton = "KernelLLM" in config.model_name
    match config.method:
        case "base":
            return prompt_main(ref_arch_src, config, triton)
        case "best-of-N":
            return prompt_main(ref_arch_src, config, triton)
        case "iterative refinement":
            return generate_prompt_iterative_refinement(work, config, ref_arch_src, inference_server, run_dir, triton, rule_path)
        case "METR":
            return generate_prompt_metr(work, config, ref_arch_src, inference_server, run_dir, triton)
        case "Stanford":
            return generate_prompt_stanford(work, config, ref_arch_src, inference_server, run_dir, triton)
        case _:
            raise ValueError(f"Invalid method: {config.method}")


def prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src: str, gpu_name: str) -> str:
    """
    Similar to prompt_generate_custom_cuda_from_prompt_template, 
    but with hardware information for the given GPU
    """

    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    gpu_spec_file_path = os.path.join(REPO_TOP_PATH, f"src/prompts/hardware/gpu_specs.py")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    gpu_spec_info = read_file(gpu_spec_file_path)

    return prompt_generate_prompt_with_hardware_info(
                                        ref_arch_src=arch, 
                                        gpu_name=gpu_name, 
                                        example_arch_src=example_arch, 
                                        example_new_arch_src=example_new_arch, 
                                        gpu_spec_info_src=gpu_spec_info
                                        )
    


def prompt_generate_prompt_with_hardware_info(ref_arch_src: str, 
                                              gpu_name: str, 
                                              example_arch_src: str, 
                                              example_new_arch_src: str, 
                                              gpu_spec_info_src: str) -> str:
    """
    Generate a prompt with hardware information for the given GPU
    gpu_spec_info_src: str of the gpu spec src file
    """

    # Create a dictionary to store the local namespace
    local_dict = {}
    
    # Execute the GPU spec file in the local namespace
    exec(gpu_spec_info_src, {}, local_dict)
    
    # Get the required variables from the local namespace
    GPU_SPEC_INFO = local_dict.get('GPU_SPEC_INFO')
    GPU_DEFINITIONS = local_dict.get('GPU_DEFINITIONS')
    GPU_BEST_PRACTICES = local_dict.get('GPU_BEST_PRACTICES')
    
    if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
        raise ValueError("GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_spec_info_src")

    assert gpu_name in GPU_SPEC_INFO, f"GPU name {gpu_name} not found in GPU_SPEC_INFO"

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
    
    curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]

    gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
    prompt += f"""
    Here is some information about the underlying hardware that you should keep in mind. \n\n
The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.\n\n"""
    
    for key, value in curr_gpu_spec_info.items():
        if key == "GPU Architecture":
            continue
        prompt += f"""- We have {value} of {key}.\n"""
    
    
    prompt += f"""\n\n
Here are some concepts about the GPU architecture that could be helpful: \n\n"""
    for key, value in GPU_DEFINITIONS.items():
        prompt += f"""- {key}: {value}\n"""

    prompt += f"""\n\n
Here are some best practices for writing CUDA kernels on GPU: \n\n"""
    for best_practice in GPU_BEST_PRACTICES:
        prompt += f"""- {best_practice}\n"""


    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
    

    prompt += PROBLEM_INSTRUCTION
    return prompt


def main():
    gpu_name = "L40S"


    ref_arch_src = read_file(os.path.join(KERNEL_BENCH_PATH, f"level1/19_ReLU.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"
    prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src, gpu_name)
    print(prompt)
    # Write prompt to temp file
    temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", "prompt_draft.txt")
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write(prompt)

if __name__ == "__main__":
    main()
