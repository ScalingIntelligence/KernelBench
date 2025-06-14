"""
Prompt Templates for test-time scaling methods
"""
import os
import random

from src.utils import read_file
from src.eval import ExecutionResult

from configs import TestTimeScalingConfig
from utils import WorkArgs, fetch_kernel_from_disk, fetch_eval_result_from_disk, get_evaluation_results_for_problem

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


############################################
# Ported from prompt_constructor.py
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""


def prompt_with_one_example(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
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


def prompt_base(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_DIR, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_DIR, f"src/prompts/model_new_ex_add.py"
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

    return prompt_with_one_example(arch, example_arch, example_new_arch)


def exec_result_to_exeution_feedback(exec_result: ExecutionResult) -> str:
    compilation_error = exec_result['metadata']['compilation_error'] if 'compilation_error' in exec_result['metadata'] else None
    runtime_error = exec_result['metadata']['runtime_error'] if 'runtime_error' in exec_result['metadata'] else None
    correctness_issue = exec_result['metadata']['correctness_issue'] if 'correctness_issue' in exec_result['metadata'] else None
    correctness_feedback = compilation_error if compilation_error else runtime_error if runtime_error else correctness_issue if correctness_issue else "All trials passed" 

    evaluation_feedback = f"""
Here is your Evaluation Result:
```
{correctness_feedback}
```
"""

    if exec_result["correctness"]:
        evaluation_feedback += f"""
Your kernel executed successfully and produced the correct output.
Here is your wall clock time: {exec_result["runtime"]} milliseconds.

{exec_result["metadata"]["profiler_info"]}
"""

    return evaluation_feedback


def prompt_refinement_from_last_kernel(ref_arch_src: str, last_kernel_src: str, last_exec_result: ExecutionResult) -> str:
    prompt = prompt_base(ref_arch_src)
    execution_feedback = exec_result_to_exeution_feedback(last_exec_result)

    prompt += f"""Your latest generated kernel:
```
{last_kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""

    prompt += PROBLEM_INSTRUCTION
    return prompt


def prompt_refinement_from_history(ref_arch_src: str, history: list[tuple[str, ExecutionResult]]) -> str:
    prompt = prompt_base(ref_arch_src)

    for kernel_src, exec_result in history:

        execution_feedback = exec_result_to_exeution_feedback(exec_result)

        prompt += f"""Your generated kernel:
```
{kernel_src}
```

Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model.
{execution_feedback}
"""
    
    prompt += PROBLEM_INSTRUCTION
    return prompt


def prompt_idea_generation(ref_arc_src: str, last_kernel_src: str, last_exec_result: ExecutionResult) -> str:
    prompt = prompt_base(ref_arc_src)
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

def prompt_refinement_from_idea(ref_arc_src: str, last_kernel_src: str, last_exec_result: ExecutionResult, idea: str) -> str:
    prompt = prompt_base(ref_arc_src)
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

    prompt += PROBLEM_INSTRUCTION
    return prompt


def generate_prompt_best_of_n(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, inference_server: callable, run_dir: str) -> str:
    # Default prompt
    return prompt_base(ref_arch_src)


def generate_prompt_iterative_refinement(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, inference_server: callable, run_dir: str) -> str:
    if work.sample_id < config.num_parallel:
        return prompt_base(ref_arch_src)
    
    # Fetch previous history of kernels
    history = []
    for sample_id in range(work.sample_id % config.num_parallel, work.sample_id):
        kernel_src = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, sample_id)
        exec_result = fetch_eval_result_from_disk(run_dir, config.level, work.problem_id, sample_id)
        history.append((kernel_src, exec_result))
    
    # Construct prompt
    prompt = prompt_refinement_from_history(ref_arch_src, history)
    
    return prompt


def generate_prompt_metr(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, inference_server: callable, run_dir: str) -> str:
    if work.sample_id <= config.num_parallel:
        return prompt_base(ref_arch_src)
    
    # Fetch evaluation results
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = get_evaluation_results_for_problem(work.problem_id, eval_file_path)

    ref_kernel_result = eval_results["0"]
    assert ref_kernel_result["correctness"], "Reference kernel is not correct"

    correct_kernels = [eval_result for eval_result in eval_results.values() if eval_result["correctness"]]
    
    # Sample from the correct kernels based on efficiency
    speedups = [ref_kernel_result["runtime"] / eval_result["runtime"] for eval_result in correct_kernels]
    sampled_kernel_eval_result = random.choices(correct_kernels, weights=speedups)[0]
    sampled_kernel_id = int(sampled_kernel_eval_result["sample_id"])
    if config.verbose:
        print(f"[METR] Sampled kernel {sampled_kernel_id} with speedup {ref_kernel_result['runtime'] / sampled_kernel_eval_result['runtime']}")

    sampled_kernel_src = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, sampled_kernel_id)

    return prompt_refinement_from_last_kernel(ref_arch_src, sampled_kernel_src, sampled_kernel_eval_result)


def generate_prompt_stanford(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, inference_server: callable, run_dir: str) -> str:
    if work.sample_id < config.num_parallel:
        return prompt_base(ref_arch_src)
    
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = get_evaluation_results_for_problem(work.problem_id, eval_file_path)
    # Get best kernel(s) from last round
    last_iteration_start_id = (work.sample_id // config.num_parallel) * config.num_parallel
    last_step_sample_id_range = range(last_iteration_start_id, last_iteration_start_id + config.num_parallel)
    last_step_eval_results = [eval_results[str(sample_id)] for sample_id in last_step_sample_id_range]
    last_step_correct_kernels = [eval_result for eval_result in last_step_eval_results if eval_result["correctness"]]
    last_step_incorrect_kernels = [eval_result for eval_result in last_step_eval_results if not eval_result["correctness"]]
    last_step_best_kernels = sorted(last_step_correct_kernels, key=lambda x: x["runtime"])
    if len(last_step_best_kernels) < config.num_best:
        # If not enough correct kernels, randomly sample incorrect kernels
        last_step_best_kernels = last_step_best_kernels + random.choices(last_step_incorrect_kernels, k=config.num_best - len(last_step_best_kernels))

    last_step_best_kernel = last_step_best_kernels[work.sample_id % config.num_best] # use top config.num_best kernels
    last_step_best_kernel_src = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, last_step_best_kernel["sample_id"])

    prompt = prompt_idea_generation(ref_arch_src, last_step_best_kernel_src, last_step_best_kernel)

    idea = inference_server(prompt)

    prompt = prompt_refinement_from_idea(ref_arch_src, last_step_best_kernel_src, last_step_best_kernel, idea)
    return prompt


def generate_prompt(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, inference_server: callable, run_dir: str) -> str:
    match config.method:
        case "best-of-N":
            return generate_prompt_best_of_n(work, config, ref_arch_src, inference_server, run_dir)
        case "iterative refinement":
            return generate_prompt_iterative_refinement(work, config, ref_arch_src, inference_server, run_dir)
        case "METR":
            return generate_prompt_metr(work, config, ref_arch_src, inference_server, run_dir)
        case "Stanford":
            return generate_prompt_stanford(work, config, ref_arch_src, inference_server, run_dir)
        case _:
            raise ValueError(f"Invalid method: {config.method}")
