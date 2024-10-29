import subprocess
import os, sys
from utils import query_server, read_file, extract_first_code, construct_problem_dataset_from_problem_dir
from prompt_constructor import prompt_generate_custom_cuda_from_file_one_example, prompt_generate_custom_cuda_oneshot_and_template, prompt_fix_compile, prompt_fix_correctness
from eval import eval_kernel_against_ref, KernelExecResult, fetch_ref_arch_from_problem_id

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

SERVER_TYPE = "gemini"

server_args = {
    "deepseek": {
        "temperature": 1.6,
        "max_tokens": 4096
    },
    "gemini": {}, # need to experiment with temperature,
    "together": { # this is Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096
    },
    "sglang": { # this is Llama 3
        "temperature": 0.7,
    }
}

def run_llm(prompt):
    '''
    Call use common API query function with monkeys
    '''
    return query_server(prompt, server_type=SERVER_TYPE
                        , **server_args[SERVER_TYPE])

def get_custom_cuda(prompt):
    custom_cuda = run_llm(prompt)
    custom_cuda = extract_first_code(custom_cuda, "python")
    return custom_cuda

def run(ref_arch_src, save_prompt=False, use_combined_prompt=False, prompt_example_ind=1) -> KernelExecResult:
    os.makedirs(os.path.join(REPO_TOP_PATH, "src/scratch"), exist_ok=True)

    # generate custom CUDA, save in scratch/model_new.py
    if use_combined_prompt:
        fn_get_prompt = prompt_generate_custom_cuda_oneshot_and_template
        custom_cuda_prompt = fn_get_prompt(ref_arch_src)
    else:
        fn_get_prompt = prompt_generate_custom_cuda_from_file_one_example
        custom_cuda_prompt = fn_get_prompt(ref_arch_src, prompt_example_ind)
    if save_prompt:
        with open(os.path.join(REPO_TOP_PATH, "src/scratch/prompt.txt"), "w") as f:
            f.write(custom_cuda_prompt)
    custom_cuda = get_custom_cuda(custom_cuda_prompt)

    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/model_new.py"), "w") as f:
        f.write(custom_cuda)

    kernel_exec_result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=False, measure_performance=False)
    return (custom_cuda, kernel_exec_result)

def compare_results(best_result, new_result):
    if best_result is None:
        return True
    if new_result.compiled and not best_result.compiled:
        return True
    if new_result.correctness and not best_result.correctness:
        return True
    # TODO: compare performance

def run_multiturn(ref_arch_src, turns=10) -> KernelExecResult:

    custom_cuda, result = run(ref_arch_src)
    best_custom_cuda = custom_cuda
    best_result = result

    for turn in range(turns):
        if not result.compiled:
            print(f"Turn {turn}: Fixing compilation error")
            print(f"Metadata: {result.metadata}")
            custom_cuda = get_custom_cuda(prompt_fix_compile(ref_arch_src, custom_cuda, result.metadata))
            result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=False, measure_performance=False)
        elif not result.correctness:
            print(f"Turn {turn}: Fixing correctness error")
            print(f"Metadata: {result.metadata}")
            custom_cuda = get_custom_cuda(prompt_fix_correctness(ref_arch_src, custom_cuda, result.metadata))
            result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=False, measure_performance=False)
        else:
            # TODO: we should try to improve performance
            # custom_cuda = get_custom_cuda(improve_perf_prompt)
            # result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=False, measure_performance=False)
            print(f"Turn {turn}: Improving performance")
        if compare_results(best_result, result):
            best_result = result
            best_custom_cuda = custom_cuda

    return (best_custom_cuda, best_result)


if __name__ == "__main__":

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level2")
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    ref_arch_src = fetch_ref_arch_from_problem_id(17, dataset)
    # write to scratch/model.py
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/model.py"), "w") as f:
        f.write(ref_arch_src)

    # run with one-shot + template combined prompt
    # print(run(ref_arch_src, use_combined_prompt=True))

    # # run with one-shot from file prompt
    # print(run(ref_arch_src, use_combined_prompt=False, prompt_example_ind=1))

    # # run with template prompt
    # print(run(ref_arch_src, use_combined_prompt=False, prompt_example_ind=2))
    
    # run multiturn with combined prompt
    print(run_multiturn(ref_arch_src))