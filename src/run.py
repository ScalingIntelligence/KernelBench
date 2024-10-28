import subprocess
import os, sys
from utils import query_server, read_file, extract_first_code, construct_problem_dataset_from_problem_dir
from prompt_constructor import prompt_generate_custom_cuda_from_file, prompt_generate_custom_cuda_from_file_save
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

def run(ref_arch_src, save_prompt=True, prompt_example_ind=1) -> KernelExecResult:

    # generate custom CUDA, save in scratch/model_new.py
    fn_get_prompt = prompt_generate_custom_cuda_from_file_save if save_prompt else prompt_generate_custom_cuda_from_file
    custom_cuda_prompt = fn_get_prompt(ref_arch_src, prompt_example_ind)
    custom_cuda = run_llm(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, "python")
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/model_new.py"), "w") as f:
        f.write(custom_cuda)

    kernel_exec_result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=False, measure_performance=False)
    return kernel_exec_result

if __name__ == "__main__":

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level1")
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    ref_arch_src = fetch_ref_arch_from_problem_id(1, dataset)
    # write to scratch/model.py
    with open(os.path.join(REPO_TOP_PATH, "src/scratch/model.py"), "w") as f:
        f.write(ref_arch_src)
    print(run(ref_arch_src))