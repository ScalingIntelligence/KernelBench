import subprocess
import os, sys
from .utils import (
    query_server,
    read_file,
    extract_first_code,
)
from .dataset import (
    construct_problem_dataset_from_problem_dir
)
from .prompt_constructor import (
    prompt_generate_custom_cuda_from_file_one_example,
    prompt_generate_custom_cuda_oneshot_and_template,
    prompt_fix_compile,
    prompt_fix_correctness,
)
from .eval import (
    eval_kernel_against_ref,
    KernelExecResult,
    fetch_ref_arch_from_problem_id,
    fetch_ref_arch_from_level_problem_id,
)

from .dataset import get_kernelbench_subset

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
import subprocess
import os, sys
from .utils import (
    query_server,
    read_file,
    extract_first_code,
)
from .dataset import (
    construct_problem_dataset_from_problem_dir
)
from .prompt_constructor import (
    prompt_generate_custom_cuda_from_file_one_example,
    prompt_generate_custom_cuda_oneshot_and_template,
    prompt_fix_compile,
    prompt_fix_correctness,
)
from .eval import (
    eval_kernel_against_ref,
    KernelExecResult,
    fetch_ref_arch_from_problem_id,
    fetch_ref_arch_from_level_problem_id,
)

from .dataset import get_kernelbench_subset

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

SERVER_TYPE = "gemini"

server_args = {
    "deepseek": {"temperature": 1.6, "max_tokens": 4096},
    "gemini": {},  # need to experiment with temperature,
    "together": {  # this is Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "sglang": {  # this is Llama 3
        "temperature": 0.7,
    },
}


def run_llm(prompt, server_type=SERVER_TYPE, temperature=None):
    """
    query the LLM server with the prompt
    """
    if temperature is not None: # always override temperature
        server_args[server_type]["temperature"] = temperature
    return query_server(prompt, server_type=server_type, **server_args[server_type])


def get_temperature_sweep_generations(
    server_type, temperatures, level_num, problem_id, num_samples=30
):
    ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(
        level_num, problem_id, with_name=True
    )
    os.makedirs(os.path.join(REPO_TOP_PATH, f"results/"), exist_ok=True)
    os.makedirs(
        os.path.join(REPO_TOP_PATH, f"results/temperature_sweep/"), exist_ok=True
    )
    for temperature in temperatures:
        for sample_ind in range(num_samples):
            # save generation in results/temperature_sweep/server_type/level_num/problem_id/temp{temperature}_sample_{sample_ind}.txt
            file_path = os.path.join(
                REPO_TOP_PATH,
                f"results/temperature_sweep/{server_type}_level{level_num}_problem{problem_id}_temp{temperature}_sample_{sample_ind}.txt",
            )
            if os.path.exists(file_path):
                print(f"Skipping {file_path} because it already exists")
                continue
            prompt = prompt_generate_custom_cuda_from_file_one_example(
                ref_arch_src, example_ind=1
            )
            result = run_llm(prompt, server_type, temperature)
            with open(file_path, "w") as f:
                f.write(result)