import subprocess
import sys, os
import src.utils as utils

from src.prompt_constructor import prompt_generate_custom_cuda_from_file_one_example

# DEFINITION
SERVER_TYPE = "anthropic"

server_args = {
    "deepseek": {"temperature": 1.6, "max_tokens": 4096},
    "gemini": {},  # need to experiment with temperature,
    "together": {  # this is Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "sglang": {  # this is Llama 3 ran locally
        "temperature": 0.7,
    },
    "anthropic": {  # for Claude 3.5 Sonnet
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.8,
        "max_tokens": 4096,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "openai_o1": { # this could be expensive!! Be careful
        "model_name": "o1-preview-2024-09-12",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "google": {
        "model_name": "gemini-1.5-flash-002",
        "temperature": 0.7,
        "max_tokens": 8192,
    },
}


def run_llm(prompt: str | list[dict], greedy_sample: bool = False):
    """
    Simple wrapper to query API server
    """
    if greedy_sample:
        server_args[SERVER_TYPE]["temperature"] = 0.0
    return utils.query_server(
        prompt, server_type=SERVER_TYPE, **server_args[SERVER_TYPE]
    )


############################################
# Run Script
############################################
def run(arch_path):
    # read in an architecture file, copy it to ./scratch/model.py
    arch = utils.read_file(arch_path)
    # Ensure the ./scratch directory exists
    os.makedirs("./scratch", exist_ok=True)

    # Write the architecture to ./scratch/model.py
    with open("./scratch/model.py", "w") as f:
        f.write(arch)

    # # generate test harness, save in ./scratch/test.py
    # test_harness = query_server(prompt_generate_test_harness_from_file(arch_path), server_type=SERVER_TYPE)
    # test_harness = extract_first_code(test_harness, "python")
    # with open("./scratch/test.py", "w") as f:
    #     f.write(test_harness)
    # generate custom CUDA, save in ./scratch/model_new.py

    fn_get_prompt = prompt_generate_custom_cuda_from_file_one_example
    custom_cuda_prompt = fn_get_prompt(arch, 1)
    with open(f"./scratch/prompt_{SERVER_TYPE}.py", "w") as f:
        f.write(custom_cuda_prompt)
    # custom_cuda_prompt = prompt_generate_custom_cuda_from_file(arch_path)
    custom_cuda = run_llm(custom_cuda_prompt)

    # import pdb; pdb.set_trace()
    custom_cuda = utils.extract_first_code(custom_cuda, "python")
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    print(
        "[Verification] Torch moduel with Custom CUDA code **GENERATED** successfully"
    )

    with open(f"./scratch/model_new_{SERVER_TYPE}.py", "w") as f:
        f.write(custom_cuda)
    # check if the generated code compiles
    try:
        exec(custom_cuda)
        print("[Verification] Custom CUDA code **COMPILES** successfully")
    except Exception as e:
        raise RuntimeError(f"Error compiling generated custom cuda code: {e}")

    # check generated code is correct / functionally equivalent.
    # run test harness, save output in log.txt
    # with open('./scratch/log.txt', 'w') as log_file:
    #     process = subprocess.run(
    #         ['python', './scratch/test.py'],
    #         stdout=log_file,  # Redirect stdout to log.txt
    #         stderr=subprocess.STDOUT  # Redirect stderr to log.txt as well
    #     )
    #     if process.returncode == 0:
    #         print("[Verification] Custom CUDA kernel is **Correct**, matches reference")
    #         return "PASS"
    #     else:
    #         print("[Verification] Custom CUDA kernel **FAIL** to match reference in terms of correctness")
    #         return "FAIL"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arch_path = sys.argv[1]
    else:
        arch_path = "./KernelBench/level1/1_Square_matrix_multiplication_.py"

    print(run(arch_path))
