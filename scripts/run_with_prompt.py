import sys, os
import src.utils as utils
import time
from src.prompt_constructor import prompt_generate_custom_cuda_from_file_one_example

"""
For quickly iterate on prompts

Uses functions in prompt_constructor
"""

# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6, 
        "model_name": "deepseek",
        "max_tokens": 4096
    },
    "google": {
        "model_name": "gemini-1.5-flash-002",
        "temperature": 0.7, # need to experiment with temperature
        "max_tokens": 8192,
    },
    "together": { # mostly for Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "sglang": {  # this is for running locally, mostly for Llama
        "temperature": 0.7,
    },
    "anthropic": {  # for Claude 3.5 Sonnet
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.8,
        "max_tokens": 4096,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",
        # "model_name": "o1-preview-2024-09-12", # be careful with this one
        "temperature": 0.0,
        "max_tokens": 4096,
    },
}


def create_inference_server_from_presets(server_type: str = None, 
                                         greedy_sample: bool = False,   
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0

        if verbose:
            print(f"Querying server {server_type} with args: {server_args}")
        
        if time_generation:
            start_time = time.time()
            response = utils.query_server(
                prompt, server_type=server_type, **server_args
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return utils.query_server(
                prompt, server_type=server_type, **server_args
            )
    
    return _query_llm


############################################
# Run Script
############################################
def inference_with_prompt(arch_path, inference_server: callable = None, log_to_local: bool = False) -> str:
    """
    Returns the generated custom CUDA code (kernel to evaluate)
    """
    # read in an architecture file, copy it to ./scratch/model.py
    arch = utils.read_file(arch_path)

    if log_to_local:
        # Ensure the ./scratch directory exists
        os.makedirs("./scratch", exist_ok=True)

        # Write the architecture to ./scratch/model.py
        with open("./scratch/model.py", "w") as f:
            f.write(arch)

    # generate custom CUDA, save in ./scratch/model_new.py
    # this is the part you could iterate on 
    fn_get_prompt = prompt_generate_custom_cuda_from_file_one_example
    custom_cuda_prompt = fn_get_prompt(arch, 1)
    if log_to_local:    
        with open(f"./scratch/prompt.py", "w") as f:
            f.write(custom_cuda_prompt)
    # custom_cuda_prompt = prompt_generate_custom_cuda_from_file(arch_path)
    custom_cuda = inference_server(custom_cuda_prompt)

    custom_cuda = utils.extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    print(
        "[Verification] Torch moduel with Custom CUDA code **GENERATED** successfully"
    )

    if log_to_local:
        with open(f"./scratch/model_new.py", "w") as f:
            f.write(custom_cuda)

    return custom_cuda


def test_inference(inference_server: callable):
    """
    Simple fucntion to intiiate call to server
    """

    start_time = time.time()
    lm_response = inference_server("Hello, world!")
    end_time = time.time()
    print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
    print(lm_response)


if __name__ == "__main__":

    inference_server = create_inference_server_from_presets(server_type="together",
                                                        model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                                                        greedy_sample=True,
                                                        verbose=True, 
                                                        time_generation=True)

    # test_inference(inference_server)
    if len(sys.argv) > 1:
        arch_path = sys.argv[1]
    else:
        # run from KernelBench top level directory
        arch_path = "./KernelBench/level1/1_Square_matrix_multiplication_.py"
    
    inference_with_prompt(arch_path, inference_server, log_to_local=True)
