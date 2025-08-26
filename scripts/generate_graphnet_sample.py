# generate_graphnet_sample.py
import os
import pydra
from pydra import Config, REQUIRED

from src.utils import (
    read_file,
    extract_first_code,
    set_gpu_arch,
    create_inference_server_from_presets,
)
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

class GenerateConfig(Config):
    def __init__(self):
        # Path to the Python file defining the graph (e.g., convnext_base/model.py)
        self.problem_path = REQUIRED

        # LLM generation settings
        self.server_type = "deepseek"
        self.model_name = "deepseek-chat"
        self.max_tokens = 4096
        self.temperature = 0.0
        self.gpu_arch = ["Hopper"]

        # Logging options
        self.logdir = os.path.join(REPO_DIR, "../results/gen_logs")
        self.log_prompt = True
        self.log_generated_kernel = True
        self.verbose = False


@pydra.main(base=GenerateConfig)
def main(config: GenerateConfig):
    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)

    os.makedirs(config.logdir, exist_ok=True)

    # Read reference architecture
    ref_arch_src = read_file(config.problem_path)

    # Create inference server
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True,
    )

    # Construct prompt and generate code
    prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    if config.log_prompt:
        with open(os.path.join(config.logdir, "prompt.txt"), "w") as f:
            f.write(prompt)

    custom_cuda = inference_server(prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.log_generated_kernel:
        with open(os.path.join(config.logdir, "generated_kernel.py"), "w") as f:
            f.write(custom_cuda)

    print("Generated CUDA kernel:\n")
    print(custom_cuda)

if __name__ == "__main__":
    main()
