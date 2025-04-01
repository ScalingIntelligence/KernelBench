import asyncio
import weave
import requests
from rich.console import Console
from litellm import acompletion
from pydantic import BaseModel, Field
from kernelbench.utils import read_file, extract_python_code
from kernelbench.prompt_constructor import (
    prompt_generate_prompt_with_hardware_info_from_template,
)
from io import BytesIO

GPU_NAME = "H100"
MODEL = "claude-3-5-sonnet-20240620"
ONE_SAMPLE_PATH = "KernelBench/level1/1_Square_matrix_multiplication_.py"
BENCHMARK_SERVER_URL = "https://tcapelle--kernel-benchmark-server-benchmarkservice-fastapi-app.modal.run/benchmark"
BENCHMARK_SERVER_PARAMS = {
    "num_correct_trials": "5",
    "num_perf_trials": "100",
    "verbose": "true",
}

console = Console()
ref_arch_src = read_file(ONE_SAMPLE_PATH)


console.rule("Generating prompt...")
# generate prompt
prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src, GPU_NAME)

console.print(prompt)


class ClaudeResponse(BaseModel):
    generated_code: str = Field(description="The optimized generated code")


@weave.op
async def generate_with_claude(prompt):
    response = await acompletion(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.5,
        response_format=ClaudeResponse,
    )
    response = ClaudeResponse.model_validate_json(response.choices[0].message.content)
    return response


weave.init("claude_cuda")

console.rule("Generating Kernel...")
response = asyncio.run(generate_with_claude(prompt))

console.print(response.generated_code)


@weave.op
def call_benchmark_server(
    ref_pytorch_code,
    optimized_code,
    benchmark_server_url=BENCHMARK_SERVER_URL,
    benchmark_server_params=BENCHMARK_SERVER_PARAMS,
):
    # Create in-memory file objects
    ref_file = BytesIO(ref_pytorch_code.encode("utf-8"))
    kernel_file = BytesIO(optimized_code.encode("utf-8"))

    # Prepare the files for the request
    files = {
        "ref_file": ("ref_file.py", ref_file),
        "kernel_file": ("kernel_file.py", kernel_file),
    }

    # Make the request with both files and data
    response = requests.post(
        benchmark_server_url, files=files, data=benchmark_server_params
    )

    # Return the response
    return response.json()


console.rule("Benchmarking...")
generated_code = extract_python_code(response.generated_code)
print(generated_code)
benchmark_result = call_benchmark_server(ref_arch_src, generated_code)
console.print(benchmark_result)
