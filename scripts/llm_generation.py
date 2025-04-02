import asyncio
import weave
from weave.trace.context import call_context
import requests
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console
import litellm
from litellm import acompletion
from pydantic import BaseModel, Field
import simple_parsing as sp
from kernelbench.utils import read_file, extract_python_code
from kernelbench.prompt_constructor import (
    prompt_generate_prompt_with_hardware_info_from_template,
)
from io import BytesIO


litellm.drop_params = True  # drop params that are not used by the model

BENCHMARK_SERVER_PARAMS: dict = {
    "num_correct_trials": "5",
    "num_perf_trials": "100",
    "verbose": "true",
}


@dataclass
class ScriptArgs:
    gpu_name: str = "H100"
    # MODEL = "claude-3-5-sonnet-20240620"
    model: str = "gpt-4o"
    dataset_folder: str = "KernelBench/level1"
    benchmark_server_url: str = (
        "https://tcapelle--kernel-benchmark-server-benchmarkservice-fastapi-app.modal.run/benchmark"
    )
    debug: bool = False
    N: int = 5


args = sp.parse(ScriptArgs)

console = Console()

console.rule("Loading Dataset...")
ds = [
    {"ref_fname": str(sf), "ref_code": read_file(str(sf)), "gpu_name": args.gpu_name}
    for sf in Path(args.dataset_folder).glob("*.py")
]


ds = ds[: args.N]

console.print(f"Loaded {len(ds)} kernels")


class LLMResponse(BaseModel):
    generated_code: str = Field(description="The optimized generated code")


class LLMCuda(weave.Model):
    model: str = args.model
    temperature: float = 0.5
    max_tokens: int = 2000

    @weave.op
    def prepare_prompt(self, filename: str, gpu_name: str):
        ref_arch_src = read_file(filename)
        prompt = prompt_generate_prompt_with_hardware_info_from_template(
            ref_arch_src, gpu_name
        )
        return prompt

    @weave.op
    async def generate_with_llm(self, prompt) -> LLMResponse:
        response = await acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates optimized CUDA code for a given PyTorch function. You Reply in JSON format,",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=LLMResponse,
        )
        response = LLMResponse.model_validate_json(response.choices[0].message.content)
        return response

    @weave.op
    async def predict(self, ref_fname: str, gpu_name: str) -> LLMResponse:
        prompt = self.prepare_prompt(ref_fname, gpu_name)
        response = await self.generate_with_llm(prompt)
        return response


weave.init("claude_cuda")

claude = LLMCuda()


@weave.op
def call_benchmark_server(
    ref_pytorch_code,
    optimized_code,
    benchmark_server_url=args.benchmark_server_url,
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

    # Add debugging info
    if args.debug:
        console.print(f"Status code: {response.status_code}")
        console.print(
            f"Response content: {response.content[:500]}"
        )  # Showing first 500 chars

    # Check for successful response before parsing JSON
    if response.status_code != 200:
        return {
            "error": f"Server error: {response.status_code}",
            "content": str(response.content),
        }

    # Try to parse JSON with better error handling
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "Invalid JSON response", "content": str(response.content)}


@weave.op
def score_kernel(output: LLMResponse, ref_code: str) -> dict:
    extracted_code = extract_python_code(output.generated_code)
    benchmark_result = call_benchmark_server(ref_code, extracted_code)
    error = benchmark_result.get("error", None)
    if error is not None:
        return {
            "compiled": False,
            "correctness": False,
            "speedup_vs_compile": 0,
            "speedup_vs_eager": 0,
            "error": benchmark_result.get("content", str(error)),
        }

    # Handle missing keys safely with .get() and provide defaults
    kernel_result = benchmark_result.get("kernel_result", {})
    return {
        "compiled": kernel_result.get("compiled", False),
        "correctness": kernel_result.get("correctness", False),
        "speedup_vs_compile": benchmark_result.get("speedup_vs_compile", 0),
        "speedup_vs_eager": benchmark_result.get("speedup_vs_eager", 0),
        "error": benchmark_result.get("error", None),
    }


if args.debug:
    console.rule("Running one sample...")
    one_sample = ds[0]
    console.print(f"One sample: {one_sample}")
    response = asyncio.run(
        claude.predict(one_sample["ref_fname"], one_sample["gpu_name"])
    )
    console.print(f"Response: {response}")
    score = score_kernel(response, one_sample["ref_code"])
    console.print(f"Score: {score}")

else:
    console.rule("Running Evaluation...")
    evaluation = weave.Evaluation(dataset=ds, scorers=[score_kernel])
    asyncio.run(evaluation.evaluate(claude))
