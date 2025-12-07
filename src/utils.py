########################
# Utils Functions
########################

import multiprocessing
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env early so query_server can see them
import subprocess
import re
import random
import tempfile
from pathlib import Path
import re
import math
import os
import json
from tqdm import tqdm

# API clients
from openai import OpenAI
from litellm import completion

# from datasets import load_dataset
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import time
import shutil
import concurrent
from functools import cache
from transformers import AutoTokenizer
import hashlib

from concurrent.futures import ProcessPoolExecutor, as_completed

SGLANG_KEY = os.environ.get("SGLANG_API_KEY")


########################################################
# Inference Helpers
########################################################

def set_gpu_arch(arch_list: list[str]):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")
    
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

def query_server(
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant",  # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50, 
    max_tokens: int = 128,  # max output tokens to generate
    num_completions: int = 1,
    server_port: int = 30000,  # only for local server hosted on SGLang
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",  # specify model type

    # for reasoning models
    is_reasoning_model: bool = False, # indiactor of using reasoning models
    budget_tokens: int = 0, # for claude thinking
    reasoning_effort: str = None, # only for o1 and o3 / more reasoning models in the future
):
    """
    Query various sort of LLM inference API providers
    Done through liteLLM:
    - Local Server (SGLang, vLLM, Tokasaurus)
    """
    # Local Server (SGLang, vLLM, Tokasaurus) - special handling
    if server_type == "local":
        url = f"http://{server_address}:{server_port}"
        client = OpenAI(
            api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
        )
        if isinstance(prompt, str):
            response = client.completions.create(
                model="default",
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.text for choice in response.choices]
        else:
            response = client.chat.completions.create(
                model="default",
                messages=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.message.content for choice in response.choices]
        
        # output processing
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    
    # All other providers - use LiteLLM unified interface
    # Build messages list with system prompt first (if not already present)
    messages = []
    
    # Check if prompt is already a list with a system message
    if isinstance(prompt, list) and prompt and prompt[0].get("role") == "system":
        # Prompt already has system message, use it directly
        messages = prompt
    else:
        # Add system prompt first if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Then add the actual prompt
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)
    
    try:
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "n": num_completions,
        }
        
        # Reasoning models (o1, o3, etc.) don't support standard sampling params
        if is_reasoning_model:
            # Note: o1/o3 models don't support temperature, top_p, top_k
            # LiteLLM will pass through reasoning_effort for OpenAI o1/o3 models
            if reasoning_effort:
                completion_kwargs["reasoning_effort"] = reasoning_effort
            # Claude extended thinking uses "thinking" parameter with dict structure
            # Format: {"type": "enabled", "budget_tokens": <int>}
            if budget_tokens > 0 and "anthropic" in model_name.lower():
                completion_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        else:
            # Standard models support temperature and top_p
            completion_kwargs["temperature"] = temperature
            completion_kwargs["top_p"] = top_p
            
            # top_k is not supported by OpenAI models
            if "openai/" not in model_name.lower() and "gpt" not in model_name.lower():
                completion_kwargs["top_k"] = top_k
        
        response = completion(**completion_kwargs)
        
        # output processing
        if num_completions == 1:
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(f"LLM returned None content for model {model_name}. finish_reason: {response.choices[0].finish_reason}")
            return content
        else:
            contents = [choice.message.content for choice in response.choices]
            if any(c is None for c in contents):
                raise ValueError(f"LLM returned None content in one or more completions for model {model_name}")
            return contents
    except Exception as e:
        print(f"Error in query_server for model {model_name}: {e}")
        raise


# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6, 
        "model_name": "deepseek/deepseek-coder",
        "max_tokens": 4096
    },
    "google": {
        "model_name": "gemini/gemini-2.5-flash",
        "temperature": 0.7, # need to experiment with temperature
        "max_tokens": 16384,
    },
    "together": { # mostly for Llama 3.1
        "model_name": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "local": {  # this is for running locally (SGLang, vLLM, Tokasaurus), mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": 10210,
        "server_address": "matx2.stanford.edu",
        "max_tokens": 8192,
    },
    "anthropic": {  # for Claude 3.7 Sonnet
        "model_name": "anthropic/claude-3-7-sonnet-20250219",
        "temperature": 0.8,
        "max_tokens": 8192,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",
        # "model_name": "o1-preview-2024-09-12", # be careful with this one
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "fireworks": {
        "model_name": "fireworks_ai/llama-v3p1-70b-instruct",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
}


def create_inference_server_from_presets(server_type: str = None, 
                                         greedy_sample: bool = False,   
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         model_name: str = None,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()
        
        if model_name is not None and model_name != "None":
            server_args["model_name"] = model_name
        
        if kwargs:
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != "None"}
            server_args.update(filtered_kwargs)
        
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        
        if verbose:
            print(f"Querying server {server_type} with model {server_args['model_name']} and args: {server_args}")
        
        if time_generation:
            start_time = time.time()
            response = query_server(
                prompt, server_type=server_type, **server_args
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_server(
                prompt, server_type=server_type, **server_args
            )
    
    return _query_llm

"""
Model output processing
#  TODO: add unit tests
"""


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def print_messages(messages):
    for message in messages:
        print(message["role"])
        print(message["content"])
        print("-" * 50)
        print("\n\n")


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    if output_string is None:
        return None
    
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    '''
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)
    
    return " \n ".join(combined_code) if combined_code else ""


def extract_cuda_and_python_code(output_string: str) -> tuple[str | None, str | None]:
    """
    Extract both CUDA (C++) and Python code blocks from model output.
    Handles two cases:
    1. Separate code blocks (```cpp ... ``` and ```python ... ```)
    2. CUDA code embedded as string in Python
    
    Returns: (cuda_code, python_code) tuple
    - cuda_code: The CUDA/C++ code block (or None if not found)
    - python_code: The Python code block, converted to use separate module import (or None if not found)
    """
    if output_string is None:
        return (None, None)
    
    trimmed = output_string.strip()
    
    # First, try to find separate code blocks
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, trimmed, re.DOTALL)
    
    cuda_code = None
    python_code = None
    
    for lang_type, code in matches:
        code = code.strip()
        lang_type = lang_type.lower() if lang_type else ""
        
        # Check for CUDA/C++ code blocks
        if lang_type in ['cpp', 'cuda', 'c++', 'cu'] or (not lang_type and 'PYBIND11_MODULE' in code):
            if cuda_code is None:  # Take the first CUDA block found
                cuda_code = code
        
        # Check for Python code blocks
        elif lang_type == 'python' or (not lang_type and ('import torch' in code or 'class ModelNew' in code)):
            if python_code is None:  # Take the first Python block found
                python_code = code
    
    # If we didn't find separate blocks, try to extract from embedded string pattern
    if cuda_code is None or python_code is None:
        # Look for Python code block that might contain embedded CUDA
        python_block = None
        for lang_type, code in matches:
            code = code.strip()
            lang_type = lang_type.lower() if lang_type else ""
            if lang_type == 'python' or ('import torch' in code or 'class ModelNew' in code or 'load_inline' in code):
                python_block = code
                break
        
        # If no code blocks found, check the raw text
        if python_block is None and len(matches) == 0:
            # Try to find Python code in raw text
            if 'import torch' in trimmed or 'class ModelNew' in trimmed or 'load_inline' in trimmed:
                python_block = trimmed
        
        if python_block:
            # Try to extract CUDA code from string variables (e.g., conv2d_cuda_source = """...""")
            # Pattern: variable_name = """...CUDA code..."""
            cuda_string_pattern = r'(\w+_cuda_source|\w+_cpp_source|\w+_source)\s*=\s*"""(.*?)"""'
            cuda_matches = re.findall(cuda_string_pattern, python_block, re.DOTALL)
            
            if cuda_matches:
                # Take the first match (usually the CUDA source)
                var_name, cuda_content = cuda_matches[0]
                if cuda_code is None:
                    cuda_code = cuda_content.strip()
                
                # Convert Python code to use separate module import
                if python_code is None:
                    python_code = convert_python_from_inline_to_module(python_block, var_name)
            else:
                # If no embedded CUDA found but we have Python, use it as-is
                if python_code is None:
                    python_code = python_block
    
    return (cuda_code, python_code)


def convert_python_from_inline_to_module(python_code: str, cuda_var_name: str) -> str:
    """
    Convert Python code that uses load_inline to use separate module import pattern.
    Removes load_inline calls and related setup, replaces with tk_kernels import.
    """
    # Use regex to remove the CUDA source string variable definitions
    # Pattern: variable_name = """...""" (handles multi-line)
    cuda_source_pattern = r'\w+_(cuda|cpp)_source\s*=\s*""".*?"""'
    converted_code = re.sub(cuda_source_pattern, '', python_code, flags=re.DOTALL)
    
    # Remove cpp_source declarations too
    cpp_source_pattern = r'\w+_cpp_source\s*=\s*""".*?"""'
    converted_code = re.sub(cpp_source_pattern, '', converted_code, flags=re.DOTALL)
    
    # Remove load_inline calls (may span multiple lines)
    # Find load_inline( ... ) and remove it
    load_inline_pattern = r'\w+\s*=\s*load_inline\([^)]*\)'
    # Handle multi-line load_inline
    lines = converted_code.split('\n')
    new_lines = []
    skip_load_inline = False
    paren_depth = 0
    
    for line in lines:
        if 'load_inline' in line:
            skip_load_inline = True
            paren_depth = line.count('(') - line.count(')')
            if paren_depth <= 0:
                skip_load_inline = False
            continue
        
        if skip_load_inline:
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0:
                skip_load_inline = False
            continue
        
        new_lines.append(line)
    
    converted_code = '\n'.join(new_lines)
    
    # Remove imports related to load_inline
    converted_code = re.sub(r'from torch\.utils\.cpp_extension import.*?load_inline[^\n]*\n', '', converted_code)
    converted_code = re.sub(r'import.*?load_inline[^\n]*\n', '', converted_code)
    
    # Remove TK_PATH setup (usually only needed for load_inline)
    converted_code = re.sub(r'# ThunderKittens header-only library path\s*\nTK_PATH\s*=.*?\n', '', converted_code, flags=re.MULTILINE)
    converted_code = re.sub(r'TK_PATH\s*=.*?\n', '', converted_code)
    
    # Remove C++ source declaration comments
    converted_code = re.sub(r'# C\+\+ source declaration\s*\n', '', converted_code)
    converted_code = re.sub(r'# CUDA source with.*?\n', '', converted_code)
    converted_code = re.sub(r'# Compile kernel\s*\n', '', converted_code)
    
    # Add import for tk_kernels (after torch imports)
    if 'import torch' in converted_code and 'import tk_kernels' not in converted_code:
        # Insert after the last import statement
        import_section = re.search(r'(import torch[^\n]*\n(?:import torch[^\n]*\n)*)', converted_code)
        if import_section:
            converted_code = converted_code.replace(import_section.group(1), import_section.group(1) + 'import tk_kernels\n')
        else:
            # Fallback: add after first import
            converted_code = re.sub(r'(import torch[^\n]*\n)', r'\1import tk_kernels\n', converted_code, count=1)
    
    # Replace references to the load_inline result variable
    # Pattern: self.conv2d_op = conv2d_tk -> remove or replace
    # Pattern: conv2d_tk.conv2d_cuda(...) -> tk_kernels.dispatch_*(...)
    # This is tricky without knowing the exact function names, so we'll leave a comment
    converted_code = re.sub(r'self\.\w+_op\s*=\s*\w+_tk\s*\n', '', converted_code)
    
    # Clean up extra blank lines
    converted_code = re.sub(r'\n\n\n+', '\n\n', converted_code)
    
    return converted_code.strip()

################################################################################
# Scale up experiments in parallel
################################################################################

def maybe_multithread(func, instances, num_workers, time_interval=0.0, *shared_args, **shared_kwargs):
    """
    Multithreaded execution of func, with optional time interval between queries
    Ideal for querying LLM APIs, does not provide process isolation
    """
    output_data = []
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

                # Submit tasks one at a time with delay between them
                futures = []
                for instance in instances:
                    futures.append(
                        executor.submit(
                            func,
                            instance,
                            *shared_args,
                            **shared_kwargs
                        )
                    )
                    time.sleep(time_interval)  # sleep between submitting each task



                # Wait for each future to complete
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            output_data.append(result)
                    except Exception as e:
                        print("Got an error!", e)
                        continue
    else:
        for instance in tqdm(instances):
            output = func(instance, *shared_args, **shared_kwargs)
            if output is not None: output_data.append(output)

    return output_data


def maybe_multiprocess_cuda(
    func, instances, num_workers, *shared_args, **shared_kwargs
):
    """
    From monkeys, but modified to work with CUDA
    """
    output_data = []
    multiprocessing.set_start_method(
        "spawn", force=True
    )  # this is necessary for CUDA to work

    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(func, instance, *shared_args, **shared_kwargs): None
                for instance in instances
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    if result is not None:
                        output_data.append(result)
                except Exception as e:
                    print("Got an error!", e)
                    continue
    return output_data

# src/random_inputs.py
import os, torch, itertools
from torch.distributions import Normal, Uniform, Laplace, Exponential, LogNormal

# Pick which distributions are allowed in “random” mode.
_DEFAULT_RANDOM_POOL = (
    ("normal",      lambda shape: Normal(0, 1).sample(shape)),
    ("uniform",     lambda shape: Uniform(-1, 1).sample(shape)),
    ("laplace",     lambda shape: Laplace(0, 1).sample(shape)),
    ("exponential", lambda shape: Exponential(1).sample(shape)),   # strictly >0
    ("lognormal",   lambda shape: LogNormal(0, 1).sample(shape)),  # strictly >0
)


def sample(shape, mode="random"):
    """
    shape : torch.Size or tuple
    mode  : "random"  – draw from a rotating pool of distributions
            "target"  – return a tensor from a randomly chosen edge-case pattern
            <dist>    – force a single distribution name, e.g. "laplace"
    """
    if mode == "random":
        # Round-robin through default pool
        idx = int(torch.empty((), dtype=torch.int64).random_()) % len(_DEFAULT_RANDOM_POOL)
        _, fn = _DEFAULT_RANDOM_POOL[idx]
        return fn(shape)

    # Explicit distribution name
    pool = dict(_DEFAULT_RANDOM_POOL)
    if mode not in pool:
        raise ValueError(f"Unknown distribution {mode}")
    return pool[mode](shape)


# ------------------------------------------------------------------
# Public helper: rand_mix / rand_mix_like
# ------------------------------------------------------------------

def rand_mix(*size, dist: str = "random", device=None, dtype=None, requires_grad: bool = False):
    """Return a tensor drawn from a chosen distribution (or randomly chosen).

    Parameters
    ----------
    *size : int or tuple
        Dimensions of the output tensor (same semantics as ``torch.randn``).
    dist : str, optional
        • "random"   – randomly cycle through the default pool defined above.
        • "target"   – pick from the specialised _TARGETED_CASES pool.
        • any key in the default pool ("normal", "uniform", "laplace", ...).
    device, dtype, requires_grad : any
        Forwarded to ``Tensor.to`` / ``Tensor.requires_grad_`` for convenience.
    """
    # normalise *size → shape tuple
    shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, torch.Size)) else size

    t = sample(shape, mode=dist)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    if requires_grad:
        t.requires_grad_(True)
    return t

def rand_mix_like(tensor: torch.Tensor, dist: str = "random", **kwargs):
    """rand_mix variant that infers shape from *tensor*."""
    return rand_mix(*tensor.shape, dist=dist, **kwargs)

# Register convenience aliases under torch namespace (does not shadow existing fns)
setattr(torch, "rand_mix", rand_mix)
setattr(torch, "rand_mix_like", rand_mix_like)