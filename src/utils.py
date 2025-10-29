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

# Unified LLM Interface
from litellm import completion

# Legacy API clients (kept for backwards compatibility during migration)
# from together import Together
from openai import OpenAI  # Still needed for OpenAI-compatible endpoints (SGLang, Fireworks, etc.)
# import google.generativeai as genai
# import anthropic

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

# API Keys
# Note: LiteLLM automatically reads API keys from environment variables:
# - OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
# We only explicitly load SGLANG_KEY for local server handling
SGLANG_KEY = os.environ.get("SGLANG_API_KEY", "EMPTY")  # for Local Deployment

# Legacy keys (no longer directly used, kept for backwards compatibility)
# LiteLLM reads these automatically from environment variables
# TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
# DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
# OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
# GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
# ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
# SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
# FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")


########################################################
# Inference Helpers
########################################################

@cache
def load_deepseek_tokenizer():
    # TODO: Should we update this for new deepseek? Same tokenizer?
    # return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Instruct-0724")
    return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)

# Buffer because deepseek totally blocks us if we send stuff that's too long :(
TOO_LONG_FOR_DEEPSEEK = 115_000


def is_safe_to_send_to_deepseek(prompt):
    tokenizer = load_deepseek_tokenizer()
    # print(f"Prompt: {len(prompt)}")
    # print(f"Prompt length: {len(tokenizer(prompt, verbose=False)['input_ids'])}")
    
    if type(prompt) == str:
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK

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
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    max_tokens: int = 128,
    num_completions: int = 1,
    server_port: int = 30000,  # only for local SGLang server
    server_address: str = "localhost",
    server_type: str = "sglang",  # kept for SGLang compatibility
    model_name: str = "default",
    
    # for reasoning models
    is_reasoning_model: bool = False,
    budget_tokens: int = 0,  # for claude thinking
    reasoning_effort: str = None,  # for o1/o3 models
):
    """
    Query LLM inference providers using LiteLLM's unified interface.
    
    LiteLLM handles routing to different providers based on the model_name format:
    - "openai/gpt-4o" -> OpenAI
    - "deepseek/deepseek-chat" -> DeepSeek
    - "anthropic/claude-..." or "claude-..." -> Anthropic
    - "gemini/..." -> Google Gemini
    - "together_ai/..." -> Together AI
    - "sambanova/..." -> Sambanova
    - "fireworks_ai/..." -> Fireworks AI
    
    Special case: SGLang (local server) uses custom OpenAI client
    """
    
    # Special handling for SGLang (local server with OpenAI-compatible API)
    if server_type == "sglang":
        url = f"http://{server_address}:{server_port}"
        client = OpenAI(
            api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
        )
        
        # Format messages for OpenAI-compatible API
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    # For all other providers, use LiteLLM's unified interface
    # Format messages for LiteLLM (OpenAI format)
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt
    
    # Check for DeepSeek-specific token limit
    if "deepseek" in model_name.lower():
        if not is_safe_to_send_to_deepseek(messages if isinstance(prompt, str) else prompt):
            raise RuntimeError("Prompt is too long for DeepSeek")
    
    # Prepare kwargs for litellm.completion()
    completion_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": num_completions,
    }
    
    # Handle reasoning models (o1, o3, claude-thinking, deepseek-reasoner)
    if is_reasoning_model:
        if "o1" in model_name or "o3" in model_name:
            # OpenAI reasoning models
            if reasoning_effort:
                completion_kwargs["reasoning_effort"] = reasoning_effort
            # Remove system message for o1/o3 (they don't support it)
            completion_kwargs["messages"] = [msg for msg in messages if msg["role"] != "system"]
        elif "claude" in model_name and budget_tokens > 0:
            # Anthropic thinking (extended thinking)
            completion_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        elif "deepseek-reasoner" in model_name:
            # DeepSeek reasoner doesn't use temperature/top_p
            completion_kwargs.pop("temperature", None)
            completion_kwargs.pop("top_p", None)
    
    # Only add top_k if the provider supports it
    # LiteLLM will handle this, but we can pass it for providers that support it
    if top_k != 50:  # Only include if non-default
        completion_kwargs["top_k"] = top_k
    
    # Make the LiteLLM call
    try:
        response = completion(**completion_kwargs)
    except Exception as e:
        print(f"Error calling LiteLLM with model {model_name}: {e}")
        raise
    
    # Extract outputs from response (OpenAI format)
    outputs = [choice.message.content for choice in response.choices]
    
    # Return single output or list based on num_completions
    return outputs[0] if len(outputs) == 1 else outputs


# a list of presets for API server configs
# LiteLLM format: "provider/model-name" (LiteLLM handles routing based on the prefix)
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 1.6, 
        "model_name": "deepseek/deepseek-chat",  # LiteLLM format
        "max_tokens": 4096
    },
    "deepseek-coder": {
        "temperature": 1.6, 
        "model_name": "deepseek/deepseek-coder",  # LiteLLM format
        "max_tokens": 4096
    },
    "google": {
        "model_name": "gemini/gemini-1.5-flash-002",  # LiteLLM format
        "temperature": 0.7,
        "max_tokens": 8192,
    },
    "together": {  # for Llama 3.1 via Together AI
        "model_name": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # LiteLLM format
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "sglang": {  # this is for running locally, mostly for Llama
        # SGLang is a local server - kept for special handling
        "temperature": 0.8,
        "server_port": 10210,
        "server_address": "matx2.stanford.edu",
        "max_tokens": 8192,
        "model_name": "hosted",  # Will use custom OpenAI client for local server
    },
    "anthropic": {  # for Claude 3.5 Sonnet
        "model_name": "claude-3-5-sonnet-20241022",  # LiteLLM supports both anthropic/ prefix or direct model name
        "temperature": 0.8,
        "max_tokens": 4096,
    },
    "openai": {
        "model_name": "gpt-4o-2024-08-06",  # LiteLLM supports both openai/ prefix or direct model name
        # "model_name": "o1-preview-2024-09-12", # be careful with this one
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "sambanova": {
        "model_name": "sambanova/Meta-Llama-3.1-405B-Instruct",  # LiteLLM format
        "temperature": 0.1,
        "max_tokens": 8192,
    },
    "fireworks": {
        "model_name": "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",  # LiteLLM format
        "temperature": 0.7,
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
    Return a callable function that queries LLM with given settings using LiteLLM.
    
    Args:
        server_type: Key from SERVER_PRESETS (e.g., "deepseek", "openai", "anthropic")
        greedy_sample: If True, use temperature=0.0 for deterministic output
        verbose: If True, print query details
        time_generation: If True, print timing information
        **kwargs: Override any preset parameters (e.g., model_name, temperature, max_tokens)
    
    Returns:
        Callable that takes a prompt and returns the model response
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[server_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Querying {server_type} with LiteLLM model: {server_args.get('model_name', 'N/A')}")
            print(f"  Args: temp={server_args.get('temperature')}, max_tokens={server_args.get('max_tokens')}")
        
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