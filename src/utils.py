########################
# Utils
# could cut this down more
########################


import multiprocessing
import subprocess
import re
from openai import OpenAI
import google.generativeai as genai
import random
import tempfile
from pathlib import Path
import re

import math
import os
import json
import pickle
from together import Together
from tqdm import tqdm
#from datasets import load_dataset
import anthropic
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



# Define API key access
TOGETHER_KEY = os.environ.get('TOGETHER_API_KEY')  
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY')
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_KEY = os.environ.get('GEMINI_API_KEY')
SGLANG_KEY = os.environ.get('SGLANG_API_KEY') # for Local Deployment
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY')

@cache
def load_deepseek_tokenizer():
    return AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-Coder-V2-Instruct-0724')

# Buffer because deepseek totally blocks us if we send stuff that's too long :(
TOO_LONG_FOR_DEEPSEEK = 115_000
def is_safe_to_send_to_deepseek(prompt):
    tokenizer = load_deepseek_tokenizer()
    if type(prompt) == str:
        return len(tokenizer(prompt, verbose=False)['input_ids']) < TOO_LONG_FOR_DEEPSEEK
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK


def query_server(
    prompt: str | list[dict], # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant", # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1, # nucleus sampling
    max_tokens: int = 128, # max output tokens to generate
    num_completions: int = 1,  
    server_port: int = 30000, # only for local server hosted on SGLang
    server_type: str = "sglang", 
    model_name: str = "default" # specify model type
):
    """
    Query various sort of LLM inference API providers
    Supports:
    - Deepseek
    - Together
    - Anthropic
    - Gemini (TODO)
    - OpenAI
    - SGLang (Local Server)
    """
    # First pass: select model and client based on arguments
    match server_type:
        case "sglang":
            url = f"http://localhost:{server_port}"
            client = OpenAI(
                api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
            )
            model = "default"
        case "deepseek":
            client = OpenAI(
                api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com", timeout=10000000, max_retries=3
            )
            model = "deepseek-coder" # only set to do this for now
            if not is_safe_to_send_to_deepseek(prompt):
                raise RuntimeError("Prompt is too long for DeepSeek")
        case "anthropic":
            client = anthropic.Anthropic(
                api_key=ANTHROPIC_KEY,
            )
            model = model_name
        case "google":
            genai.configure(api_key=GEMINI_KEY)

        case "together":
            client = Together(api_key=TOGETHER_KEY)
            model = model_name
        case "openai":
            client = OpenAI(api_key=OPENAI_KEY)
            model = model_name
        case _:
            raise NotImplementedError
    
    if server_type != "google":
        assert client is not None, "Client is not set, cannot proceed to generations"

    if server_type == "anthropic":
        assert type(prompt) == str
        assert model=="claude-3-5-sonnet-20241022", "Only test this version of Claude for now"
        print(f"Querying Anthropic {model} with temp {temperature} max tokens {max_tokens}")

        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        outputs = [choice.text for choice in response.content]
    
    elif server_type == "google":

        assert model_name == "gemini-1.5-flash-002", "Only test this for now"
        print(f"Querying Gemini {model_name} ... with temp {temperature} max tokens {max_tokens}")

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }

        assert model_name == "gemini-1.5-flash-002", "Only test this for now"

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )

        response = model.generate_content(prompt)

        return response.text
    
    elif server_type == "deepseek":
        assert model=="deepseek-coder", "Only test this for now" 
        print(f"Querying DeepSeek ... with temp {temperature} max tokens {max_tokens}")

        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "openai":
        # Don't use o1 unless for baseline, as it is expensive
        assert model=="gpt-4o-2024-08-06", "Only test this for now"
        print(f"Querying OpenAI {model} ... with temp {temperature} max tokens {max_tokens}")
        if model == "o1-preview-2024-09-12":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                # n=num_completions,
                # max_tokens=max_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "together":
        assert model=="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" or "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Only test this for now" 
        print(f"Querying Together {model} with temp {temperature} max tokens {max_tokens}")
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
            top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>","<|eom_id|>"],
            # truncate=32256,
            stream=False
        )
        outputs = [choice.message.content for choice in response.choices]

    else:  
        if type(prompt) == str:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p
            )
            outputs = [choice.text for choice in response.choices]
        else:
            print("Chat prompt")
            # print("Temperature:",temperature)
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p
            )
            outputs = [choice.message.content for choice in response.choices]


    # output processing
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs



"""
Model output processing
#  TODO: add unit tests
"""

def read_file(file_path) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
    
def print_messages(messages):
    for message in messages:
        print(message["role"])
        print(message["content"])
        print("-"*50)
        print("\n\n")

def extract_python_code(text):
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return '\n'.join(matches) if matches else ""

def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code

def extract_first_code(output_string: str, code_language_type: str) -> str:
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out

        if code.startswith(code_language_type):
            code = code[len(code_language_type) :].strip()

        return code

    return None

def construct_problem_dataset_from_problem_dir(problem_dir: str) -> list[str]:
    """
    Construct a list of relative paths to all the python files in the problem directory
    Sorted by the numerical prefix of the filenames
    """
    DATASET = []

    for file_name in os.listdir(problem_dir):
        if file_name.endswith(".py"):
            # Construct the path starting with "CUDABench/..."
            # relative_path = os.path.join("KernelBenchInternal", file_name)
            relative_path = os.path.join(problem_dir, file_name)
            DATASET.append(relative_path)

    # Sort the DATASET based on the numerical prefix of the filenames
    DATASET.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

    return DATASET



def maybe_multithread(func, instances, num_workers, *shared_args, **shared_kwargs):
    output_data = []
    if num_workers not in [1, None]:
        with tqdm(total=len(instances), smoothing=0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Create a future for running each instance
                futures = {
                    executor.submit(
                        func,
                        instance,
                        *shared_args,
                        **shared_kwargs
                    ): None
                    for instance in instances
                }
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


def maybe_multiprocess_cuda(func, instances, num_workers, *shared_args, **shared_kwargs):
    """
    From monkeys, but modified to work with CUDA
    """
    output_data = []
    multiprocessing.set_start_method('spawn', force=True)  # this is necessary for CUDA to work

    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    func,
                    instance,
                    *shared_args,
                    **shared_kwargs
                ): None
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
