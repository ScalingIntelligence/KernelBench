########################
# API LLM Utils Functions
########################

import os

# API clients
from together import Together
from openai import OpenAI
import google.generativeai as genai
import anthropic

# from datasets import load_dataset
import time
from functools import cache
from transformers import AutoTokenizer


# Define API key access
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")  # for Local Deployment
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")


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
    
    if isinstance(prompt, str):
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK

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
    Supports:
    - OpenAI
    - Deepseek
    - Together
    - Sambanova
    - Anthropic
    - Gemini / Google AI Studio
    - Fireworks (OpenAI compatbility)
    - SGLang (Local Server)
    """
    # Select model and client based on arguments
    match server_type:
        case "sglang":
            url = f"http://{server_address}:{server_port}"
            client = OpenAI(
                api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
            )
            model = "default"
        case "deepseek":
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name
            assert model in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"], "Only support deepseek-chat or deepseek-coder for now"
            if not is_safe_to_send_to_deepseek(prompt):
                raise RuntimeError("Prompt is too long for DeepSeek")
        case "fireworks":
            client = OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "anthropic":
            client = anthropic.Anthropic(
                api_key=ANTHROPIC_KEY,
            )
            model = model_name
        case "google":
            genai.configure(api_key=GEMINI_KEY)
            model = model_name
        case "together":
            client = Together(api_key=TOGETHER_KEY)
            model = model_name
        case "sambanova":
            client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
            model = model_name
            
        case "openai":
            client = OpenAI(api_key=OPENAI_KEY)
            model = model_name
        case _:
            raise NotImplementedError

    if server_type != "google":
        assert client is not None, "Client is not set, cannot proceed to generations"
    else:
        print(
            f"Querying {server_type} {model} with temp {temperature} max tokens {max_tokens}"
        )
    # Logic to query the LLM
    if server_type == "anthropic":
        assert isinstance(prompt, str), f"The prompt must be a string for Anthropic, but it was a {type(prompt)}"

        if is_reasoning_model:
            # Use beta endpoint with thinking enabled for reasoning models
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                # Claude thinking requires budget_tokens for thinking (reasoning)
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        else:
            # Use standard endpoint for normal models
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
        outputs = [choice.text for choice in response.content if not hasattr(choice, 'thinking') or not choice.thinking]

    elif server_type == "google":
        # assert model_name == "gemini-1.5-flash-002", "Only test this for now"

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )

        response = model.generate_content(prompt)

        return response.text

    elif server_type == "deepseek":
        
        if model in ["deepseek-chat", "deepseek-coder"]:
            # regular deepseek model 
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
                top_p=top_p,
            )

        else: # deepseek reasoner
            assert is_reasoning_model, "Only support deepseek-reasoner for now"
            assert model == "deepseek-reasoner", "Only support deepseek-reasoner for now"
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                n=num_completions,
                max_tokens=max_tokens,
                # do not use temperature or top_p
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "openai":
        if is_reasoning_model:
            assert "o1" in model or "o3" in model, "Only support o1 and o3 for now"
            print(f"Using OpenAI reasoning model: {model} with reasoning effort {reasoning_effort}")
            print(f"Using OpenAI reasoning model: {model} with reasoning effort {reasoning_effort}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort=reasoning_effort,
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
                top_p=top_p,
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "together":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
            top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            # truncate=32256,
            stream=False,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "fireworks":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            # top_p=top_p,
            # top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            # truncate=32256,
            stream=False,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "sambanova":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
    # for all other kinds of servers, use standard API
    else:
        if isinstance(prompt, str):
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.text for choice in response.choices]
        else:
            response = client.chat.completions.create(
                model=model,
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
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": 10210,
        "server_address": "matx2.stanford.edu",
        "max_tokens": 8192,
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
    "sambanova": {
        "model_name": "Meta-Llama-3.1-405B-Instruct",
        "temperature": 0.1,
        "max_tokens": 8192,
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
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Querying server {server_type} with args: {server_args}")
        
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