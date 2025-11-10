"""
Backend module for AIDE integration
Provides LLM query interface compatible with AIDE's agent
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from litellm import completion
from src.utils import query_server

logger = logging.getLogger("aide")


@dataclass
class FunctionSpec:
    """Function specification for structured outputs (not used in kernel search)."""
    name: str
    json_schema: dict
    description: str


def compile_prompt_to_md(prompt: dict | str) -> str:
    """
    Convert a prompt dictionary to markdown string.
    
    Args:
        prompt: Either a string or a dict with sections
        
    Returns:
        Markdown formatted string
    """
    if isinstance(prompt, str):
        return prompt
    
    lines = []
    for key, value in prompt.items():
        lines.append(f"## {key}\n")
        
        if isinstance(value, str):
            lines.append(value)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                lines.append(f"### {subkey}\n")
                if isinstance(subvalue, str):
                    lines.append(subvalue)
                elif isinstance(subvalue, list):
                    for item in subvalue:
                        lines.append(f"- {item}")
                lines.append("")
        elif isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
        
        lines.append("")
    
    return "\n".join(lines)


def query(
    system_message: dict[str, Any] | str,
    user_message: str | None = None,
    func_spec: dict[str, Any] | None = None,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    debug: bool = False,
) -> str | dict[str, Any]:
    """
    Query an LLM via LiteLLM.
    
    Args:
        system_message: System prompt (dict or string)
        user_message: User prompt (optional)
        func_spec: Function specification for structured output (optional)
        model: Model name (with provider prefix, e.g., "openai/gpt-4")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        String response or dict if func_spec provided
    """
    if debug:
        print(f"\n[DEBUG backend.query] Called with:")
        print(f"  - model: {model}")
        print(f"  - temperature: {temperature}")
        print(f"  - max_tokens: {max_tokens}")
        print(f"  - func_spec: {'Yes' if func_spec else 'No'}")
        print(f"  - system_message type: {type(system_message)}")
    
    # Format messages
    messages = []
    
    if isinstance(system_message, dict):
        formatted = compile_prompt_to_md(system_message)
        messages.append({"role": "system", "content": formatted})
    else:
        messages.append({"role": "system", "content": system_message})
    
    if user_message:
        messages.append({"role": "user", "content": user_message})
    
    if debug:
        print(f"[DEBUG backend.query] Formatted {len(messages)} messages")
    
    # Call LiteLLM
    try:
        if debug:
            print(f"[DEBUG backend.query] Calling litellm.completion()...")
        if func_spec:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            if debug:
                print(f"[DEBUG backend.query] Got response (structured)")
            content = response.choices[0].message.content
            if content is None:
                if debug:
                    print(f"[DEBUG backend.query] Warning: Response content is None")
                raise ValueError("LLM returned None response content")
            return json.loads(content)
        else:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                if debug:
                    print(f"[DEBUG backend.query] Warning: Response content is None")
                    print(f"[DEBUG backend.query] Full response: {response}")
                raise ValueError("LLM returned None response content")
            if debug:
                print(f"[DEBUG backend.query] Got response (text), length: {len(content)}")
            return content
    except Exception as e:
        if debug:
            print(f"[DEBUG backend.query] Exception: {type(e).__name__}: {e}")
        logger.error(f"Error in query_server for model {model}: {e}")
        raise
