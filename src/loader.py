# src/loader.py
import os
import runpy
import tomli  # pip install tomli
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import read_file  # your existing util

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _abs_path(rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(REPO_TOP_PATH, rel)

@dataclass
class PromptConfig:
    data: Dict[str, Any]

    @classmethod
    def from_toml(cls, path: str) -> "PromptConfig":
        with open(path, "rb") as f:
            data = tomli.load(f)
        return cls(data)

    def compose_blocks(self, keys: List[str]) -> str:
        text_parts = []
        for key in keys:
            node: Any = self.data
            for part in key.split("."):
                if part not in node:
                    raise KeyError(f"compose key not found: {key}")
                node = node[part]
            if not isinstance(node, str):
                raise TypeError(f"compose key must resolve to string: {key}")
            text_parts.append(node.strip() + "\n")
        return "\n".join(text_parts).strip() + "\n"

def _gpu_context_from_py(py_path: str, gpu_name: str) -> Dict[str, str]:
    """
    Load GPU_* dicts from a Python file (no exec of raw strings; use runpy).
    Expected globals:
      - GPU_SPEC_INFO: dict[str, dict]
      - GPU_DEFINITIONS: dict[str, str]
      - GPU_BEST_PRACTICES: list[str]  OR {"list": [...]} for compatibility
    """
    mod = runpy.run_path(py_path)
    spec_info = mod.get("GPU_SPEC_INFO", {})
    definitions = mod.get("GPU_DEFINITIONS", {})
    best = mod.get("GPU_BEST_PRACTICES", [])

    if not spec_info or not definitions or best is None:
        raise ValueError("GPU_SPEC_INFO / GPU_DEFINITIONS / GPU_BEST_PRACTICES missing in gpu specs .py")

    if isinstance(best, dict) and "list" in best:
        best = best["list"]

    if gpu_name not in spec_info:
        raise KeyError(f"GPU name {gpu_name} not found in GPU_SPEC_INFO")

    curr = spec_info[gpu_name]
    gpu_architecture = curr.get("GPU Architecture", "Unknown")
    specs_bullets = "\n".join([f"- We have {v} of {k}." for k, v in curr.items() if k != "GPU Architecture"])
    defs_bullets = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
    best_bullets = "\n".join([f"- {x}" for x in (best or [])])

    return {
        "gpu_name": gpu_name,
        "gpu_architecture": gpu_architecture,
        "gpu_specs_bullets": specs_bullets,
        "gpu_definitions_bullets": defs_bullets,
        "gpu_best_practices_bullets": best_bullets,
    }

def render_prompt_by_option(
    *,
    prompts_toml: str,
    language: str,
    option: str,
    context: Dict[str, str],
    gpu_specs_py: Optional[str] = None,
    gpu_name: Optional[str] = None,
) -> str:
    """
    New function that uses languages.X and options.Y structure
    
    Args:
        prompts_toml: Path to the prompts.toml file
        language: The kernel language (triton, cuda, cute)
        option: The prompt option (basic, few_shot, hardware_info, fix_compile, fix_correctness)
        context: Variables to fill in the prompt template
        gpu_specs_py: Optional path to GPU specs Python file
        gpu_name: Optional GPU name (required if option requires_gpu)
    """
    cfg = PromptConfig.from_toml(prompts_toml)
    
    # Get language-specific content
    try:
        lang_data = cfg.data["languages"][language]
    except KeyError:
        raise KeyError(f"Unknown language: {language}")
    
    # Get option configuration
    try:
        option_data = cfg.data["options"][option]
    except KeyError:
        raise KeyError(f"Unknown option: {option}")
    
    # Get shared templates
    shared = cfg.data.get("shared", {})
    language_display = lang_data.get("language_display", language.upper())
    
    # Fill in shared templates with language-specific terms
    problem_statement = shared.get("problem_statement", "").format(language_display=language_display)
    instruction = shared.get("instruction", "").format(language_display=language_display)
    
    # Add language-specific content to context
    context = {
        **context,
        "language": language.upper() if language in ["cuda", "cute"] else language.capitalize(),
        "language_display": language_display,
        "problem_statement": problem_statement,
        "instruction": instruction,
    }
    
    # Load example files if requested
    if option_data.get("requires_example"):
        # Use language-specific example arch, or fall back to shared one
        ex_arch_path = _abs_path(
            lang_data.get("few_shot_example_arch") or shared.get("few_shot_example_arch")
        )
        ex_new_path = _abs_path(lang_data["few_shot_new_arch"])
        context = {
            **context,
            "example_arch_src": read_file(ex_arch_path),
            "example_new_arch_src": read_file(ex_new_path),
        }
    
    # Load GPU details if requested
    if option_data.get("requires_gpu"):
        if not (gpu_specs_py and gpu_name):
            raise ValueError(f"Option '{option}' requires GPU info; provide gpu_specs_py and gpu_name")
        context = {**context, **_gpu_context_from_py(_abs_path(gpu_specs_py), gpu_name)}
    
    # Build the prompt from components
    prompt_parts = []
    for component in option_data["components"]:
        if component == "problem_statement":
            # Use the already-formatted problem_statement from context
            prompt_parts.append(context["problem_statement"])
        elif component == "instruction":
            # Use the already-formatted instruction from context
            prompt_parts.append(context["instruction"])
        elif component.startswith("hardware_"):
            # Hardware components from templates.hardware
            template_key = f"templates.hardware.{component}"
            prompt_parts.append(cfg.compose_blocks([template_key]))
        else:
            # Other components from templates.common
            template_key = f"templates.common.{component}"
            prompt_parts.append(cfg.compose_blocks([template_key]))
    
    prompt_text = "\n".join(prompt_parts).strip() + "\n"
    
    try:
        return prompt_text.format(**context).strip() + "\n"
    except KeyError as e:
        raise KeyError(f"Missing placeholder in context: {e.args[0]}. Available: {list(context.keys())}") from e
