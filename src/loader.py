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
class TomlConfig:
    data: Dict[str, Any]

    @classmethod
    def from_toml(cls, path: str) -> "TomlConfig":
        """Load a TOML file and return a TomlConfig instance."""
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

def _hardware_context_from_path(specs_path: str, hardware_type: str, hardware_name: str) -> Dict[str, str]:
    """
    Load hardware spec dicts from a TOML file (.toml).
    - HARDWARE_SPEC_INFO = { type: { name: { ... } } }

    For TOML files we expect structure like [hardware_type.hardware_name], plus optional
    [hardware_type.definitions], [hardware_type.best_practices], etc.
    """
    cfg = TomlConfig.from_toml(specs_path)
    data = cfg.data

    # Resolve hardware_type default from meta if not provided
    if not hardware_type:
        hardware_type = data.get("meta", {}).get("default_hardware_type")

    hw_section = data.get(hardware_type)
    if not isinstance(hw_section, dict):
        raise KeyError(f"Hardware type '{hardware_type}' not found in specs TOML")

    curr = None
    if isinstance(hw_section.get(hardware_name), dict):
        curr = hw_section[hardware_name]
    if curr is None:
        raise KeyError(f"Hardware '{hardware_name}' not found under type '{hardware_type}' in {specs_path}")

    # definitions
    definitions = {}
    if isinstance(hw_section.get("definitions"), dict):
        definitions.update(hw_section.get("definitions"))

    # best practices
    best_list: List[str] = []
    if isinstance(hw_section.get("best_practices"), dict):
        best_list.extend(hw_section.get("best_practices", {}).get("items", []))

    # Derive architecture name from common keys
    hardware_architecture = curr.get("architecture") or "Unknown"

    # Build human-readable bullets for specs/definitions/best practices
    specs_bullets = "\n".join([f"- {k}: {v}" for k, v in curr.items() if k != "architecture"])
    defs_bullets = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
    best_bullets = "\n".join([f"- {x}" for x in best_list])

    return {
        "hardware_type": hardware_type,
        "hardware_name": hardware_name,
        "hardware_architecture": hardware_architecture,
        "hardware_specs_bullets": specs_bullets,
        "hardware_definitions_bullets": defs_bullets,
        "hardware_best_practices_bullets": best_bullets,
    }

def render_prompt_by_option(
    *,
    prompts_toml: str,
    language: str,
    option: str,
    context: Dict[str, str],
    hardware_specs_toml: Optional[str] = None,
    hardware_type: Optional[str] = None,
    hardware_name: Optional[str] = None,
) -> str:
    """
    New function that uses languages.X and options.Y structure from prompts.toml

    Args:
        prompts_toml: Path to the prompts.toml file
        language: The kernel language (triton, cuda, cute)
        option: The prompt option (basic, few_shot, hardware_info, fix_compile, fix_correctness)
        context: Variables to fill in the prompt template
        hardware_specs_py: Optional path to hardware specs file (.py or .toml)
        hardware_type: Hardware type (e.g., "GPU", "Tenstorrent")
        hardware_name: Hardware name (e.g., "A100", "H100")
    """
    cfg = TomlConfig.from_toml(prompts_toml)
    
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
        context.update(
            {
                "example_arch_src": read_file(ex_arch_path),
                "example_new_arch_src": read_file(ex_new_path),
            }
        )

    # Load hardware details if requested
    if option_data.get("requires_hardware"):
        if not (hardware_specs_toml and hardware_type and hardware_name):
            raise ValueError(f"Option '{option}' requires hardware info; provide hardware_specs_toml, hardware_type, and hardware_name")
        context.update(**_hardware_context_from_path(_abs_path(hardware_specs_toml), hardware_type, hardware_name),)

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
