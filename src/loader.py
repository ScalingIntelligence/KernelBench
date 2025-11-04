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

    def get_template_node(self, backend: str, template: str) -> Dict[str, Any]:
        try:
            return self.data["backends"][backend]["templates"][template]
        except KeyError as e:
            raise KeyError(f"Unknown backend/template: {backend}/{template}") from e

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

def render_prompt(
    *,
    prompts_toml: str,
    backend: str,
    template: str,
    context: Dict[str, str],
    gpu_specs_py: Optional[str] = None,
    gpu_name: Optional[str] = None,
) -> str:
    cfg = PromptConfig.from_toml(prompts_toml)
    node = cfg.get_template_node(backend, template)

    # Load example files if requested
    if node.get("requires_example"):
        ex_arch_path = _abs_path(node["example_arch_path"])
        ex_new_path = _abs_path(node["example_new_arch_path"])
        context = {
            **context,
            "example_arch_src": read_file(ex_arch_path),
            "example_new_arch_src": read_file(ex_new_path),
        }

    # Load GPU details (from .py) if requested
    if node.get("requires_gpu"):
        if not (gpu_specs_py and gpu_name):
            raise ValueError("Template requires GPU info; provide gpu_specs_py and gpu_name")
        context = {**context, **_gpu_context_from_py(_abs_path(gpu_specs_py), gpu_name)}

    # Compose & fill
    compose_keys = node["compose"]
    prompt_text = cfg.compose_blocks(compose_keys)

    try:
        return prompt_text.format(**context).strip() + "\n"
    except KeyError as e:
        raise KeyError(f"Missing placeholder in context: {e.args[0]}") from e
