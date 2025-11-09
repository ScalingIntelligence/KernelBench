# src/prompt_constructor_multilang.py  (new option-based prompt constructor)
import os
from .loader import render_prompt_by_option, _abs_path

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPTS_TOML = _abs_path("src/prompts/prompts.toml")
GPU_SPECS_PY = "src/prompts/hardware/gpu_specs.py"  # still a Python file

def get_prompt_for_language(ref_arch_src: str, language: str = "triton", option: str = "few_shot") -> str:
    """
    Generate a prompt for a specific language and option.
    
    Args:
        ref_arch_src: The reference architecture source code
        language: The kernel language (triton, cuda, cute)
        option: The prompt option (basic, few_shot, hardware_info)
    """
    return render_prompt_by_option(
        prompts_toml=PROMPTS_TOML,
        language=language.lower(),
        option=option,
        context={"ref_arch_src": ref_arch_src},
    )

def get_prompt_with_hardware(ref_arch_src: str, language: str, gpu_name: str) -> str:
    """
    Generate a hardware-aware prompt for a specific language.
    
    Args:
        ref_arch_src: The reference architecture source code
        language: The kernel language (triton, cuda, cute)
        gpu_name: The name of the GPU (e.g., "A100", "H100")
    """
    return render_prompt_by_option(
        prompts_toml=PROMPTS_TOML,
        language=language.lower(),
        option="hardware_info",
        context={"ref_arch_src": ref_arch_src},
        gpu_specs_py=GPU_SPECS_PY,
        gpu_name=gpu_name,
    )

def prompt_fix_compile(language: str, ref_arch_src: str, custom_kernel: str, metadata: str) -> str:
    """
    Generate a prompt to fix compilation errors.
    
    Args:
        language: The kernel language (triton, cuda, cute)
        ref_arch_src: The reference architecture source code
        custom_kernel: The custom kernel code that failed
        metadata: Compilation error metadata
    """
    return render_prompt_by_option(
        prompts_toml=PROMPTS_TOML,
        language=language.lower(),
        option="fix_compile",
        context={
            "ref_arch_src": ref_arch_src,
            "custom_kernel": custom_kernel,
            "metadata": metadata,
            "failure_type": "to compile",
        },
    )

def prompt_fix_correctness(language: str, ref_arch_src: str, custom_kernel: str, metadata: str) -> str:
    """
    Generate a prompt to fix correctness errors.
    
    Args:
        language: The kernel language (triton, cuda, cute)
        ref_arch_src: The reference architecture source code
        custom_kernel: The custom kernel code that failed
        metadata: Correctness error metadata
    """
    return render_prompt_by_option(
        prompts_toml=PROMPTS_TOML,
        language=language.lower(),
        option="fix_correctness",
        context={
            "ref_arch_src": ref_arch_src,
            "custom_kernel": custom_kernel,
            "metadata": metadata,
            "failure_type": "correctness",
        },
    )

__all__ = [
    "get_prompt_for_language",
    "get_prompt_with_hardware",
    "prompt_fix_compile",
    "prompt_fix_correctness",
]
