# src/prompt_constructor_multilang.py  (new option-based prompt constructor)
import os
from .loader import render_prompt_by_option

def get_prompt_for_language(ref_arch_src: str,
                            language: str = "triton",
                            option: str = "few_shot",
                            hardware_name: str = None,
                            hardware_type: str = None) -> str:
    """
    Generate a prompt for a specific language and option.
    
    Args:
        ref_arch_src: The reference architecture source code
        language: The kernel language (triton, cuda, cute)
        option: The prompt option (basic, few_shot, hardware_info)
    """
    return render_prompt_by_option(
        language=language.lower(),
        option=option,
        context={"ref_arch_src": ref_arch_src},
        hardware_type=hardware_type,
        hardware_name=hardware_name,
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
