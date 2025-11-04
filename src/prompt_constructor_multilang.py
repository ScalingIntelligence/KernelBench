# src/prompts/prompt_constructor.py  (public facade; keep old imports working)
import os
from .loader import render_prompt, _abs_path

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPTS_TOML = _abs_path("src/prompts/prompts.toml")
GPU_SPECS_PY = "src/prompts/hardware/gpu_specs.py"  # still a Python file

def get_prompt_for_backend(ref_arch_src: str, backend: str = "triton") -> str:
    return render_prompt(
        prompts_toml=PROMPTS_TOML,
        backend=backend.lower(),
        template="default",
        context={"ref_arch_src": ref_arch_src},
    )

def get_prompt_with_hardware(ref_arch_src: str, backend: str, gpu_name: str) -> str:
    return render_prompt(
        prompts_toml=PROMPTS_TOML,
        backend=backend.lower(),
        template="with_hardware",
        context={"ref_arch_src": ref_arch_src},
        gpu_specs_py=GPU_SPECS_PY,  # <-- python file, not TOML
        gpu_name=gpu_name,
    )

def prompt_fix_compile(backend: str, ref_arch_src: str, custom_kernel: str, metadata: str) -> str:
    return render_prompt(
        prompts_toml=PROMPTS_TOML,
        backend=backend.lower(),
        template="fix_compile",
        context={
            "ref_arch_src": ref_arch_src,
            "custom_kernel": custom_kernel,
            "metadata": metadata,
            "failure_type": "to compile",
        },
    )

def prompt_fix_correctness(backend: str, ref_arch_src: str, custom_kernel: str, metadata: str) -> str:
    return render_prompt(
        prompts_toml=PROMPTS_TOML,
        backend=backend.lower(),
        template="fix_correctness",
        context={
            "ref_arch_src": ref_arch_src,
            "custom_kernel": custom_kernel,
            "metadata": metadata,
            "failure_type": "correctness",
        },
    )

# Optional legacy convenience wrappers (if callers use backend-specific names)
def prompt_fix_compile_triton(ref_arch_src, custom_kernel, metadata):
    return prompt_fix_compile("triton", ref_arch_src, custom_kernel, metadata)

def prompt_fix_correctness_triton(ref_arch_src, custom_kernel, metadata):
    return prompt_fix_correctness("triton", ref_arch_src, custom_kernel, metadata)

def prompt_fix_compile_cute(ref_arch_src, custom_kernel, metadata):
    return prompt_fix_compile("cute", ref_arch_src, custom_kernel, metadata)

def prompt_fix_correctness_cute(ref_arch_src, custom_kernel, metadata):
    return prompt_fix_correctness("cute", ref_arch_src, custom_kernel, metadata)

def prompt_fix_compile_cuda(ref_arch_src, custom_kernel, metadata):
    return prompt_fix_compile("cuda", ref_arch_src, custom_kernel, metadata)

def prompt_fix_correctness_cuda(ref_arch_src, custom_kernel, metadata):
    return prompt_fix_correctness("cuda", ref_arch_src, custom_kernel, metadata)

__all__ = [
    "get_prompt_for_backend",
    "get_prompt_with_hardware",
    "prompt_fix_compile",
    "prompt_fix_correctness",
    "prompt_fix_compile_triton",
    "prompt_fix_correctness_triton",
    "prompt_fix_compile_cute",
    "prompt_fix_correctness_cute",
    "prompt_fix_compile_cuda",
    "prompt_fix_correctness_cuda",
]
