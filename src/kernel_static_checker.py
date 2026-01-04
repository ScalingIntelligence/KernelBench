"""
Kernel Static Checker - Pattern-based validation for GPU kernel code.

The goal is flag reward hacking patterns (both strictly prohibited and possible ones).
through statically examining the code.

Warning: This list is by no means complete and nor this is not a replacement for runtime checks.
We welcome feedback and contributions as community find new ways of hacks.

- Bypass hacks (PyTorch wrapping, try-except fallback, inheritance bypass)
- Disallow some high-level torch operations (depends on the settings)
- Backend implementation requirements, that CUDA or DSL features must be used

Usage:
    result = validate_kernel_static(code, backend="cuda")
    will return a tuple (valid, errors, warnings) 
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Callable

def _strip_comments(code: str) -> str:
    """Remove # and // comments from code."""
    lines = []
    for line in code.split('\n'):
        if '#' in line:
            line = line[:line.index('#')]
        if '//' in line:
            line = line[:line.index('//')]
        lines.append(line)
    return '\n'.join(lines)


# =============================================================================
# BYPASS CHECKS - Strictly Prohibited 
# some of this is from Kevin RL Paper (arxiv:2507.11948)
# =============================================================================

# --- Try-Except Fallback ---
# Rationale: Models wrap incomplete CUDA in exception handlers that fall back to PyTorch.
# This allows them to pass tests without actually implementing the kernel.
TRY_EXCEPT_PATTERNS = [r"\btry\s*:", r"\bexcept\s*:", r"\bexcept\s+\w+"]

# --- Pass Statement / Inheritance Bypass ---
# Rationale: Model inherits from reference class and uses 'pass' to do nothing,
# effectively just calling the parent implementation.
PASS_PATTERN = r"\bpass\b"

def check_code_bypass(code: str) -> Tuple[bool, str]:
    """
    Check for code bypass patterns (strictly prohibited).
    
    Detects two Kevin paper bypass techniques:
    1. Try-Except Fallback: Models wrap incomplete CUDA in exception handlers
       that fall back to PyTorch when custom code fails.
    2. Pass Statement: Models inherit from reference and use 'pass' to do nothing,
       effectively calling parent implementation.
    
    Both receive zero reward per Kevin paper rules.
    Uses word boundary for 'pass' to avoid matching 'passed', 'bypass', etc.
    """
    code = _strip_comments(code)
    
    # Check for try-except fallback
    for pattern in TRY_EXCEPT_PATTERNS:
        if re.search(pattern, code):
            return (True, "Contains try-except block (potential fallback bypass)")
    
    # Check for pass statement
    if re.search(PASS_PATTERN, code):
        return (True, "Contains 'pass' statement (inheritance bypass)")
    
    return (False, "")

# Since KernelBench problems uses PyTorch as a reference
# sometimes model chose to wrap PyTorch high-level APIs instead of implementing a custom kernel.
# Depends on the setting that is used, (configure this in the STRICT and WARNING checks)
# some left model replaces some ops with custom kernels in required backend,
# others require the model to write the whole program in custom kernels.

# --- PyTorch Wrapping  ---
# Rationale: Using torch.nn.functional or F.* means the model is just wrapping
# PyTorch high-level APIs instead of implementing a custom kernel.
# This defeats the purpose of kernel generation.
PYTORCH_WRAP_PATTERNS = [
    r"torch\.nn\.functional",
    r"\bnn\.functional\.",
    r"\bF\.(conv|linear|relu|gelu|softmax|batch_norm|layer_norm|dropout)",
]
# Allow only containers (Module, ModuleList, etc.), Parameter, and init
PYTORCH_DISALLOWED_NN_PATTERN = r'torch\.nn\.(?!(Module|parameter|Parameter|ParameterList|ParameterDict|ModuleList|ModuleDict|init)\b)'

def check_pytorch_wrap(code: str) -> Tuple[bool, str]:
    """
    Check for PyTorch API wrapping (torch.nn.functional, F.*, etc.).
    
    Kevin paper rule: Zero reward for kernels containing torch.nn or torch.nn.functional.
    These indicate the model is wrapping PyTorch instead of writing custom kernels.
    """
    code = _strip_comments(code)
    for pattern in PYTORCH_WRAP_PATTERNS:
        if re.search(pattern, code):
            return (True, "Uses torch.nn.functional or F.* - not a custom kernel")
    # NOTE: should we make this strict or optional?
    if re.search(PYTORCH_DISALLOWED_NN_PATTERN, code):
        return (True, "Uses disallowed torch.nn (only containers, Parameter, init allowed)")
    return (False, "")


# --- Forbidden Torch Computational Operations ---
# Rationale: These are high-level PyTorch ops that conduct computation
# Using them directly defeat the purpose writing custom CUDA kernels.
TORCH_COMPUTATION_OPS = [
    # Matrix operations
    "torch.mm", "torch.bmm", "torch.matmul", "torch.einsum",
    # Convolutions
    "torch.conv1d", "torch.conv2d", "torch.conv3d", "torch.conv",
    "torch.conv_transpose1d", "torch.conv_transpose2d", "torch.conv_transpose3d",
    # Pooling
    "torch.avg_pool1d", "torch.avg_pool2d", "torch.avg_pool3d",
    "torch.max_pool1d", "torch.max_pool2d", "torch.max_pool3d",
    "torch.adaptive_avg_pool1d", "torch.adaptive_avg_pool2d", "torch.adaptive_avg_pool3d",
    "torch.adaptive_max_pool1d", "torch.adaptive_max_pool2d", "torch.adaptive_max_pool3d",
    # Activations
    "torch.relu", "torch.hardtanh", "torch.elu", "torch.selu",
    "torch.leaky_relu", "torch.gelu", "torch.softsign", "torch.softplus",
    "torch.softmax", "torch.log_softmax", "torch.tanh", "torch.sigmoid",
    "torch.hardsigmoid", "torch.silu", "torch.mish",
    # Normalization
    "torch.batch_norm", "torch.group_norm", "torch.layer_norm",
    "torch.instance_norm", "torch.rms_norm", "torch.normalize",
    # Linear & Loss
    "torch.linear", "torch.cross_entropy", "torch.kl_div", "torch.mse_loss",
    "torch.huber_loss", "torch.triplet_margin_loss", "torch.cosine_similarity",
    # Others
    "torch.logsumexp", "torch.clamp", "torch.dropout",
] # hopefully didn't miss any other ones

def check_torch_computation_ops(code: str) -> Tuple[bool, str]:
    """
    Check for high-level torch computation operations.
    
    Note: This check is optional/taste-based. Some projects allow partial
    torch usage, others require pure custom kernels. Configure as needed.
    """
    code = _strip_comments(code)
    pattern = r'\b(' + '|'.join(re.escape(f) for f in TORCH_COMPUTATION_OPS) + r')(?=\s*\(|\s|$)'
    match = re.search(pattern, code)
    if match:
        return (True, f"Uses torch computation op: {match.group(0)}")
    return (False, "")

# =============================================================================
# Backend Specific Checks
# =============================================================================

# <========= CUDA CHECKS =========>
# Rationale: Valid CUDA kernels must have __global__ (kernel definition) and
# use load_inline or cpp_extension (PyTorch's inline compilation).
CUDA_COMPILE_PATTERNS = ["load_inline", "cpp_extension"]

def check_cuda_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid CUDA kernel implementation.
    
    Requirements:
    - Must have __global__ void kernel_name (kernel definition)
    - Must have load_inline or cpp_extension (PyTorch inline compilation)
    """
    code = _strip_comments(code)
    if "__global__" not in code:
        return (True, "Missing __global__ kernel definition")
    if not any(p in code for p in CUDA_COMPILE_PATTERNS):
        return (True, "Missing load_inline or cpp_extension for compilation")
    return (False, "")

# <========= TRITON CHECKS =========>
# Rationale: Triton kernels are compiled from @triton.jit decorated functions.
# They must use tl.* operations (tl.load, tl.store, etc.) for actual kernel work.
TRITON_JIT_PATTERN = r"@triton\.(jit|autotune)"
TRITON_OPS_PATTERN = r"\btl\.\w+"

def check_triton_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid Triton kernel implementation.
    
    Requirements:
    - Must have @triton.jit or @triton.autotune decorator
    - Must have tl.* operations (enforces actual Triton code, not wrapper)
    
    Note: Triton's compiler itself prevents PyTorch ops inside @triton.jit.
    """
    code = _strip_comments(code)
    if not re.search(TRITON_JIT_PATTERN, code):
        return (True, "Missing @triton.jit or @triton.autotune")
    if not re.search(TRITON_OPS_PATTERN, code):
        return (True, "No tl.* operations found in Triton kernel")
    return (False, "")


# <========= THUNDERKITTENS CHECKS =========>
# Rationale: ThunderKittens uses warp/warpgroup primitives and tile abstractions.
# Valid TK code must have namespace patterns and tile declarations.
TK_WARP_PATTERNS = [
    r"kittens::warp\b", r"kittens::warpgroup\b",
    r"::warpgroup::", r"::warp::", r"warpgroup::", r"warp::"
]
TK_TILE_PATTERN = r"(?:kittens::)?(?:st|rt)_\w+\s*<[^>]+>"

def check_tk_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid ThunderKittens kernel implementation.
    
    Requirements:
    - Must have warp/warpgroup namespace patterns (kittens::warp, etc.)
    - Must have tile declarations (st_bf<...>, rt_fl<...>, etc.)
    
    TODO: Add producer-consumer pattern check for complex kernels.
    """
    code = _strip_comments(code)
    if not any(re.search(p, code) for p in TK_WARP_PATTERNS):
        return (True, "Missing ThunderKittens warp/warpgroup patterns")
    if not re.search(TK_TILE_PATTERN, code):
        return (True, "Missing ThunderKittens tile declarations (st_*/rt_*)")
    return (False, "")


# <========= CUTE/CUTLASS CHECKS =========>
# CUTLASS uses cute:: namespace for tensor operations
# Check: https://github.com/NVIDIA/cutlass 
CUTE_PATTERNS = [
    r"cute::",           # cute:: namespace (CuTe library)
    r"cutlass::",        # cutlass:: namespace
    r"from cutlass",     # Python CUTLASS bindings
]

def check_cute_impl(code: str) -> Tuple[bool, str]:
    """Check for valid CUTLASS/CuTe kernel implementation."""
    code = _strip_comments(code)
    if not any(p in code for p in ["cute::", "cutlass::", "from cutlass"]):
        return (True, "Missing cute:: or cutlass:: namespace")
    return (False, "")


# <========= TILELANG CHECKS =========>
# TileLang uses TVM's T.prim_func decorator
# https://github.com/tile-ai/tilelang
TILELANG_PATTERNS = [
    r"@T\.prim_func",    # TVM primitive function decorator
    r"tvm\.build",       # TVM build call
    r"T\.grid",          # TileLang grid
]

def check_tilelang_impl(code: str) -> Tuple[bool, str]:
    """Check for valid TileLang kernel implementation."""
    code = _strip_comments(code)
    if not re.search(r"@T\.prim_func", code):
        return (True, "Missing @T.prim_func decorator")
    return (False, "")


# =============================================================================
# TODO: Future checks from our adversarial hack PR and DeepReinforce blog 
# =============================================================================
# maybe we can group some of that later
# def check_stream_injection(code): ...  # torch.cuda.Stream()
# def check_thread_injection(code): ...  # threading.Thread()
# def check_lazy_eval(code): ...         # torch.Tensor._make_subclass
# def check_monkey_patch(code): ...      # torch.cuda.Event.elapsed_time =
# def check_precision_downgrade(code): ... # .half(), .bfloat16() # <-- this could be a soft check

# =============================================================================
# in the future, we can add a LM as a judge checker
# =============================================================================


# =============================================================================
# REGISTRY & PRESETS
# =============================================================================

CHECK_FUNCTIONS: Dict[str, Callable[[str], Tuple[bool, str]]] = {
    "code_bypass": check_code_bypass,
    "pytorch_wrap": check_pytorch_wrap,
    "torch_computation_ops": check_torch_computation_ops,
    "cuda_impl": check_cuda_impl,

    # backend-specific checks
    "triton_impl": check_triton_impl,
    "tk_impl": check_tk_impl,
    "cute_impl": check_cute_impl,
    "tilelang_impl": check_tilelang_impl,
}

# These checks are NECESSARY for all kernels (strict = error)
STRICT_CHECKS = [
    "code_bypass",
    "pytorch_wrap", 
]

# Backend-specific checks are added later at entry point

# These are optional checks (by taste) - flagged as warnings
# Move to STRICT_CHECKS if you want to enforce them
WARNING_CHECKS: List[str] = [
    "torch_computation_ops",  # up to user to allow torch computation ops
]

BACKEND_IMPL_CHECK = {
    "cuda": "cuda_impl",
    "triton": "triton_impl",
    "thunderkittens": "tk_impl",
    "cute": "cute_impl",
    "cutlass": "cute_impl",  # alias
    "tilelang": "tilelang_impl",
}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def validate_kernel_static(
    code: str,
    backend: str = "cuda",
    precision: str = "fp16",
    forbidden: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate kernel code through statically inspecting the code
    We configure the checks against check groups that we have provided for common hacks.
    Note we do not guarantee that all checks are exhaustive. This is also only on the static level.
    
    Args:
        code: Kernel source code
        backend: "cuda", "triton", or "thunderkittens"
        precision: "fp16", "fp32", or "bf16" (for future precision checks)
        forbidden: Check categories that cause errors (default: STRICT_CHECKS)
        warnings: Check categories that cause warnings (default: WARNING_CHECKS)
        
    Returns:
        (valid, errors, warnings)
        valid: bool
        errors: List[str]
        warnings: List[str]
    """
    # Copy defaults to avoid mutating global lists
    forbidden_checks = list(forbidden) if forbidden is not None else list(STRICT_CHECKS)
    warning_checks = list(warnings) if warnings is not None else list(WARNING_CHECKS)
    
    # Add backend implementation check if specified
    if backend in BACKEND_IMPL_CHECK:
        impl_check = BACKEND_IMPL_CHECK[backend]
        if impl_check not in forbidden_checks:
            forbidden_checks.append(impl_check)
    
    # Aggregate results
    errors: List[str] = []
    warnings_list: List[str] = []
    
    for check_name in set(forbidden_checks + warning_checks):
        if check_name not in CHECK_FUNCTIONS:
            continue
        has_issue, msg = CHECK_FUNCTIONS[check_name](code)
        if has_issue:
            if check_name in forbidden_checks:
                errors.append(msg)
            else:
                warnings_list.append(msg)
    
    valid = len(errors) == 0 # valid if no errors
    return valid, errors, warnings_list
