"""
Kernel Static Checker - Pattern-based validation for GPU kernel code.

Detects reward hacking patterns in generated kernel code:
- Bypass hacks (PyTorch wrapping, try-except fallback, inheritance bypass)
- Forbidden high-level torch operations
- Backend implementation requirements (CUDA, Triton, ThunderKittens)

Usage:
    result = validate_kernel(code, backend="cuda", forbidden=["pytorch_wrap", "forbidden_ops"])
    if not result["valid"]:
        for error in result["errors"]:
            print(f"ERROR: {error}")
"""

import re
from typing import List, Tuple, Dict, Any, Optional

# =============================================================================
# GENERIC PATTERNS (apply to all backends)
# =============================================================================

# PyTorch wrapping patterns (Kevin paper: zero reward for torch.nn/torch.nn.functional)
PYTORCH_WRAP_PATTERNS = [
    r"torch\.nn\.functional",
    r"\bnn\.functional\.",
    r"\bF\.(conv|linear|relu|gelu|softmax|batch_norm|layer_norm|dropout)",
]

# Disallowed nn patterns (allow only containers, Parameter, init)
DISALLOWED_NN_PATTERN = r'nn\.(?!(Module|parameter|Parameter|ParameterList|ParameterDict|ModuleList|ModuleDict|init)\b)'

# Try-except patterns (Kevin paper: zero reward for try/except fallback)
TRY_EXCEPT_PATTERNS = [r"\btry\s*:", r"\bexcept\s*:", r"\bexcept\s+\w+"]

# Pass statement (Kevin paper: zero reward for inheritance bypass)
PASS_PATTERN = r"\bpass\b"

# Forbidden high-level torch operations
FORBIDDEN_TORCH_OPS = [
    # Convolutions
    "torch.conv1d", "torch.conv2d", "torch.conv3d",
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
]

# =============================================================================
# BACKEND-SPECIFIC PATTERNS
# =============================================================================

# CUDA patterns
CUDA_KERNEL_REQUIRED = ["__global__", "load_inline"]

# Triton patterns
TRITON_JIT_PATTERN = r"@triton\.(jit|autotune)"
TRITON_OPS_PATTERN = r"\btl\.\w+"

# ThunderKittens patterns
TK_WARP_PATTERNS = [
    r"kittens::warp\b", r"kittens::warpgroup\b",
    r"::warpgroup::", r"::warp::", r"warpgroup::", r"warp::"
]
TK_TILE_PATTERN = r"(?:kittens::)?(?:st|rt)_\w+\s*<[^>]+>"

# =============================================================================
# TODO: Future checks from DeepReinforce blog
# =============================================================================
# STREAM_PATTERNS = [r"torch\.cuda\.Stream\s*\(", r"with\s+torch\.cuda\.stream\s*\("]
# THREAD_PATTERNS = [r"threading\.Thread\s*\(", r"multiprocessing\.Process\s*\("]
# LAZY_EVAL_PATTERNS = [r"torch\.Tensor\._make_subclass", r"def\s+__new__\s*\("]
# MONKEY_PATCH_PATTERNS = [r"torch\.cuda\.Event\.elapsed_time\s*="]
# DTYPE_CONVERSION_PATTERNS = [r"\.half\s*\(\)", r"\.bfloat16\s*\(\)", r"\.float16\s*\(\)"]
# AUTOCAST_PATTERNS = [r"torch\.autocast", r"torch\.cuda\.amp\.autocast"]

# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

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


def check_pytorch_wrap(code: str) -> Tuple[bool, str]:
    """Check for PyTorch API wrapping (torch.nn.functional, F.*, etc.)."""
    code = _strip_comments(code)
    for pattern in PYTORCH_WRAP_PATTERNS:
        if re.search(pattern, code):
            return (True, "Uses torch.nn.functional or F.* - not a custom kernel")
    if re.search(DISALLOWED_NN_PATTERN, code):
        return (True, "Uses disallowed torch.nn (only containers, Parameter, init allowed)")
    return (False, "")


def check_try_except(code: str) -> Tuple[bool, str]:
    """Check for try-except blocks (potential PyTorch fallback)."""
    code = _strip_comments(code)
    for pattern in TRY_EXCEPT_PATTERNS:
        if re.search(pattern, code):
            return (True, "Contains try-except block (potential fallback)")
    return (False, "")


def check_pass_stmt(code: str) -> Tuple[bool, str]:
    """Check for pass statements (inheritance bypass)."""
    code = _strip_comments(code)
    if re.search(PASS_PATTERN, code):
        return (True, "Contains 'pass' statement (inheritance bypass)")
    return (False, "")


def check_forbidden_ops(code: str) -> Tuple[bool, str]:
    """Check for forbidden high-level torch operations."""
    code = _strip_comments(code)
    pattern = r'\b(' + '|'.join(re.escape(f) for f in FORBIDDEN_TORCH_OPS) + r')(?=\s*\(|\s|$)'
    match = re.search(pattern, code)
    if match:
        return (True, f"Uses forbidden operation: {match.group(0)}")
    return (False, "")


def check_cuda_impl(code: str) -> Tuple[bool, str]:
    """Check for valid CUDA kernel implementation."""
    code = _strip_comments(code)
    missing = [p for p in CUDA_KERNEL_REQUIRED if p not in code]
    if missing:
        return (True, f"Missing CUDA requirements: {', '.join(missing)}")
    return (False, "")


def check_triton_impl(code: str) -> Tuple[bool, str]:
    """Check for valid Triton kernel implementation."""
    code = _strip_comments(code)
    if not re.search(TRITON_JIT_PATTERN, code):
        return (True, "Missing @triton.jit or @triton.autotune")
    if not re.search(TRITON_OPS_PATTERN, code):
        return (True, "No tl.* operations found in Triton kernel")
    return (False, "")


def check_tk_impl(code: str) -> Tuple[bool, str]:
    """Check for valid ThunderKittens kernel implementation."""
    code = _strip_comments(code)
    if not any(re.search(p, code) for p in TK_WARP_PATTERNS):
        return (True, "Missing ThunderKittens warp/warpgroup patterns")
    if not re.search(TK_TILE_PATTERN, code):
        return (True, "Missing ThunderKittens tile declarations (st_*/rt_*)")
    return (False, "")


# =============================================================================
# CHECK REGISTRY
# =============================================================================

# All available checks
CHECK_FUNCTIONS: Dict[str, callable] = {
    # Bypass hacks
    "pytorch_wrap": check_pytorch_wrap,
    "try_except": check_try_except,
    "pass_stmt": check_pass_stmt,
    # Forbidden operations
    "forbidden_ops": check_forbidden_ops,
    # Implementation checks (backend-specific)
    "cuda_impl": check_cuda_impl,
    "triton_impl": check_triton_impl,
    "tk_impl": check_tk_impl,
}

# Suggested presets
STRICT_CHECKS = ["pytorch_wrap", "try_except", "pass_stmt", "forbidden_ops"]
WARNING_CHECKS = []  # Add checks here that should warn but not fail

# Backend implementation check mapping
BACKEND_IMPL_CHECK = {
    "cuda": "cuda_impl",
    "triton": "triton_impl",
    "thunderkittens": "tk_impl",
}

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def validate_kernel_static(
    code: str,
    backend: Optional[str] = None,
    forbidden: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate kernel code against configurable check groups.
    
    Args:
        code: Kernel source code
        backend: "cuda", "triton", or "thunderkittens" (adds impl check)
        forbidden: List of check names that cause errors (default: STRICT_CHECKS)
        warnings: List of check names that cause warnings (default: WARNING_CHECKS)
    
    Returns:
        {
            "valid": bool,           # True if no forbidden checks failed
            "errors": List[str],     # Error messages
            "warnings": List[str],   # Warning messages
            "details": Dict[str, bool]  # Per-check results
        }
    """
    if forbidden is None:
        forbidden = STRICT_CHECKS.copy()
    if warnings is None:
        warnings = WARNING_CHECKS.copy()
    
    # Add backend implementation check if specified
    if backend and backend in BACKEND_IMPL_CHECK:
        impl_check = BACKEND_IMPL_CHECK[backend]
        if impl_check not in forbidden:
            forbidden.append(impl_check)
    
    errors = []
    warnings_list = []
    details = {}
    
    # Run all requested checks
    all_checks = set(forbidden + warnings)
    for check_name in all_checks:
        if check_name not in CHECK_FUNCTIONS:
            continue
        has_issue, msg = CHECK_FUNCTIONS[check_name](code)
        details[check_name] = has_issue
        if has_issue:
            if check_name in forbidden:
                errors.append(msg)
            elif check_name in warnings:
                warnings_list.append(msg)
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "details": details,
    }
