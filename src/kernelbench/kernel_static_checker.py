"""
Kernel Static Checker - Pattern-based validation for GPU kernel code.

The goal is flag reward hacking patterns (both strictly prohibited and possible ones).
through statically examining the code.

In the future we can add 
- AST-based detections 
- LM as a judge checker

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
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

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
    1. Try-Except Fallback: Models wrap incomplete CUDA in exception handlers
       that fall back to PyTorch when custom code fails.
    2. Pass Statement: Models inherit from reference and use 'pass' to do nothing,
       effectively calling parent implementation.
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

# Since KernelBench problems uses PyTorch as a reference, there could be settigs where
# Model generated code
# 1. Replaces some (not all) ops with custom kernels, others are kept in Torch
# --> More practical from a performance perspective (ie. make better systems) as you want to use whatever makes the best system for your use case. 
# 2. All compuational ops must be replaced with custom kernels
# --> Could be helpful from an eval (model ability on transpile + optimization) / RL training perspective 
# Depends the setting you use, you can move the checks below (pytorch_wrap, torch_computation_ops) 
# from WARNING to STRICT

# --- PyTorch NN Module Wrapping ---
# Allows: nn.Module, nn.Parameter, nn.ParameterList, nn.ParameterDict, 
#         nn.ModuleList, nn.ModuleDict, nn.init (needed for model structure)
# Blocks: nn.Linear, nn.Conv2d, nn.ReLU, etc. (compute layers)
PYTORCH_DISALLOWED_NN_PATTERN = r'torch\.nn\.(?!(Module|parameter|Parameter|ParameterList|ParameterDict|ModuleList|ModuleDict|init)\b)'

def check_pytorch_wrap(code: str) -> Tuple[bool, str]:
    """
    Check for PyTorch nn module usage (nn.Linear, nn.Conv2d, etc.).
    
    Allows containers (nn.Module, nn.Parameter, nn.init) needed for model structure.
    Blocks compute layers (nn.Linear, nn.Conv2d, nn.ReLU, etc.).
    """
    code = _strip_comments(code)
    if re.search(PYTORCH_DISALLOWED_NN_PATTERN, code):
        return (True, "Uses torch.nn compute layer (only containers, Parameter, init allowed)")
    return (False, "")


# --- Torch Computation Operations ---
# Rationale: These are high-level PyTorch ops that conduct computation.
# Using them directly defeats the purpose of writing custom kernels.
# Includes both torch.* and F.* (torch.nn.functional) patterns.
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
]

# F.* patterns (torch.nn.functional equivalents)
TORCH_FUNCTIONAL_PATTERNS = [
    r"torch\.nn\.functional\.\w+",       # torch.nn.functional.*
    r"\bnn\.functional\.\w+",            # nn.functional.*
    r"\bF\.(conv|linear|relu|gelu|softmax|batch_norm|layer_norm|dropout|max_pool|avg_pool)",
]

def check_torch_computation_ops(code: str) -> Tuple[bool, str]:
    """
    Check for high-level torch computation operations.
    
    Matches both torch.* ops (torch.matmul) and F.* ops (F.relu).
    This check is optional/taste-based. Configure as needed.
    """
    code = _strip_comments(code)
    
    # Check torch.* ops
    torch_pattern = r'\b(' + '|'.join(re.escape(f) for f in TORCH_COMPUTATION_OPS) + r')(?=\s*\(|\s|$)'
    match = re.search(torch_pattern, code)
    if match:
        return (True, f"Uses torch computation op: {match.group(0)}")
    
    # Check F.* / nn.functional ops
    for pattern in TORCH_FUNCTIONAL_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return (True, f"Uses torch.nn.functional op: {match.group(0)}")
    
    return (False, "")

# =============================================================================
# Backend Specific Checks
# =============================================================================

# <========= CUDA CHECKS =========>
# Rationale: Valid CUDA kernels must have __global__ (kernel definition) and
# use load_inline or cpp_extension (PyTorch's inline compilation).
CUDA_COMPILE_PATTERNS = ["load_inline", "cpp_extension"]

# Core CUDA patterns that indicate actual kernel implementation
CUDA_THREAD_PATTERNS = [
    r"\bthreadIdx\.",       # threadIdx.x, threadIdx.y, threadIdx.z
    r"\bblockIdx\.",        # blockIdx.x, blockIdx.y, blockIdx.z
    r"\bblockDim\.",        # blockDim.x, blockDim.y, blockDim.z
    r"\bgridDim\.",         # gridDim.x, gridDim.y, gridDim.z
]

# CUDA synchronization and memory patterns
CUDA_KERNEL_PATTERNS = [
    r"__syncthreads\s*\(",      # Thread synchronization
    r"__shared__\s+",           # Shared memory declaration
    r"__device__\s+",           # Device function
    r"atomicAdd\s*\(",          # Atomic operations
    r"atomicMax\s*\(",
    r"atomicMin\s*\(",
]

def check_cuda_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid CUDA kernel implementation.
    
    Requirements:
    - Must have __global__ void kernel_name (kernel definition)
    - Must have load_inline or cpp_extension (PyTorch inline compilation)
    - Must use CUDA thread indexing (threadIdx, blockIdx, blockDim, or gridDim)
      OR CUDA kernel features (__syncthreads, __shared__, __device__, atomics)
    
    Rationale: Ensures code actually implements a CUDA kernel rather than
    just wrapping PyTorch operations.
    """
    code = _strip_comments(code)
    
    # Check for kernel definition
    if "__global__" not in code:
        return (True, "Missing __global__ kernel definition")
    
    # Check for compilation method
    if not any(p in code for p in CUDA_COMPILE_PATTERNS):
        return (True, "Missing load_inline or cpp_extension for compilation")
    
    # Check for actual CUDA kernel features (thread indexing or kernel patterns)
    has_thread_patterns = any(re.search(p, code) for p in CUDA_THREAD_PATTERNS)
    has_kernel_patterns = any(re.search(p, code) for p in CUDA_KERNEL_PATTERNS)
    
    if not (has_thread_patterns or has_kernel_patterns):
        return (True, "Missing CUDA thread indexing or kernel features (threadIdx, blockIdx, __syncthreads, __shared__, etc.)")
    
    return (False, "")

# <========= TRITON CHECKS =========>
# Rationale: Triton kernels are compiled from @triton.jit decorated functions.
# They must use tl.* operations (tl.load, tl.store, etc.) for actual kernel work.
TRITON_JIT_PATTERN = r"@triton\.(jit|autotune)"
TRITON_OPS_PATTERN = r"\btl\.\w+"

# Core Triton memory operations (must-have)
TRITON_MEMORY_OPS = [
    r"tl\.load\s*\(",           # Memory load
    r"tl\.store\s*\(",          # Memory store
]

# Core Triton kernel patterns
TRITON_KERNEL_PATTERNS = [
    r"tl\.program_id\s*\(",     # Program/block ID
    r"tl\.num_programs\s*\(",   # Number of programs
    r"tl\.constexpr",           # Compile-time constants
    r"tl\.arange\s*\(",         # Index generation
    r"tl\.cdiv\s*\(",           # Ceiling division
]

# Triton data types
TRITON_DTYPE_PATTERNS = [
    r"tl\.(float16|float32|float64|int32|int64|bfloat16)",
]

def check_triton_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid Triton kernel implementation.
    
    Requirements:
    - Must have @triton.jit or @triton.autotune decorator
    - Must have tl.* operations (enforces actual Triton code, not wrapper)
    - Must have tl.load or tl.store (core memory operations)
    - Should have tl.program_id or other kernel patterns (for proper indexing)
    
    Note: Triton's compiler itself prevents PyTorch ops inside @triton.jit.
    """
    code = _strip_comments(code)
    
    # Check for decorator
    if not re.search(TRITON_JIT_PATTERN, code):
        return (True, "Missing @triton.jit or @triton.autotune")
    
    # Check for any tl.* operations
    if not re.search(TRITON_OPS_PATTERN, code):
        return (True, "No tl.* operations found in Triton kernel")
    
    # Check for memory operations (load or store)
    has_memory_ops = any(re.search(p, code) for p in TRITON_MEMORY_OPS)
    if not has_memory_ops:
        return (True, "Missing Triton memory operations (tl.load or tl.store)")
    
    # Check for kernel patterns (program_id is essential for indexing)
    has_kernel_patterns = any(re.search(p, code) for p in TRITON_KERNEL_PATTERNS)
    if not has_kernel_patterns:
        return (True, "Missing Triton kernel patterns (tl.program_id, tl.arange, etc.)")
    
    return (False, "")


# <========= THUNDERKITTENS CHECKS =========>
# Rationale: ThunderKittens uses warp/warpgroup primitives and tile abstractions.
# Valid TK code must have namespace patterns and tile declarations.
# Reference: https://github.com/HazyResearch/ThunderKittens/
TK_NAMESPACE_PATTERNS = [
    r"kittens::",               # Core namespace
    r"using namespace kittens", # Using declaration
]

TK_WARP_PATTERNS = [
    r"kittens::warp\b", 
    r"kittens::warpgroup\b",
    r"kittens::group\s*<\s*\d+\s*>",  # kittens::group<4> for warpgroup operations
    r"::warpgroup::", 
    r"::warp::", 
    r"warpgroup::", 
    r"warp::"
]

# ThunderKittens tile types: rt (register tile), st (shared tile)
# Examples: kittens::rt_bf<32,16>, kittens::st_hf<32,64>, rt_fl<32,64>
TK_TILE_PATTERN = r"(?:kittens::)?(?:st|rt)_(?:bf|fl|hf|i8|i32)\s*<[^>]+>"

# ThunderKittens vector types (associated with tiles)
TK_VECTOR_PATTERN = r"::(?:col_vec|row_vec)\b"

# ThunderKittens memory operations (often namespaced)
TK_MEMORY_OPS = [
    r"kittens::load\s*\(",      # Namespaced load
    r"kittens::store\s*\(",     # Namespaced store
    r"\bload\s*\(",             # Tile load (in using namespace context)
    r"\bstore\s*\(",            # Tile store (in using namespace context)
    r"load_async\s*\(",         # Async load
]

# ThunderKittens compute operations (from the manual)
TK_COMPUTE_OPS = [
    r"kittens::(?:warpgroup::)?mma_AB\s*\(",     # Warpgroup MMA: mma_AB
    r"kittens::(?:warpgroup::)?mma_ABt\s*\(",    # MMA variants
    r"kittens::(?:warpgroup::)?mma_AtB\s*\(",
    r"(?:warpgroup::)?mma_AB[t]?\s*\(",          # Without namespace (in using context)
    r"kittens::mul\s*\(",       # Namespaced element-wise ops
    r"kittens::add\s*\(",
    r"kittens::sub\s*\(",
    r"kittens::copy\s*\(",
    r"kittens::zero\s*\(",
    r"\bmul\s*\(",              # Element-wise multiply (in using namespace)
    r"\badd\s*\(",              # Element-wise add
    r"\bsub\s*\(",              # Element-wise subtract
    r"\bcopy\s*\(",             # Copy operation
    r"\bzero\s*\(",             # Zero initialization
]

# ThunderKittens control and utilities
TK_CONTROL_PATTERNS = [
    r"kittens::warpid\s*\(",    # Get warp ID
    r"tma::",                   # Tensor Memory Accelerator namespace
    r"__syncthreads\s*\(",      # Thread synchronization (CUDA primitive often used)
    r"__syncwarp\s*\(",         # Warp synchronization
]

def check_tk_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid ThunderKittens kernel implementation.
    
    Requirements:
    - Must have kittens namespace (kittens::, using namespace kittens)
    - Must have tile declarations (st_bf, rt_fl, st_hf, rt_i8, etc.)
    - Must have memory operations (kittens::load, kittens::store, load_async)
    - Should have compute operations (mma_AB, mul, add, copy, zero, etc.)
    - Optional: warp/warpgroup patterns (kittens::warpgroup, kittens::group<N>)
      for warpgroup-specific operations
    
    ThunderKittens is a tile-based programming model that abstracts
    warp-level operations with register (rt) and shared (st) tiles.
    By default, operations exist at warp-level, so explicit warp/warpgroup
    scope is only needed for warpgroup operations like mma_AB.
    
    Reference: https://github.com/HazyResearch/ThunderKittens/
    """
    code = _strip_comments(code)
    
    # Check for kittens namespace (fundamental requirement)
    has_namespace = any(re.search(p, code) for p in TK_NAMESPACE_PATTERNS)
    if not has_namespace:
        return (True, "Missing kittens namespace (kittens:: or using namespace kittens)")
    
    # Check for tile declarations (rt_* or st_*)
    has_tiles = re.search(TK_TILE_PATTERN, code)
    has_vectors = re.search(TK_VECTOR_PATTERN, code)
    if not (has_tiles or has_vectors):
        return (True, "Missing ThunderKittens tile/vector declarations (rt_bf, st_fl, ::col_vec, etc.)")
    
    # Check for memory operations
    has_memory_ops = any(re.search(p, code) for p in TK_MEMORY_OPS)
    if not has_memory_ops:
        return (True, "Missing ThunderKittens memory operations (kittens::load, kittens::store, load_async)")
    
    # Check for compute operations
    has_compute_ops = any(re.search(p, code) for p in TK_COMPUTE_OPS)
    if not has_compute_ops:
        return (True, "Missing ThunderKittens compute operations (mma_AB, mul, add, copy, zero, etc.)")
    
    return (False, "")


# <========= CUTE/CUTLASS CHECKS =========>
# CUTLASS uses cute:: namespace for tensor operations
# Check: https://github.com/NVIDIA/cutlass 
CUTE_PATTERNS = [
    r"cute::",           # cute:: namespace (CuTe library)
    r"cutlass::",        # cutlass:: namespace
    r"from cutlass",     # Python CUTLASS bindings
]

# CuTe tensor operations
CUTE_TENSOR_OPS = [
    r"make_tensor\s*\(",        # Tensor creation
    r"make_layout\s*\(",        # Layout creation
    r"make_shape\s*\(",         # Shape creation
    r"make_stride\s*\(",        # Stride creation
]

# CuTe/CUTLASS copy operations
CUTE_COPY_OPS = [
    r"copy\s*\(",               # Generic copy
    r"copy_if\s*\(",            # Conditional copy
    r"cute::copy",              # Namespaced copy
    r"Copy_Atom",               # Copy atom template
]

# CUTLASS GEMM patterns
CUTLASS_GEMM_PATTERNS = [
    r"cutlass::gemm",           # GEMM namespace
    r"cutlass::epilogue",       # Epilogue operations
    r"Gemm\w*<",                # GEMM templates (Gemm, GemmUniversal, etc.)
    r"GemmConfiguration",       # GEMM configuration
    r"ThreadblockSwizzle",      # Threadblock scheduling
]

# CUTLASS kernel patterns
CUTLASS_KERNEL_PATTERNS = [
    r"cutlass::arch",           # Architecture-specific code
    r"cutlass::layout",         # Layout specifications
    r"RowMajor|ColumnMajor",    # Layout types
    r"TensorRef\s*<",           # Tensor reference template
]

def check_cute_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid CUTLASS/CuTe kernel implementation.
    
    Requirements:
    - Must have cute:: or cutlass:: namespace (or Python bindings)
    - Must have tensor operations (make_tensor, make_layout) OR
      copy operations (copy, Copy_Atom) OR
      CUTLASS GEMM patterns (cutlass::gemm, Gemm templates)
    
    CuTe is a layout/tensor abstraction library used by CUTLASS 3.x.
    We check for both high-level CUTLASS templates and low-level CuTe ops.
    """
    code = _strip_comments(code)
    
    # Check for namespace
    if not any(p in code for p in ["cute::", "cutlass::", "from cutlass"]):
        return (True, "Missing cute:: or cutlass:: namespace")
    
    # Check for actual operations (tensor, copy, or GEMM)
    has_tensor_ops = any(re.search(p, code) for p in CUTE_TENSOR_OPS)
    has_copy_ops = any(re.search(p, code) for p in CUTE_COPY_OPS)
    has_gemm_patterns = any(re.search(p, code) for p in CUTLASS_GEMM_PATTERNS)
    has_kernel_patterns = any(re.search(p, code) for p in CUTLASS_KERNEL_PATTERNS)
    
    if not (has_tensor_ops or has_copy_ops or has_gemm_patterns or has_kernel_patterns):
        return (True, "Missing CUTLASS/CuTe operations (make_tensor, copy, gemm patterns, etc.)")
    
    return (False, "")


# <========= TILELANG CHECKS =========>
# TileLang uses TVM's T.prim_func decorator
# https://github.com/tile-ai/tilelang
TILELANG_PATTERNS = [
    r"@T\.prim_func",    # TVM primitive function decorator
    r"tvm\.build",       # TVM build call
    r"T\.grid",          # TileLang grid
]

# TileLang/TVM iteration patterns
TILELANG_ITERATION = [
    r"T\.grid\s*\(",            # Grid iteration
    r"T\.serial\s*\(",          # Serial loop
    r"T\.parallel\s*\(",        # Parallel loop
    r"T\.vectorized\s*\(",      # Vectorized loop
    r"T\.unroll\s*\(",          # Unrolled loop
]

# TileLang/TVM buffer operations
TILELANG_BUFFER_OPS = [
    r"T\.alloc_buffer\s*\(",    # Buffer allocation
    r"T\.buffer_store\s*\(",    # Buffer store
    r"T\.buffer_load\s*\(",     # Buffer load
    r"T\.match_buffer\s*\(",    # Buffer matching
]

# TileLang/TVM compute patterns
TILELANG_COMPUTE_PATTERNS = [
    r"T\.reads\s*\(",           # Read annotations
    r"T\.writes\s*\(",          # Write annotations
    r"T\.block\s*\(",           # Computation block
    r"T\.block_attr\s*\(",      # Block attributes
]

# TVM build/compile patterns
TILELANG_BUILD_PATTERNS = [
    r"tvm\.build\s*\(",         # Build function
    r"tvm\.IRModule",           # IR module
    r"tvm\.target\.Target",     # Target specification
]

def check_tilelang_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid TileLang kernel implementation.
    
    Requirements:
    - Must have @T.prim_func decorator
    - Must have iteration constructs (T.grid, T.serial, T.parallel, etc.)
    - Must have buffer operations (T.alloc_buffer, T.buffer_store/load) OR
      compute patterns (T.block, T.reads, T.writes)
    - Should have tvm.build or IRModule for compilation
    
    TileLang is a tensor program DSL built on TVM that uses structured
    iteration spaces and explicit buffer operations.
    """
    code = _strip_comments(code)
    
    # Check for decorator
    if not re.search(r"@T\.prim_func", code):
        return (True, "Missing @T.prim_func decorator")
    
    # Check for iteration constructs
    has_iteration = any(re.search(p, code) for p in TILELANG_ITERATION)
    if not has_iteration:
        return (True, "Missing TileLang iteration constructs (T.grid, T.serial, T.parallel, etc.)")
    
    # Check for buffer operations or compute patterns
    has_buffer_ops = any(re.search(p, code) for p in TILELANG_BUFFER_OPS)
    has_compute_patterns = any(re.search(p, code) for p in TILELANG_COMPUTE_PATTERNS)
    
    if not (has_buffer_ops or has_compute_patterns):
        return (True, "Missing TileLang buffer operations (T.alloc_buffer, T.buffer_store/load) or compute patterns (T.block, T.reads, T.writes)")
    
    # Check for build patterns (optional but recommended)
    has_build = any(re.search(p, code) for p in TILELANG_BUILD_PATTERNS)
    if not has_build:
        return (True, "Missing TVM build/compilation (tvm.build, IRModule)")
    
    return (False, "")


# =============================================================================
# TIMING MANIPULATION CHECKS - Reward Hacking Patterns
# From adversarial hack PR and DeepReinforce blog
# =============================================================================

# <========= STREAM INJECTION CHECKS =========>
# Rationale: Models may use CUDA streams to defer computation, manipulate timing,
# or execute operations asynchronously to cheat benchmark measurements.
STREAM_PATTERNS = [
    r"torch\.cuda\.Stream\s*\(",       # torch.cuda.Stream()
    r"cuda\.Stream\s*\(",               # cuda.Stream() (if imported)
    r"with\s+torch\.cuda\.stream",      # with torch.cuda.stream(...)
    r"\.wait_stream\s*\(",              # .wait_stream() method
    r"\.record_stream\s*\(",            # .record_stream() method
]

def check_stream_injection(code: str) -> Tuple[bool, str]:
    """
    Check for CUDA stream injection patterns.
    
    Detects code that uses CUDA streams to potentially manipulate timing:
    1. Stream creation: torch.cuda.Stream(), cuda.Stream()
    2. Stream context managers: with torch.cuda.stream(...)
    3. Stream synchronization: .wait_stream(), .record_stream()
    
    Rationale: Streams can defer computation or change execution order,
    potentially affecting benchmark timing measurements.
    """
    code = _strip_comments(code)
    
    for pattern in STREAM_PATTERNS:
        if re.search(pattern, code):
            if "wait_stream" in pattern or "record_stream" in pattern:
                return (True, "Uses stream synchronization (potential timing manipulation)")
            elif "with" in pattern:
                return (True, "Uses stream context manager (potential timing manipulation)")
            else:
                return (True, "Uses CUDA streams (potential timing manipulation)")
    
    return (False, "")


# <========= THREAD INJECTION CHECKS =========>
# Rationale: Models may use threading to parallelize work or manipulate execution
# order in ways that could affect benchmark timing.
THREAD_PATTERNS = [
    r"threading\.Thread\s*\(",          # threading.Thread()
    r"import\s+threading",              # import threading
    r"from\s+threading\s+import",       # from threading import ...
    r"multiprocessing\.(Process|Pool|Manager|Queue|Pipe)",
    r"import\s+multiprocessing",        # import multiprocessing
    r"concurrent\.futures",             # concurrent.futures (thread pools)
    r"ThreadPoolExecutor",              # ThreadPoolExecutor
    r"ProcessPoolExecutor",             # ProcessPoolExecutor
]

def check_thread_injection(code: str) -> Tuple[bool, str]:
    """
    Check for thread/multiprocessing injection patterns.
    
    Detects code that uses threading or multiprocessing:
    1. Thread creation: threading.Thread()
    2. Threading imports: import threading
    3. Multiprocessing: Process, Pool, Manager, Queue, Pipe
    4. Concurrent futures: ThreadPoolExecutor, ProcessPoolExecutor
    
    Rationale: Threading can defer computation or change execution order,
    affecting timing measurements.
    
    Note: ProcessPoolExecutor might be legitimate in eval code but should
    not appear in kernel implementations.
    """
    code = _strip_comments(code)
    
    for pattern in THREAD_PATTERNS:
        if re.search(pattern, code):
            if "multiprocessing" in pattern:
                return (True, "Uses multiprocessing (potential timing manipulation)")
            elif "concurrent" in pattern or "Executor" in pattern:
                return (True, "Uses concurrent futures (potential timing manipulation)")
            else:
                return (True, "Uses threading (potential timing manipulation)")
    
    return (False, "")


# <========= LAZY EVALUATION CHECKS =========>
# Rationale: Models may create fake/lazy tensors that don't actually compute
# anything, passing correctness checks without real implementation.
LAZY_TENSOR_PATTERNS = [
    r"_make_subclass",                  # torch.Tensor._make_subclass (common lazy hack)
    r"class\s+\w+.*\(torch\.Tensor\)",  # Custom tensor subclasses
    r"class\s+\w+.*\(Tensor\)",         # Custom tensor subclasses (imported Tensor)
    r"torch\.Tensor\.__new__",          # Direct tensor construction (potential lazy)
]

def check_lazy_eval(code: str) -> Tuple[bool, str]:
    """
    Check for lazy tensor creation patterns.
    
    Detects patterns commonly used to create lazy/fake tensors:
    1. _make_subclass: Common way to create custom tensor subclasses
    2. Custom tensor subclasses: Classes inheriting from torch.Tensor
    3. Direct tensor construction: torch.Tensor.__new__ manipulation
    
    Rationale: Lazy tensors can pass correctness checks without actually
    computing anything, which is a form of reward hacking.
    """
    code = _strip_comments(code)
    
    for pattern in LAZY_TENSOR_PATTERNS:
        if re.search(pattern, code):
            if "_make_subclass" in pattern:
                return (True, "Uses _make_subclass (potential lazy tensor hack)")
            elif "class" in pattern:
                return (True, "Defines custom tensor subclass (potential lazy tensor hack)")
            else:
                return (True, "Uses direct tensor construction (potential lazy tensor hack)")
    
    return (False, "")


# <========= Timing Monkey Patch CHECKS =========>
# Rationale: Models may monkey-patch torch timing functions to fake benchmark results.
# This detects static patterns where timing functions are reassigned.
# especially when relying on timing markers like cuda.Event or time.perf_counter.
TIMING_EVENT_PATCH_PATTERNS = [
    r"torch\.cuda\.Event\.record\s*=",      # torch.cuda.Event.record = ...
    r"torch\.cuda\.Event\.elapsed_time\s*=", # torch.cuda.Event.elapsed_time = ...
    r"torch\.cuda\.synchronize\s*=",         # torch.cuda.synchronize = ...
    r"torch\.cuda\.Event\s*=",               # torch.cuda.Event = ... (replacing class)
    r"time\.perf_counter\s*=",               # time.perf_counter = ...
    r"time\.time\s*=",                       # time.time = ...
]

def check_timing_event_patch(code: str) -> Tuple[bool, str]:
    """
    Check for monkey patching of timing functions (static analysis).
    
    Detects code that reassigns torch or time functions used for benchmarking:
    1. torch.cuda.Event.record = ...
    2. torch.cuda.Event.elapsed_time = ...
    3. torch.cuda.synchronize = ...
    4. time.perf_counter = ...
    
    Rationale: Monkey patching timing functions can fake benchmark results,
    making slow code appear fast.
    """
    code = _strip_comments(code)
    
    for pattern in TIMING_EVENT_PATCH_PATTERNS:
        if re.search(pattern, code):
            return (True, "Reassigns timing function (monkey patch detected)")
    
    return (False, "")


# <========= PRECISION DOWNGRADE CHECKS =========>
# Rationale: Models may downgrade precision internally (FP32 -> FP16)
# to make kernels faster while still returning FP32 output. This is a reward
# hacking pattern where models cheat on timing while appearing correct.
# This might be a valid optimization, but is still worth flagging as a warning.
#
# We use explicit, high-confidence patterns that indicate intentional precision
# downgrading. These patterns have minimal false positives and clear semantic intent.

# Specific patterns that indicate FP32 -> FP16 precision downgrading
FP32_TO_FP16_PATTERNS = [
    # ========== CUDA / CUDA C++ ==========
    # 1.1 Explicit float -> half intrinsics (⭐ gold standard)
    # __float2half(f), __float2half_rn(f)
    r"__float2half(_rn)?\s*\(",
    
    # 1.2 Explicit C-style cast to __half
    # (__half)f
    r"\(\s*__half\s*\)\s*[\w\->\.]+",
    
    # 1.3 static_cast<half> / static_cast<__half>
    # static_cast<half>(f), static_cast<__half>(f)
    r"static_cast\s*<\s*(__half|half)\s*>\s*\(",
    
    # ========== Triton (Python) ==========
    # 2.1 Explicit tl.astype(..., tl.float16) (⭐ best signal)
    # tl.astype(x, tl.float16)
    r"tl\.astype\s*\(\s*[^,]+,\s*tl\.float16\s*\)",
    
    # ========== CUTLASS ==========
    # 3.1 NumericConverter float -> half (⭐ extremely reliable)
    # NumericConverter<half_t, float>
    r"NumericConverter\s*<\s*half_t\s*,\s*float\s*>",
    
    # 3.2 Epilogue output type = FP16
    # LinearCombination<half_t, ...>
    r"LinearCombination\s*<\s*half_t",
    
    # ========== CUTE (when used directly) ==========
    # 4.1 Explicit type conversion utility
    # type_convert<half_t>(x)
    r"type_convert\s*<\s*half_t\s*>\s*\(",
    
    # ========== ThunderKittens ==========
    # 5.1 Explicit half types in cast contexts
    # tk::half(x)
    r"tk::half\s*\(",
    
    # ========== Additional gaping patterns ==========
    # PyTorch explicit casts (common in inline CUDA contexts)
    r"\.half\s*\(",                              # .half() -> FP16
    r"\.float16\s*\(",                           # .float16() -> FP16
    r"\.to\s*\(\s*torch\.(float16|half)\b",      # .to(torch.float16)
    r"\.to\s*\(\s*dtype\s*=\s*torch\.(float16|half)\b",  # .to(dtype=torch.float16)
]

def check_precision_downgrade(code: str, precision: str = "fp32") -> Tuple[bool, str]:
    """
    Check for FP32 -> FP16 precision downgrading patterns in kernel code.
    
    Detects code that downgrades precision from FP32 to FP16 internally,
    which can make kernels faster but with reduced precision. This is a reward
    hacking pattern where models cheat on timing while appearing correct.
    
    Currently only checks for FP32 -> FP16 downgrades. Other precision combinations
    (e.g., FP16 -> BF16) are not flagged.
    
    Args:
        code: Kernel source code
        precision: Required precision - only checks when "fp32" or "float32"
    
    Returns:
        (True, error_message) if FP32 -> FP16 downgrade detected
        (False, "") if no downgrade detected
    
    Examples of detected patterns:
    - .half(), .float16()
    - .to(torch.float16), .to(torch.half)
    - dtype=torch.float16
    - __half, half2 (CUDA)
    - tl.float16 (Triton)
    """
    code = _strip_comments(code)
    precision = precision.lower()
    
    # Normalize precision to standard form
    precision_map = {"fp32": "fp32", "float32": "fp32", "fp16": "fp16", "bf16": "bf16", "bfloat16": "bf16"}
    precision = precision_map.get(precision, precision)
    
    # Only check for FP32 -> FP16 downgrades
    if precision != "fp32":
        return (False, "")
    
    # Check for FP16 patterns
    for pattern in FP32_TO_FP16_PATTERNS:
        if re.search(pattern, code):
            return (True, "Precision downgrade detected: required FP32 but code uses FP16")
    
    return (False, "")

# =============================================================================
# In the future, we can add a AST-based checker and a LM-as-a-judge checker
# =============================================================================


# =============================================================================
# REGISTRY & PRESETS
# =============================================================================

# Check functions can take either (code) or (code, precision) arguments
# Most checks take only code, but precision-dependent checks take both
CHECK_FUNCTIONS: Dict[str, Union[Callable[[str], Tuple[bool, str]], Callable[[str, str], Tuple[bool, str]]]] = {
    # Bypass checks (strict)
    "code_bypass": check_code_bypass,
    "pytorch_wrap": check_pytorch_wrap,
    "timing_event_patch": check_timing_event_patch,  # clearly malicious
    
    # Torch ops (depends on your setups)
    "torch_computation_ops": check_torch_computation_ops,
    
    # Timing manipulation checks (usually warnings)
    "stream_injection": check_stream_injection,
    "thread_injection": check_thread_injection,
    "lazy_eval": check_lazy_eval,
    "precision_downgrade": check_precision_downgrade,  # precision-dependent
    
    # Backend-specific implementation checks
    # should be strict
    "cuda_impl": check_cuda_impl,
    "triton_impl": check_triton_impl,
    "tk_impl": check_tk_impl,
    "cute_impl": check_cute_impl,
    "tilelang_impl": check_tilelang_impl,
}

# Checks that require additional parameters beyond just code
PRECISION_DEPENDENT_CHECKS = {"precision_downgrade"}

# Here are some presets for you to use
# You are welcome to adapt them to your settings
# These checks are NECESSARY for all kernels (strict = error)
STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",  
    "lazy_eval",         
]

# Backend-specific checks are added later at entry point
# per backend implementation check, usually strict
BACKEND_IMPL_CHECK = {
    "cuda": "cuda_impl",
    "triton": "triton_impl",
    "thunderkittens": "tk_impl",
    "cute": "cute_impl",
    "cutlass": "cute_impl",  # alias
    "tilelang": "tilelang_impl",
}

# These are optional checks (by user's decision) - flagged as warnings
# Move to STRICT_CHECKS if you want to enforce them
WARNING_CHECKS: List[str] = [
    # up to user to allow program to still have some torch computation ops
    "pytorch_wrap",
    "torch_computation_ops",  
    "stream_injection",       # could have legitimate uses (async ops), but should be careful!
    "precision_downgrade",    # precision downgrading - can be intentional but often a hack
]


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
        
        # Handle precision-dependent checks
        if check_name in PRECISION_DEPENDENT_CHECKS:
            has_issue, msg = CHECK_FUNCTIONS[check_name](code, precision)
        else:
            has_issue, msg = CHECK_FUNCTIONS[check_name](code)
        
        if has_issue:
            if check_name in forbidden_checks:
                errors.append(msg)
            else:
                warnings_list.append(msg)
    
    valid = len(errors) == 0 # valid if no errors
    return valid, errors, warnings_list
