
"""
TODO: ongoing effort

Code validity checker 
- Make sure it is actually CUDA / DSL code (valid)


Call site:
- generation
- eval -> throw error to reject 
"""

import re

def check_valid_kernel_code(code: str, backend: str) -> bool:
    """
    Check if the provided code is valid CUDA/DSL code for the given backend.
    """
    # TODO: Implement actual code validation logic
    match backend:
        case "cuda":
            # Validate CUDA-specific syntax
            pass
        case "triton":
            # Validate Triton-specific syntax
            pass
        case "thunderkittens":
            return check_thunderkittens_code(code)
        case _:
            # TO ADD MORE
            # Unknown backend
            return False
    return True

def check_valid_cuda(code: str) -> bool:
    """
    Check if the provided code is valid CUDA/DSL code.
    This is a placeholder for the actual implementation.
    """

    # TODO: Implement actual code validation logic

    # migrate from Kevin implementations

    return True

# list torch ops that are allowed
# we probably won't allow torch operations 

# decide degree on the torch code allowed
# do we all in cuda? do we want part of it to be in cuda?

def check_triton_code(code: str) -> bool:
    """
    Check if the provided code is valid Triton code.
    """
    # TODO: Implement actual code validation logic
    # detect it is triton jit 
    # 
    return True

def check_thunderkittens_code(code: str) -> bool:
    """
    Check if the provided code is valid ThunderKittens code.
    
    Uses the following heuristics that the code:
    1. Contains ThunderKittens-specific namespace patterns:
       - "kittens::warp" or "kittens::warpgroup" or "::warpgroup::" or "::warp::" or "tma::" TODO: Get a big namespace list!
    2. Contains tile declarations:
       - st_{bf/fl}<...> (shared memory tiles)
       - rt_{bf/fl}<...> (register tiles)
    """
    # (1) Namespace patterns to search for: if it contains a single one of these, then it's valid!
    # TODO: For more complicated programs, you really want to make sure it's following the producer-consumer pattern
    #   - could search specifically for "producer" and "consumer" structs, or the presence of "tma::load_async"
    warp_patterns = [
        r"kittens::warp\b",
        r"kittens::warpgroup\b",
        r"::warpgroup::",
        r"::warp::",
        r"warpgroup::",
        r"warp::"
    ]
    has_warp_pattern = any(re.search(pattern, code) for pattern in warp_patterns)
    if not has_warp_pattern:
        return False
    
    # (2) Check that the file actually uses tiles: st_<type><...> or rt_<type><...>
    # Pattern matches: st_bf<...>, st_fl<...>, rt_bf<...>, rt_fl<...>, etc. (any type)
    # TODO: we don't look for global gl here...
    # Also handles namespaced versions like kittens::st_bf<...>
    tile_pattern = r"(?:kittens::)?(?:st|rt)_\w+\s*<[^>]+>"
    has_tiles = bool(re.search(tile_pattern, code))
    if not has_tiles:
        return False

    # # (3) Producer-Consumer semantics
    # pc_patterns = [
    #     r"tma::\b",
    #     r"load_async\b"
    # ]
    # has_pc_pattern = any(re.search(pattern, code) for pattern in pc_patterns)
    # if not has_pc_pattern:
    #     return False

    passes_compilation = False
    # TODO: Try compilation here and return false if it fails

    
    # ALL of the above conditions must be met for this code string to pass
    return True