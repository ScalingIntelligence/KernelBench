
"""
TODO: ongoing effort

Code validity checker 
- Make sure it is actually CUDA / DSL code (valid)


Call site:
- generation
- eval -> throw error to reject 
"""

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