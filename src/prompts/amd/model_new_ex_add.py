import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ctypes
from hip import hip, hiprtc

"""
AMD (HiP) version for inline compilation
"""

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(str(err))
    return result

# Define the custom HiP kernel for element-wise addition
elementwise_add_source = b"""
extern "C" __global__ void elementwise_add(const float* a, const float* b, float* out, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] + b[tid];
    }
}
"""

# Compilation
# Following example: https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html#hiprtc-launch-kernel-args


# Compile HIP kernel once and store handles
def compile_hip_add_kernel():
    prog = hip_check(hiprtc.hiprtcCreateProgram(elementwise_add_source, b"elementwise_add", 0, [], []))

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props,0))
    arch = props.gcnArchName

    print(f"Compiling kernel for {arch}")

    cflags = [b"--offload-arch="+arch]
    err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
        raise RuntimeError(log.decode())
    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"elementwise_add"))

    return prog, module, kernel


# Call to compilation 
prog, module, kernel = compile_hip_add_kernel()

def hip_element_wise_add(a, b, kernel):
    """
    Performs element-wise addition of two PyTorch tensors using a HIP kernel.
    
    Args:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor 
        kernel: Compiled HIP kernel function
        
    Returns:
        torch.Tensor: Result of element-wise addition
    """
    # Validate inputs
    assert a.shape == b.shape, "Input tensors must have the same shape"
    size = a.numel()
    
    # Ensure inputs are contiguous float tensors
    a = a.contiguous().float()
    b = b.contiguous().float()
    
    # Allocate output tensor
    out = torch.empty_like(a)
    
    # Get pointers to GPU memory
    a_ptr = ctypes.c_void_p(a.data_ptr())
    b_ptr = ctypes.c_void_p(b.data_ptr())
    out_ptr = ctypes.c_void_p(out.data_ptr())
    
    # Configure kernel launch parameters
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block
    
    block = hip.dim3(threads_per_block, 1, 1)
    grid = hip.dim3(blocks, 1, 1)

    # Launch kernel with parameters
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=(
                a_ptr,  # First input tensor
                b_ptr,  # Second input tensor
                out_ptr,  # Output tensor
                ctypes.c_int(size)  # Total number of elements
            )
        )
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, a, b):
        return hip_element_wise_add(a, b, self.kernel)

    def __del__(self):
        if hasattr(self, 'module'):
            hip_check(hip.hipModuleUnload(self.module))