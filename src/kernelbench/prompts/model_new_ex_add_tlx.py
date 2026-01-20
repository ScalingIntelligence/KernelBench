import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

@triton.jit
def add_warp_specialized_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Split the block work across two async tasks to demonstrate specialization
    # Task 1: Process first half of the block
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offsets = block_start + tl.arange(0, BLOCK_SIZE // 2)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(z_ptr + offsets, x + y, mask=mask)
            
        # Task 2: Process second half of the block with 4 warps
        with tlx.async_task(num_warps=4):
            offsets = block_start + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(z_ptr + offsets, x + y, mask=mask)

def add_warp_specialized(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # Ensure BLOCK_SIZE is even for the split
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_warp_specialized_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return add_warp_specialized(a, b)
