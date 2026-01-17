import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extensions as tlx


@triton.jit
def tlx_add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Create barriers for synchronization
    b0 = tlx.barrier_create()
    b1 = tlx.barrier_create()
    
    phase = 0
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tlx.barrier_wait(bar=b1, phase=phase ^ 1)

            # Placeholder block to do something

            tlx.barrier_arrive(bar=b0)  # Release

        with tlx.async_task(num_warps=4):
            tlx.barrier_wait(bar=b0, phase=phase)  # Wait

            # Some arith ops
            block_start = tl.program_id(0) * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            z = x + y
            tl.store(out_ptr + offsets, z, mask=mask)

            tlx.barrier_arrive(bar=b0)  # Wait


def tlx_add(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the TLX kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the TLX kernel.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the TLX kernel
    tlx_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our TLX-based addition
        return tlx_add(a, b)

