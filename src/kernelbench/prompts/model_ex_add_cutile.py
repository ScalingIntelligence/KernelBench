import torch
import torch.nn as nn
import cuda.tile as ct

TILE_SIZE = 256


@ct.kernel
def add_kernel(a, b, result):
    """
    cuTile kernel for adding two dense tensors element-wise.
    Each block processes TILE_SIZE elements.
    """
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using cuTile kernel for elementwise addition.
        
        Args:
            a: First input tensor on CUDA
            b: Second input tensor on CUDA (same shape as a)
            
        Returns:
            Result tensor of a + b
        """
        assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
        a = a.contiguous()
        b = b.contiguous()

        # Store original shape for reshaping back
        original_shape = a.shape

        # Flatten tensors for 1D processing
        a_flat = a.view(-1)
        b_flat = b.view(-1)

        # Allocate output tensor
        result = torch.empty_like(a_flat)

        # Calculate grid dimensions
        n_elements = a_flat.shape[0]
        grid = (ct.cdiv(n_elements, TILE_SIZE), 1, 1)

        # Get current CUDA stream
        stream = torch.cuda.current_stream()._as_parameter_

        # Launch the kernel
        ct.launch(stream, grid, add_kernel, (a_flat, b_flat, result))

        # Reshape back to original shape
        return result.view(original_shape)