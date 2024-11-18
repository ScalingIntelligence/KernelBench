import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that gathers values from an input tensor along a specified dimension using an index tensor.

    Parameters:
        dim (int): The dimension along which to gather values.
    """
    
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, index):
        """
        Gather values along `dim` using the index tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            index (torch.LongTensor): Index tensor with values to gather along `dim`.
        
        Returns:
            torch.Tensor: Gathered tensor of the same shape as `index`.
        """
        return torch.gather(x, self.dim, index)

# Define input dimensions and parameters
batch_size = 128
input_shape = (8, 10)  # Example shape (arbitrary)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    index = torch.randint(0, input_shape[dim], (batch_size, *input_shape[:dim], *input_shape[dim:]))
    return [x, index]

def get_init_inputs():
    return [dim]