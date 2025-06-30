import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    scale = torch.randn(())
    return [torch.randn(batch_size, *input_shape)*scale, torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []
