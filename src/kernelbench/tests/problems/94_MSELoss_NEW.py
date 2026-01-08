import torch
import torch.nn as nn

from torch.distributions import Normal

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

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    m1, m2 = torch.rand(2)
    s1, s2 = torch.rand(2) + 0.1
    predictions = Normal(m1, s1).sample((batch_size, *input_shape))
    targets = Normal(m2, s2).sample((batch_size, *input_shape))
    return [predictions*scale, targets]

def get_init_inputs():
    return []
