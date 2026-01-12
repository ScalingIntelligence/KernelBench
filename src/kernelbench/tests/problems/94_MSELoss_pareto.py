import torch
import torch.nn as nn

def sample_pareto(shape, scale=0.01, alpha=1.5):
    u = torch.rand(shape)
    return scale / u.pow(1 / alpha)

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
    predictions = sample_pareto((batch_size, *input_shape))
    targets = sample_pareto((batch_size, *input_shape))
    return [predictions, targets]

def get_init_inputs():
    return []

