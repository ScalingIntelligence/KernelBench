import torch
import torch.nn as nn

def sample_pareto(shape, scale=0.01, alpha=1.5):
    u = torch.rand(shape).clamp(min=1e-6)
    return scale / u.pow(1 / alpha)

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    predictions = sample_pareto((batch_size, *input_shape))
    targets = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    return [predictions, targets]

def get_init_inputs():
    return []

