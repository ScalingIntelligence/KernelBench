import torch
import torch.nn as nn

from torch.distributions import Pareto

class Model(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.smooth_l1_loss(predictions, targets)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    predictions = Pareto(0.01, 1.5).sample((batch_size, *input_shape))
    targets = Pareto(0.01, 1.5).sample((batch_size, *input_shape))
    return [predictions*scale, targets]

def get_init_inputs():
    return []
