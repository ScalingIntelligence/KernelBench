import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
head_dimension = 32

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, head_dimension)
    K = torch.rand(batch_size, num_heads, sequence_length, head_dimension)
    V = torch.rand(batch_size, num_heads, sequence_length, head_dimension)
    return [Q, K, V]

def get_init_inputs():
    return []
