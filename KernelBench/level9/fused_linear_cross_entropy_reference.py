import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Fused Linear + Cross Entropy Loss.
    In many optimized libraries, this is fused to avoid materializing the full logit tensor.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Hidden states of shape [batch_size, seq_len, hidden_size]
            labels (torch.Tensor): Target labels of shape [batch_size, seq_len]
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Compute logits for all tokens
        logits = self.linear(x) # [batch_size, seq_len, vocab_size]
        # Flatten and compute standard cross entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

# Kernelbench Parameters
batch_size = 16
seq_len = 1024
hidden_size = 1024
vocab_size = 32000 # Example vocabulary size

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    # Generate random integer labels in range [0, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return [x, labels]

def get_init_inputs():
    return [hidden_size, vocab_size]
