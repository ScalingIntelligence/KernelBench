import torch
import torch.nn as nn

class L2WrapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, logits, penalty_factor=1e-4):
        # Find max logit per token for the penalty
        # This is a trick used in some models to prevent logit drift
        maxx, ids = torch.max(logits, dim=-1, keepdim=True)
        ctx.logits_shape = logits.shape
        # Average penalty over batch and sequence
        factor = penalty_factor / (logits.shape[0] * logits.shape[1])
        ctx.save_for_backward(maxx * factor, ids)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        maxx_scaled, ids = ctx.saved_tensors
        glogits = torch.zeros(ctx.logits_shape, device=grad_output.device, dtype=grad_output.dtype)
        # Scatter the scaled max value back to the winning logit's position
        glogits.scatter_(-1, ids, maxx_scaled)
        # grad_output is the gradient of the loss
        return grad_output, glogits, None

class Model(nn.Module):
    """
    Reference implementation of L2Wrap.
    Maintains the loss value in forward but adds a logit-dependent penalty to the gradient.
    """
    def __init__(self, penalty_factor: float = 1e-4):
        super(Model, self).__init__()
        self.penalty_factor = penalty_factor

    def forward(self, loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.Tensor): Scalar loss tensor.
            logits (torch.Tensor): Logits tensor of shape [B, T, V].
        Returns:
            torch.Tensor: The same scalar loss tensor.
        """
        return L2WrapFunction.apply(loss, logits, self.penalty_factor)

# Kernelbench Parameters
batch_size = 8
seq_len = 1024
vocab_size = 32000

def get_inputs():
    # Loss is usually a single scalar
    loss = torch.tensor(2.5, requires_grad=True)
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    return [loss, logits]

def get_init_inputs():
    return [1e-4] # penalty_factor
