import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation for Fused Cross Entropy Loss with extra features:
    - Label Smoothing
    - Logit Scaling
    - LSE Square Scale (z-loss)
    - Ignore Index
    """
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        reduction: str = "mean"
    ):
        super(Model, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): [batch_size, num_classes]
            target (torch.Tensor): [batch_size]
        Returns:
            torch.Tensor: Reduced scalar loss
        """
        # 1. Apply logit scaling
        logits = logits.float() * self.logit_scale
        
        # 2. Compute log-sum-exp for z-loss
        # z-loss = lse_square_scale * (lse(logits)^2)
        lse = torch.logsumexp(logits, dim=-1)
        z_loss = self.lse_square_scale * lse.pow(2)
        
        # 3. Compute cross entropy loss with label smoothing
        # Note: F.cross_entropy handles label_smoothing directly in modern PyTorch
        loss = F.cross_entropy(
            logits, 
            target, 
            ignore_index=self.ignore_index, 
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # 4. Combine with z-loss
        # Apply mask for ignore_index to z_loss as well
        mask = (target != self.ignore_index).float()
        total_loss = (loss + z_loss) * mask
        
        if self.reduction == 'mean':
            return total_loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

# Kernelbench Parameters
batch_size = 1024
num_classes = 32000

def get_inputs():
    logits = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    # Add some ignore_index entries
    target[target % 10 == 0] = -100
    return [logits, target]

def get_init_inputs():
    # ignore_index, label_smoothing, logit_scale, lse_square_scale
    return [-100, 0.1, 1.0, 1e-4]
