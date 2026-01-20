import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation of GRPO (Group Relative Policy Optimization) Loss.
    This module computes the policy loss and KL divergence term for a given batch of completions.
    """
    def __init__(self, beta: float = 0.1):
        super(Model, self).__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor, ref_logp: torch.Tensor, input_ids: torch.Tensor, advantages: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): [B, L, V] Model logits for the completions
            ref_logp (torch.Tensor): [B, L] Reference model log probabilities
            input_ids (torch.Tensor): [B, L] Actual token IDs for the completions
            advantages (torch.Tensor): [B] Group relative advantages
            completion_mask (torch.Tensor): [B, L] Mask for valid completion tokens
        Returns:
            torch.Tensor: Scalar GRPO loss
        """
        # 1. Get per-token log probabilities from logits and input_ids
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        
        # 2. Compute KL divergence: KL(Ref || Policy) = exp(ref_logp - logp) - (ref_logp - logp) - 1
        # This is a common approximation used in GRPO/PPO
        diff = ref_logp - per_token_logps
        per_token_kl = torch.exp(diff) - diff - 1
        
        # 3. Compute the policy loss part
        # loss = - (exp(logp - logp_old) * advantage - beta * kl)
        # Assuming logp_old = logp.detach() for the first iteration step
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        # 4. Mask and reduce
        masked_loss = per_token_loss * completion_mask
        # Average over valid completion tokens per sequence, then average over batch
        loss = (masked_loss.sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        return loss

# Kernelbench Parameters
batch_size = 16
seq_len = 128
vocab_size = 32000

def get_inputs():
    logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_logp = torch.randn(batch_size, seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    advantages = torch.randn(batch_size)
    completion_mask = torch.ones(batch_size, seq_len)
    return [logits, ref_logp, input_ids, advantages, completion_mask]

def get_init_inputs():
    return [0.1] # beta
