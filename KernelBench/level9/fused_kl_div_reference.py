import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation of Fused KL Divergence Loss.
    Computes KL divergence between student logits and teacher logits.
    Logits are expanded from hidden states on the fly to save memory.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.student_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.teacher_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Student hidden states [N, H]
            target_x (torch.Tensor): Teacher hidden states [N, H]
        Returns:
            torch.Tensor: Scalar KL divergence loss
        """
        # In the fused version, we calculate logits in chunks.
        # Here we do it at once for the reference.
        student_logits = self.student_head(x)
        teacher_logits = self.teacher_head(target_x)
        
        # KL Divergence: KL(Teacher || Student)
        # Loss = sum(P_teacher * (log P_teacher - log P_student))
        
        log_p_student = F.log_softmax(student_logits, dim=-1)
        log_p_teacher = F.log_softmax(teacher_logits, dim=-1)
        p_teacher = F.softmax(teacher_logits, dim=-1)
        
        kl_div = p_teacher * (log_p_teacher - log_p_student)
        return kl_div.sum(dim=-1).mean()

# Kernelbench Parameters
batch_size = 16
seq_len = 1024
hidden_size = 1024
vocab_size = 32000

def get_inputs():
    x = torch.randn(batch_size * seq_len, hidden_size)
    target_x = torch.randn(batch_size * seq_len, hidden_size)
    return [x, target_x]

def get_init_inputs():
    return [hidden_size, vocab_size]
