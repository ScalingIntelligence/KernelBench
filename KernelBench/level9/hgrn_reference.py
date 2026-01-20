import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Reference implementation of HGRN (Hierarchically Gated Recurrent Network).
    Adapted for kernelbench reference format.
    """
    def __init__(self, hidden_size: int = 1024, expand_ratio: int = 1, use_short_conv: bool = True, conv_size: int = 4):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size

        # Linear projections for i (input), f (forget gate), and g (output gate)
        self.i_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, self.input_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.input_dim, bias=False)

        if use_short_conv:
            # Depthwise convolutions for i and f to provide some local temporal context
            self.i_conv1d = nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.input_dim,
                kernel_size=conv_size,
                groups=self.input_dim,
                bias=False,
                padding=conv_size - 1,
            )
            self.f_conv1d = nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.input_dim,
                kernel_size=conv_size,
                groups=self.input_dim,
                bias=False,
                padding=conv_size - 1,
            )

        # Gated RMSNorm weight
        self.g_norm_weight = nn.Parameter(torch.ones(self.input_dim))
        self.norm_eps = 1e-5

        # Output projection
        self.o_proj = nn.Linear(self.input_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HGRN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, hidden_size).
        """
        B, T, H = x.shape
        
        # 1. Projections
        i_raw = self.i_proj(x)
        f_raw = self.f_proj(x)
        g_raw = self.g_proj(x)
        
        # 2. Short Convolution (Causal 1D Depthwise)
        if self.use_short_conv:
            # i_raw: [B, T, D] -> [B, D, T]
            i_raw = i_raw.transpose(1, 2)
            # nn.Conv1d with padding=K-1 and taking [:T] is causal
            i_raw = self.i_conv1d(i_raw)[..., :T].transpose(1, 2)
            f_raw = f_raw.transpose(1, 2)
            f_raw = self.f_conv1d(f_raw)[..., :T].transpose(1, 2)
            
        # 3. HGRN Gates
        # f_log: log of the forget gate: logsigmoid(f_raw)
        # forget_gate: sigmoid(f_raw)
        f_log = F.logsigmoid(f_raw)
        forget_gate = f_log.exp() 
        # i: input modulated by silu activation and (1 - forget_gate)
        # Matches swiglu(i_raw, 1 - exp(f_log)) in the original code
        i = F.silu(i_raw) * (1.0 - forget_gate)
        
        # 4. Recurrence: h_t = forget_gate_t * h_{t-1} + i_t
        # Using a loop for numerical stability as this is a reference implementation.
        h = torch.zeros(B, self.input_dim, device=x.device, dtype=x.dtype)
        o_rec = torch.zeros(B, T, self.input_dim, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            h = forget_gate[:, t] * h + i[:, t]
            o_rec[:, t] = h
            
        # 5. Gated RMSNorm
        # Formula: RMSNorm(h) * weight * silu(g_raw)
        # Normalized by the hidden dimension
        rms = torch.rsqrt(o_rec.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        o_norm = o_rec * rms * self.g_norm_weight
        o = o_norm * F.silu(g_raw)
        
        # 6. Final output projection
        o = self.o_proj(o)
        
        return o

# Hyperparameters
hidden_size = 1024
expand_ratio = 1
use_short_conv = True
conv_size = 4

# Test dimensions
batch_size = 8
seq_len = 2048

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size, expand_ratio, use_short_conv, conv_size]
