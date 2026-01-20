import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool = True,
        activation: str = 'tanh',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Not supported activation `{activation}`.")

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            self.activation,
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.lora[0].weight)
        shape = self.lora[2].weight.shape
        weight_fp32 = torch.zeros(shape)
        gain = math.sqrt(shape[1] / shape[0]) if shape[1] > shape[0] else 1
        nn.init.orthogonal_(weight_fp32, gain=gain * 0.1)
        self.lora[2].weight.data.copy_(weight_fp32.to(self.lora[2].weight.dtype))
        if self.lora[2].bias is not None:
            nn.init.zeros_(self.lora[2].bias)

    def set_bias_value(self, value):
        if self.bias and self.lora[2].bias is not None:
            if isinstance(value, torch.Tensor):
                self.lora[2].bias.data.copy_(value.to(self.lora[2].bias.dtype))
            else:
                nn.init.constant_(self.lora[2].bias, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)

class Model(nn.Module):
    """
    Reference implementation of RWKV7 Linear Attention.
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        head_dim: int = 64,
        layer_idx: int = 0,
        num_hidden_layers: int = 24,
    ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # LoRA dimensions as per RWKV7 implementation
        factor = head_dim / 64
        decay_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        gate_low_rank_dim = max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
        a_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        v_low_rank_dim = max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))

        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(hidden_size))
        self.k_a = nn.Parameter(torch.zeros(hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, head_dim))

        self.r_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, hidden_size, low_rank_dim=decay_low_rank_dim, activation='tanh')
        self.a_lora = LoRA(hidden_size, hidden_size, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, hidden_size, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        if layer_idx != 0:
            self.v_lora = LoRA(hidden_size, hidden_size, low_rank_dim=v_low_rank_dim, activation=None)

        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads,
            num_channels=hidden_size,
            eps=head_dim * 1e-5,
            affine=True,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)

        ddd = torch.ones(1, 1, self.hidden_size)
        www = torch.zeros(self.hidden_size)
        zigzag = torch.zeros(self.hidden_size)
        linear = torch.zeros(self.hidden_size)
        for n in range(self.hidden_size):
            linear[n] = n / (self.hidden_size-1) - 0.5
            zigzag[n] = ((n % self.head_dim) - ((self.head_dim-1) / 2)) / ((self.head_dim-1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (self.hidden_size - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            ddd[0, 0, n] = n / self.hidden_size

        self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        nn.init.constant_(self.k_a, 1.02)
        nn.init.constant_(self.r_k, -0.04)
        self.k_k.data.copy_(0.71 - linear*0.1)
        self.w_lora.set_bias_value(www + 0.5 + zigzag*2.5)
        self.a_lora.set_bias_value(-0.19 + zigzag*0.3 + linear*0.4)

        if self.layer_idx != 0:
            self.v_lora.set_bias_value(0.73 - linear*0.4)

        self.g_norm.weight.data[:] = ((self.layer_idx + 1) / self.num_hidden_layers) ** 0.7
        nn.init.orthogonal_(self.r_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight, gain=0.1)
        nn.init.orthogonal_(self.v_proj.weight)
        self.o_proj.weight.data.zero_()

    def forward(self, hidden_states: torch.Tensor, v_first: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input states of shape (B, T, H*D).
            v_first (torch.Tensor): Value from the first layer of shape (B, T, H*D).
        Returns:
            torch.Tensor: Output states of shape (B, T, H*D).
        """
        B, T, C = hidden_states.shape
        H = self.num_heads
        D = self.head_dim

        # Token shift (time_shift - current)
        shifted = F.pad(hidden_states, (0, 0, 1, -1))
        delta = shifted - hidden_states

        # Fused addcmul equivalent
        xr = hidden_states + delta * self.x_r
        xw = hidden_states + delta * self.x_w
        xk = hidden_states + delta * self.x_k
        xv = hidden_states + delta * self.x_v
        xa = hidden_states + delta * self.x_a
        xg = hidden_states + delta * self.x_g

        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        # K update: k = k * (1 + (a - 1) * k_a)
        k = k + k * (a - 1) * self.k_a
        
        # kk needs the k BEFORE the fused_k_rwkv7 update
        k_for_kk = self.k_proj(xk)
        # kk = F.normalize(rearrange(k_for_kk * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)
        kk = F.normalize((k_for_kk * self.k_k).view(B, T, H, D), dim=-1, p=2.0)

        # Reshape for linear attention
        r = r.view(B, T, H, D)
        w = w.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)
        a_lora = a.view(B, T, H, D)
        
        # Attention core (Recurrence)
        # H_t = diag(exp(w_t)) H_{t-1} + (kk_t * a_t) @ (-kk_t^T @ H_{t-1}) + k_t @ v_t^T
        a_recur = -kk
        b_recur = kk * a_lora
        
        out = torch.zeros_like(v)
        state = torch.zeros(B, H, D, D, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for t in range(T):
            r_t = r[:, t] # (B, H, D)
            w_t = w[:, t] # (B, H, D)
            k_t = k[:, t] # (B, H, D)
            v_t = v[:, t] # (B, H, D)
            a_t = a_recur[:, t] # (B, H, D)
            b_t = b_recur[:, t] # (B, H, D)
            
            # state: (B, H, D, D)
            # exp(w_t): (B, H, D)
            state = state * torch.exp(w_t).unsqueeze(-1)
            
            # term_a = a_t^T @ state
            # a_t: (B, H, D), state: (B, H, D, D) -> (B, H, 1, D) @ (B, H, D, D) -> (B, H, 1, D)
            term_a = torch.matmul(a_t.unsqueeze(-2), state)
            
            # state = state + b_t @ term_a
            # b_t: (B, H, D), term_a: (B, H, 1, D) -> (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
            state = state + torch.matmul(b_t.unsqueeze(-1), term_a)
            
            # state = state + k_t @ v_t^T
            # k_t: (B, H, D), v_t: (B, H, D) -> (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
            state = state + torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
            
            # out_t = r_t @ state -> (B, H, 1, D) @ (B, H, D, D) -> (B, H, 1, D)
            out_t = torch.matmul(r_t.unsqueeze(-2), state)
            out[:, t] = out_t.squeeze(-2)

        o = out.reshape(B, T, -1)
        
        # Norm and output correction
        o = self.g_norm(o.transpose(1, 2)).transpose(1, 2)
        
        r_k_b = self.r_k.view(1, 1, H, D)
        correction_term = (r * k * r_k_b).sum(-1, keepdim=True) * v
        o = (o + correction_term.reshape(B, T, -1)) * g
        
        o = self.o_proj(o)
        return o

# Benchmarking parameters
B = 8
T = 64 # Small sequence for reference implementation (O(T) loop)
C = 1024
H = 16
D = 64

def get_inputs():
    hidden_states = torch.randn(B, T, C)
    v_first = torch.randn(B, T, C)
    return [hidden_states, v_first]

def get_init_inputs():
    return [C, D, 0, 24]
