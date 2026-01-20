import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper functions from fla/layers/mamba2.py
def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    # Assumes that we only have tensors of either size 4 or 3
    if len(input_tensor.shape) == 4:
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0)
    else:
        pad_shape = (0, 0, 0, pad_size, 0, 0)
    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)

def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] ->
        # [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3],
        )

def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum

class RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, norm_before_gate=False):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, z=None):
        dtype = x.dtype
        weight = self.weight.float()
        x = x.float()
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z.float())
        
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + self.eps)
        out = x * rstd * weight
        
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z.float())
        return out.to(dtype)

class Model(nn.Module):
    """
    Reference implementation of Mamba-2 (Linear Attention / SSD)
    """
    def __init__(
        self,
        num_heads: int = 64,
        head_dim: int = 64,
        hidden_size: int = 2048,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        chunk_size: int = 256,
    ):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.expand = expand
        self.intermediate_size = int(expand * hidden_size)
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.conv_kernel_size = 4
        
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=True)
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1).float()))
        self.norm = RMSNormGated(self.intermediate_size, norm_before_gate=False)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input of shape (batch, seq_len, hidden_size)
        Returns:
            torch.Tensor: Output of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size -
                 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1,
        )

        # 2. Convolution sequence transformation
        hidden_states_B_C = F.silu(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # 3. SSM transformation (SSD naive implementation)
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)
        
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
        C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
        
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        padded_hidden = pad_tensor_by_size(hidden_states, pad_size)
        D_residual = self.D.view(1, 1, self.num_heads, 1) * padded_hidden

        hidden_states = padded_hidden
        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt

        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Intra-chunk (diagonal blocks)
        L = torch.exp(segment_sum(A))
        G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

        # 2. Intra-chunk state (right term of factorization)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

        # 3. Inter-chunk SSM recurrence
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states = new_states[:, :-1]

        # 4. State -> output conversion per chunk (left term of factorization)
        state_decay_out = torch.exp(A_cumsum)
        C_times_states = (C[..., None, :] * states[:, :, None, ...])
        Y_off = (C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None])

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        return self.out_proj(scan_output.to(dtype))

# Configuration for get_inputs/get_init_inputs
batch_size = 4
seq_len = 1024
num_heads = 64
head_dim = 64
hidden_size = 2048
state_size = 128
expand = 2
n_groups = 1
chunk_size = 256

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    return [hidden_states]

def get_init_inputs():
    return [num_heads, head_dim, hidden_size, state_size, expand, n_groups, chunk_size]
