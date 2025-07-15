import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from torch.nn.functional import silu, softplus, cross_entropy, softmax
from typing import Tuple, List, Dict, Generator

class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.scale = d ** 0.5
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return self.g * (x / (x.norm(dim=-1, keepdim=True) + 1e-5)) * self.scale

def pscan(A: Tensor, X: Tensor) -> Tensor:
    """Parallel scan for computing hidden states efficiently."""
    return rearrange(A * X, 'l b d s -> b l d s')

class MambaBlock(nn.Module):
    def __init__(
        self, 
        d_input: int,
        d_model: int,
        d_state: int = 16,
        d_discr: int = None,
        ker_size: int = 4,
        parallel: bool = False,
    ) -> None:
        super().__init__()
        
        d_discr = d_discr or d_model // 16
        
        self.in_proj = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(
            nn.Linear(d_model, d_discr, bias=False),
            nn.Linear(d_discr, d_model, bias=False),
        )
        
        self.conv = nn.Conv1d(
            d_model, d_model, ker_size,
            padding=ker_size - 1, groups=d_model, bias=True
        )
        
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.parallel = parallel
        
    def forward(self, seq: Tensor, cache: Tuple = None) -> Tuple[Tensor, Tuple]:
        b, l, d = seq.shape
        prev_hid, prev_inp = (None, None) if cache is None else cache
        
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        
        x = rearrange(a, 'b l d -> b d l')
        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)
        a = self.conv(x)[..., :l]
        a = rearrange(a, 'b d l -> b l d')
        
        a = silu(a)
        a, hid = self.ssm(a, prev_hid)
        b = silu(b)
        
        out = a * b
        out = self.out_proj(out)
        
        if cache is not None:
            cache = (hid.squeeze(), x[..., 1:])
        
        return out, cache

    def ssm(self, seq: Tensor, prev_hid: Tensor = None) -> Tuple[Tensor, Tensor]:
        A = -self.A
        D = self.D
        
        B = self.s_B(seq)
        C = self.s_C(seq)
        Δ = softplus(D + self.s_D(seq))
        
        A_bar = einsum(torch.exp(A), Δ, 'd s, b l d -> b l d s')
        B_bar = einsum(B, Δ, 'b l s, b l d -> b l d s')
        X_bar = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        
        hid = self._hid_states(A_bar, X_bar, self.parallel, prev_hid)
        out = einsum(hid, C, 'b l d s, b l s -> b l d')
        out = out + D * seq
        
        return out, hid
    
    def _hid_states(self, A: Tensor, X: Tensor, parallel: bool = False, prev_hid: Tensor = None) -> Tensor:
        b, l, d, s = A.shape
        A = rearrange(A, 'b l d s -> l b d s')
        X = rearrange(X, 'b l d s -> l b d s')
        
        if prev_hid is not None:
            return rearrange(A * prev_hid + X, 'l b d s -> b l d s')
            
        h = None if parallel else torch.zeros(b, d, s, device=self.device)
        return pscan(A, X) if parallel else torch.stack([
            h := A_t * h + X_t for A_t, X_t in zip(A, X)
        ], dim=1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        num_layers: int = 8,
        d_input: int = 1024,
        d_model: int = 1024,
        d_state: int = 16,
        d_discr: int = None,
        ker_size: int = 4,
        parallel: bool = False,
    ) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_input)
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MambaBlock(d_input, d_model, d_state, d_discr, ker_size, parallel),
                RMSNorm(d_input)
            ])
            for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(d_input, vocab_size, bias=False)
        
    def forward(self, tok: Tensor, cache: Tuple = None) -> Tuple[Tensor, Tuple]:
        tok = torch.atleast_2d(tok)
        seq = self.embedding(tok)
        
        for mamba, norm in self.layers:
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
            
        return self.head(seq)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device 
    
batch_size = 8
seq_len = 1024

def get_inputs():
    return [torch.randint(0, 16384, (2, 128))]

def get_init_inputs():
    return []

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    model = Model()
    print(model(*get_inputs()).size())