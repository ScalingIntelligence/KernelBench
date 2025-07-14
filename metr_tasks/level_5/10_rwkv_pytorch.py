# Copyright (C) Ronsor Labs. All rights reserved.
#
# The license of this software is specified in the LICENSE file at the root of
# this repository.
#
# For the PyTorch WKV implementation,
# License: Apache-2.0
# From: https://github.com/RWKV/RWKV-infctx-trainer/blob/main/RWKV-v6/src/module/rwkv_inner.py @ 2908b589

# RWKV x060 implementation

# We can use one of the following WKV6 kernels:
# - Pure-Python/PyTorch implementation
# - The official CUDA kernel
# - The Triton kernel from Flash Linear Attention (FLA)
# We try the FLA backend first (if available), followed by CUDA (if available), and fall back to the
# optimized pure-PyTorch implementation.
# You can change the backend order with the wkv6_kernel() context manager.

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from contextlib import contextmanager
from einops import einsum, rearrange
from enum import Enum
from torch.utils.cpp_extension import load
from types import SimpleNamespace

WKVBackend = Enum('WKVBackend', ['FLA', 'CUDA', 'PYTORCH_OPTIMIZED'])

_wkv6_config = SimpleNamespace(
  has_fla=False,
  has_cuda=False, # initial implementation does not come with custom cuda kernels
  backend_order=[WKVBackend.FLA, WKVBackend.CUDA, WKVBackend.PYTORCH_OPTIMIZED],
)

_wkv6_cuda = SimpleNamespace(
  head_size=64,
  max_seq_len=4096,
  verbose=False,
  kernel=None,
)

@contextmanager
def wkv6_kernel(backends, cuda_head_size=None, cuda_max_seq_len=None, cuda_verbose=None, cuda_cache=True):
  global _wkv6_config

  if isinstance(backends, str):
    backends = [backends]

  old_config = _wkv6_config
  _wkv6_config = SimpleNamespace(**vars(_wkv6_config))

  _wkv6_config.backend_order = [*backends]

  cuda_dirty = False
  if cuda_max_seq_len is not None:
    _wkv6_cuda.max_seq_len = cuda_max_seq_len
    cuda_dirty = True
  if cuda_verbose is not None:
    _wkv6_cuda.verbose = cuda_verbose
    cuda_dirty = True

  try:
    if cuda_cache and WKVBackend.CUDA in backends:
      if _wkv6_cuda.kernel is None:
        load_wkv6_cuda()
      else:
        assert not cuda_dirty, "reloading the WKV6 CUDA kernel with different options is not yet supported"
    yield _wkv6_config
  finally:
    _wkv6_config = old_config

def load_wkv6_cuda():
  _wkv6_cuda.kernel = load(
    name=f"wkv6_{_wkv6_cuda.head_size}_{_wkv6_cuda.max_seq_len}",
    sources=[
      os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cuda', x) for x in ("wkv6_op.cpp", "wkv6_cuda.cu")
    ],
    verbose=_wkv6_cuda.verbose,
    extra_cuda_cflags=[
      "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization",
      f"-D_N_={_wkv6_cuda.head_size}", f"-D_T_={_wkv6_cuda.max_seq_len}",
    ],
  )

class WKV6CUDA(torch.autograd.Function):
  @staticmethod
  def forward(ctx, r, k, v, w, u):
    # note: B, L, H*K = B, T, C
    B, L, H, K = k.shape

    assert all([tensor.dtype == torch.bfloat16 for tensor in (r, k, v, u)]), "r, k, v, u must be dtype bfloat16"
    assert all([tensor.is_contiguous() for tensor in (r, k, v, w, u)]), "r, k, v, w, u must be contiguous"

    ctx.save_for_backward(r, k, v, w, u)

    y = torch.empty_like(v, device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    _wkv6_cuda.kernel.forward(B, L, H*K, H, r, k, v, w, u, y)
    return y

  @staticmethod
  @torch.no_grad
  def backward(ctx, gy):
    r, k, v, w, u = ctx.saved_tensors
    B, L, H, K = k.shape

    assert gy.dtype == torch.bfloat16
    assert gy.is_contiguous()

    gr, gk, gv, gw = map(lambda x: (
      torch.empty_like(x, device=x.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    ), (r, k, v, w))
    gu = torch.empty(B, H, K, device=u.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    _wkv6_cuda.kernel.backward(B, L, H*K, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
    gu = gu.sum(dim=0)
    return gr, gk, gv, gw, gu


@torch._dynamo.disable
@torch.jit.ignore
def wkv6_cuda(r, k, v, w, u):
  if _wkv6_cuda.kernel is None:
    load_wkv6_cuda()

  # Unlike wkv6_torch and FLA's kernel, the CUDA kernel expects the shapes of
  # r, k, v, w, u to be (B, L [or T], H, *).
  return WKV6CUDA.apply(r, k, v, w, u)

@torch.jit.ignore
def wkv6_torch(r, k, v, w, u, kv_state=None, chunk_len=128, dtype=torch.float64):
  (B, H, L, K), V, T = k.size(), v.size(-1), chunk_len

  if chunk_len > 24 and dtype != torch.float64:
    warnings.warn("dtype should be torch.float64 if chunk_len > 24", RuntimeWarning)

  if kv_state is None:
    kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)
  else:
    kv_state = kv_state.to(r.dtype)

  w = w.exp()

  if L == 1:
    u = rearrange(u.to(r.dtype), 'h k -> () h k ()')

    kv = k.mT @ v
    out = r @ (kv_state + u * kv)

    kv_state = w.mT * kv_state + kv
    return out, kv_state
  else:
    assert dtype in (torch.float32, torch.float64)

    if L % T != 0:
      if L % 2 != 0:
        T = 1
      else:
        while L % T != 0:
          T -= 2

    r, k, v = map(lambda x: rearrange(x, 'b h (n t) d -> b h n t d', t=T), (r, k, v))

    w = w.clamp(0.005) # precision_min_val = 0.005
    wc_log = rearrange(w.float().log(), 'b h (n t) k -> b h n t k', t=T)
    wc_log_cum = wc_log.cumsum(dim=-2)

    shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))

    ws = wc_log.sum(dim=-2, keepdim=True)
    w_inter = ws - wc_log_cum
    w_intra = wc_log_cum - wc_log

    ws = list(ws.mT.exp().to(r.dtype).unbind(dim=-3))
    w_inter = w_inter.exp().to(r.dtype)
    w_intra = w_intra.exp().to(r.dtype)

    u = rearrange(u.to(r.dtype), 'h k -> () h () () k')

    wc_log_offset = shifted_wc_log_cum[...,T//2:T//2+1,:]
    r_decay = (shifted_wc_log_cum - wc_log_offset).to(dtype).exp()
    k_inv_decay = (wc_log_offset - wc_log_cum).to(dtype).exp()
    a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1)
    a = a + einsum(r, u * k, 'b h n t k, b h n t k -> b h n t').diag_embed()
    out = a @ v

    wkv = (k * w_inter).mT @ v
    wkv = list(wkv.unbind(dim=-3))

    states = []
    for i in range(L // T):
      states.append(kv_state)
      kv_state = kv_state * ws[i] + wkv[i]
    states = torch.stack(states, dim=2)

    out = out + (r * w_intra) @ states
    out = rearrange(out, 'b h n t v -> b h (n t) v')
    return out, kv_state

@torch.no_grad
def init_orthogonal_(x, gain=1.0):
  if x.dtype == torch.bfloat16:
    return x.copy_(nn.init.orthogonal_(torch.empty_like(x, device=x.device, dtype=torch.float32), gain=gain))
  else:
    return nn.init.orthogonal_(x, gain=gain)

class TimeMix(nn.Module):
  _HEAD_SIZE_DIVISOR = 8
  _TM_EXTRA_DIM      = 32
  _TD_EXTRA_DIM      = 64

  def __init__(
    self,
    d_model,
    d_head=64,
    bias=False,
    layer_idx=0,
    n_layer=1,
    wkv_backend=None,
    wkv_chunk_len=128,
    wkv_dtype=torch.float64,
    device=None,
    dtype=None,
  ):
    cls = self.__class__
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.d_model = d_model
    self.d_head = d_head
    self.layer_idx = layer_idx
    self.n_layer = n_layer

    self.wkv_backend = wkv_backend
    self.wkv_chunk_len = wkv_chunk_len
    self.wkv_dtype = wkv_dtype

    self.n_head = d_model // d_head
    assert d_model % d_head == 0
    mixing_init_scale = 0.1
    self.time_maa_x = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs)*mixing_init_scale)
    self.time_maa_r = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs)*mixing_init_scale)
    self.time_maa_w = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs)*mixing_init_scale)
    self.time_maa_k = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs)*mixing_init_scale)
    self.time_maa_v = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs)*mixing_init_scale)
    self.time_maa_g = nn.Parameter(torch.randn(1, 1, d_model, **factory_kwargs)*mixing_init_scale)

    self.time_maa_w1 = nn.Parameter(torch.randn(self.d_model, cls._TM_EXTRA_DIM * 5, **factory_kwargs)*mixing_init_scale)
    self.time_maa_w2 = nn.Parameter(torch.randn(5, cls._TM_EXTRA_DIM, self.d_model, **factory_kwargs)*mixing_init_scale)

    self.time_decay_w1 = nn.Parameter(torch.randn(self.d_model, cls._TD_EXTRA_DIM, **factory_kwargs)*mixing_init_scale)
    self.time_decay_w2 = nn.Parameter(torch.randn(cls._TD_EXTRA_DIM, self.d_model, **factory_kwargs)*mixing_init_scale)

    self.time_decay = nn.Parameter(torch.randn(1, 1, self.d_model, **factory_kwargs)*mixing_init_scale)

    self.time_faaaa = nn.Parameter(torch.randn(self.n_head, self.d_head, **factory_kwargs)*mixing_init_scale)

    self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    self.receptance = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
    self.key = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
    self.value = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
    self.output = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
    self.gate = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    self.ln_x = nn.GroupNorm(self.n_head, self.d_model, eps=(1e-5) * (cls._HEAD_SIZE_DIVISOR ** 2), **factory_kwargs)

  def forward(self, x, state=None, need_state=True):
    tm_state, kv_state = (None, None) if state is None else state

    xx = self.time_shift(x) if tm_state is None else torch.concat((tm_state.unsqueeze(1), x[:, :-1]), dim=1)
    xx = xx - x

    xxx = x + xx * self.time_maa_x
    xxx = rearrange(torch.tanh(xxx @ self.time_maa_w1), 'b l (n x) -> n (b l) x', n=5)
    xxx = rearrange(torch.bmm(xxx, self.time_maa_w2), 'n (b l) x -> n b l x', b=x.size(0))

    mw, mk, mv, mr, mg = xxx.unbind(dim=0)
    xw = x + xx * (self.time_maa_w + mw)
    xr = x + xx * (self.time_maa_r + mr)
    xk = x + xx * (self.time_maa_k + mk)
    xv = x + xx * (self.time_maa_v + mv)
    xg = x + xx * (self.time_maa_g + mg)

    tm_state = x[:, -1]

    r = self.receptance(xr)
    k = self.key(xk)
    v = self.value(xv)
    g = self.gate(xg)

    w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
    w = -w.float().exp()

    if self.wkv_backend is not None:
      backend = self.wkv_backend
    else:
      backend = None
      for bk in _wkv6_config.backend_order:
        if bk == WKVBackend.FLA and _wkv6_config.has_fla:
          backend = bk
          break
        elif (
          bk == WKVBackend.CUDA and _wkv6_config.has_cuda and
          state is None and not need_state and
          (not x.requires_grad or x.size(1) < _wkv6_cuda.max_seq_len)
        ):
          backend = bk
          break
        elif bk == WKVBackend.PYTORCH_OPTIMIZED:
          backend = bk
          break

    if backend in (WKVBackend.FLA, WKVBackend.PYTORCH_OPTIMIZED):
      r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.n_head), (r, w, k, v))

      if backend == WKVBackend.FLA and x.size(1) == 1:
        y, new_state = fla.ops.rwkv6.fused_recurrent_rwkv6(r, k, v, w, self.time_faaaa, scale=1, initial_state=kv_state, output_final_state=need_state)
      elif backend == WKVBackend.FLA and x.size(1) > 1:
        y, new_state = fla.ops.rwkv6.chunk_rwkv6(r, k, v, w, self.time_faaaa, scale=1, initial_state=kv_state, output_final_state=need_state)
      elif backend == WKVBackend.PYTORCH_OPTIMIZED:
        y, new_state = wkv6_torch(r, k, v, w, self.time_faaaa, kv_state, self.wkv_chunk_len, self.wkv_dtype)

      if kv_state is not None:
        kv_state.copy_(new_state)
      else:
        kv_state = new_state

      y = rearrange(y, 'b h l v -> (b l) (h v)')
    elif backend == WKVBackend.CUDA:
      r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b l h d', h=self.n_head), (r, w, k, v))

      y = wkv6_cuda(r, k, v, w, self.time_faaaa)
      y = rearrange(y, 'b l h v -> (b l) (h v)')
    elif backend is None:
      raise "Could not find usable backend"
    else:
      raise f"Unknown backend: {backend}"

    y = rearrange(self.ln_x(y), '(b l) d -> b l d', b=x.size(0))
    y = self.output(y * F.silu(g))

    return (y, (tm_state, kv_state)) if need_state else y


class ChannelMix(nn.Module):
  def __init__(self, d_model, expand=3.5, bias=False, layer_idx=0, n_layer=1, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.d_model = d_model
    self.expand = expand
    self.layer_idx = layer_idx
    self.n_layer = n_layer

    d_ffn = int(d_model * expand)

    self.time_maa_k = nn.Parameter(torch.randn(1, 1, self.d_model, **factory_kwargs)*0.1)
    self.time_maa_r = nn.Parameter(torch.randn(1, 1, self.d_model, **factory_kwargs)*0.1)

    self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    self.key = nn.Linear(d_model, d_ffn, bias=bias, **factory_kwargs)
    self.receptance = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
    self.value = nn.Linear(d_ffn, d_model, bias=bias, **factory_kwargs)


  def forward(self, x, state=None, need_state=True):
    xx = self.time_shift(x) if state is None else torch.concat((state.unsqueeze(1), x[:, :-1]), dim=1)
    xx = xx - x

    xk = x + xx * self.time_maa_k
    xr = x + xx * self.time_maa_r
    kv = self.value(F.relu(self.key(xk)) ** 2)

    y = F.sigmoid(self.receptance(xr)) * kv
    return (y, x[:, -1]) if need_state else y

class Block(nn.Module):
  def __init__(
    self,
    d_model,
    d_head=64,
    expand=3.5,
    bias=False,
    layer_idx=0,
    n_layer=1,
    use_ln0=True,
    tmix_kwargs={},
    cmix_kwargs={},
    device=None,
    dtype=None,
  ):
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.layer_idx = layer_idx
    self.n_layer = n_layer

    if layer_idx == 0 and use_ln0:
      self.ln0 = nn.LayerNorm(d_model, **factory_kwargs)
    else:
      self.ln0 = None

    self.ln1 = nn.LayerNorm(d_model, **factory_kwargs)
    self.att = TimeMix(d_model, d_head, bias, layer_idx, n_layer, **factory_kwargs, **tmix_kwargs)
    self.ln2 = nn.LayerNorm(d_model, **factory_kwargs)
    self.ffn = ChannelMix(d_model, expand, bias, layer_idx, n_layer, **factory_kwargs, **cmix_kwargs)


  def forward(self, x, state=None, need_state=True):
    state = (None, None) if state is None else state

    if self.ln0 is not None:
      x = self.ln0(x)

    if not need_state:
      x = x + self.att(self.ln1(x), state=state[0], need_state=False)
      x = x + self.ffn(self.ln2(x), state=state[1], need_state=False)
      return x
    else:
      x_t, s_t = self.att(self.ln1(x), state=state[0], need_state=True)
      x = x + x_t
      x_c, s_c = self.ffn(self.ln2(x), state=state[1], need_state=True)
      x = x + x_c
      return x, (s_t, s_c)


class Model(nn.Module):
  "Simple RWKV model"
  def __init__(
    self,
    d_model=1024,
    d_head=64,
    expand=3.5,
    bias=False,
    n_layer=1,
    vocab_size=16384,
    tmix_kwargs={},
    cmix_kwargs={},
    device=None,
    dtype=None,
  ):
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.d_model = d_model
    self.n_layer = n_layer
    self.vocab_size = vocab_size

    if vocab_size is not None:
      self.emb = nn.Embedding(vocab_size, d_model, **factory_kwargs)
      with torch.no_grad():
          self.emb.weight*=0.01
      self.head = nn.Linear(d_model, vocab_size, bias=bias, **factory_kwargs)
    else:
      self.emb = None
      self.head = None

    self.blocks = nn.ModuleList([
      Block(
        d_model,
        d_head,
        expand=expand,
        bias=bias,
        layer_idx=i,
        n_layer=n_layer,
        tmix_kwargs=tmix_kwargs,
        cmix_kwargs=cmix_kwargs,
        **factory_kwargs,
      ) for i in range(n_layer)
    ])

    self.ln_out = nn.LayerNorm(d_model, **factory_kwargs)

  def forward(self, x, state=None, need_state=False, need_x_emb=True, need_x_unemb=True, grad_cp=None):
    state = [None] * self.n_layer if state is None else [*state]
    grad_cp = (lambda f, *a, **k: f(*a, **k)) if grad_cp is None or not x.requires_grad else grad_cp

    if self.emb is not None and need_x_emb:
      x = self.emb(x)

    if need_state:
      for i, block in enumerate(self.blocks):
        x, state[i] = grad_cp(block, x, state=state[i], need_state=True)
    else:
      for i, block in enumerate(self.blocks):
        x = grad_cp(block, x, state=state[i], need_state=False)

    x = self.ln_out(x)
    if self.head is not None and need_x_unemb:
      x = self.head(x)

    return (x, state) if need_state else x
    

batch_size = 8
seq_len = 1024

def get_inputs():
    return [torch.randint(0, 16384, (2, 128))]

def get_init_inputs():
    return []

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    model = Model()
    print(model(*get_inputs()).size())
