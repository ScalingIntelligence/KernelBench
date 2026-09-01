import torch
import tilelang
from tilelang import language as T
import pytest
import tilelang.testing
from . import sparse_mla_fwd_pipelined_v2 as prefill_v2
from . import sparse_mla_decode_fwd_pipelined_v2 as decode_v2
torch.random.manual_seed(42)
def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")

@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize("sq", [1, 2, 4])
@pytest.mark.parametrize("skv", [1024, 4096])
@pytest.mark.parametrize("heads", [128, ])
@pytest.mark.parametrize("hkv", [1, ])
@pytest.mark.parametrize("dqk", [576, ])
@pytest.mark.parametrize("dv", [512, ])
@pytest.mark.parametrize("topk", [2048, ])
@pytest.mark.parametrize("dtype", [torch.bfloat16, ])
def test_dsa_decode(batch, sq, skv, heads, hkv, dqk, dv, topk, dtype):
    device = get_test_device()
    total_q = sq * batch
    q = torch.randn((total_q, heads, dqk), dtype=dtype, device=device)
    kv = torch.randn((skv, hkv, dqk), dtype=dtype, device=device)

    indices = torch.full((total_q, hkv, topk), -1, dtype=torch.int32, device=device)
    for t in range(total_q):
        for h in range(hkv):
            i_i = torch.randperm(skv, device=device)[:topk]
            indices[t, h, :len(i_i)] = i_i
    quant_scales = torch.tensor([0.6, 0.7, 1.0, 0.9], dtype=torch.float32, device=device)
    quant_scales = quant_scales.view(1, 1, 4)
    quant_scales = quant_scales.repeat_interleave(skv, dim=0)
    quant_scales = quant_scales.repeat_interleave(hkv, dim=1)
    k_latent_fp8 = kv[..., :dv].to(torch.float8_e4m3fn).contiguous().view(skv, hkv, dv)
    k_pe = kv[..., dv:].to(torch.bfloat16).contiguous().view(skv, hkv, dqk - dv)
    k_cache_bytes = torch.cat(
        [k_latent_fp8.view(torch.uint8),
         quant_scales.view(torch.uint8),
         k_pe.view(torch.uint8)],
        dim=-1).contiguous()
    tl_out, _ = decode_v2.sparse_mla_fwd_interface(q, k_cache_bytes, indices)
    tl_out_2, _ = decode_v2.sparse_mla_fwd_interface(q, k_cache_bytes, indices)
    torch.testing.assert_close(tl_out_2, tl_out, rtol=1e-7, atol=1e-7)
    k_scales = quant_scales.repeat_interleave(128, dim=-1)
    k_latent_fp32 = k_latent_fp8.to(torch.float32) * k_scales
    k_latent_fp32[k_latent_fp32 != k_latent_fp32] = 0.0
    k_latent_bf16 = k_latent_fp32.to(torch.bfloat16)
    kv_ref = torch.cat([k_latent_bf16, k_pe], dim=-1).contiguous()
    ref_out, _ = decode_v2.ref_sparse_mla_fwd_interface(q, kv_ref, indices)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)

@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("total_q", [32, 64, 128])
@pytest.mark.parametrize("skv", [1024, 4096])
@pytest.mark.parametrize("heads", [128, ])
@pytest.mark.parametrize("hkv", [1, ])
@pytest.mark.parametrize("dqk", [576, ])
@pytest.mark.parametrize("dv", [512, ])
@pytest.mark.parametrize("topk", [2048, ])
@pytest.mark.parametrize("dtype", [torch.bfloat16, ])
def test_dsa_prefill(total_q, skv, heads, hkv, dqk, dv, topk, dtype):
    device = get_test_device()
    q = torch.randn((total_q, heads, dqk), dtype=dtype, device=device)
    kv = torch.randn((skv, hkv, dqk), dtype=dtype, device=device)

    indices = torch.full((total_q, hkv, topk), -1, dtype=torch.int32, device=device)
    for t in range(total_q):
        for h in range(hkv):
            i_i = torch.randperm(max(1, t), device=device)[:topk]
            indices[t, h, :len(i_i)] = i_i
    tl_out, _ = prefill_v2.sparse_mla_fwd_interface(q, kv, indices)
    tl_out_2, _ = prefill_v2.sparse_mla_fwd_interface(q, kv, indices)
    torch.testing.assert_close(tl_out_2, tl_out, rtol=1e-7, atol=1e-7)
    ref_out = prefill_v2.ref_sparse_mla_fwd_interface(q, kv, indices)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)