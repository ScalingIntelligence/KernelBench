"""Run fused FP8 MLA RoPE quantization followed by sparse decode on MUSA."""

from __future__ import annotations

import math

import torch

import flashinfer


def _make_cos_sin_cache(tokens: int) -> torch.Tensor:
    positions = torch.arange(tokens, device="musa", dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, 64, 2, device="musa", dtype=torch.float32)
    inv_freq = torch.pow(10000.0, -dimensions / 64.0)
    angles = positions * inv_freq.unsqueeze(0)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def main() -> None:
    torch.manual_seed(20260722)
    tokens, heads, topk = 64, 64, 64
    q_quant_scale = 3.0
    kv_quant_scale = 1.5
    attention_scale = 1.0 / math.sqrt(576)

    q_rope = torch.randn(tokens, heads, 64, device="musa", dtype=torch.bfloat16) * 0.1
    q_nope = torch.randn(tokens, heads, 512, device="musa", dtype=torch.bfloat16) * 0.1
    k_rope = torch.randn(tokens, 64, device="musa", dtype=torch.bfloat16) * 0.1
    k_nope = torch.randn(tokens, 512, device="musa", dtype=torch.bfloat16) * 0.1
    cos_sin_cache = _make_cos_sin_cache(tokens)
    pos_ids = torch.arange(tokens, device="musa", dtype=torch.int32)

    q_fp8 = torch.empty(tokens, heads, 576, device="musa", dtype=torch.float8_e4m3fn)
    kv_fp8 = torch.empty(tokens, 576, device="musa", dtype=torch.float8_e4m3fn)

    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=True,
        quantize_dtype=torch.float8_e4m3fn,
        quant_scale_q=q_quant_scale,
        quant_scale_kv=kv_quant_scale,
        q_rope_out=q_fp8[..., 512:],
        k_rope_out=kv_fp8[..., 512:],
        q_nope_out=q_fp8[..., :512],
        k_nope_out=kv_fp8[..., :512],
    )

    query = q_fp8[:1].view(1, 1, heads, 576)
    kv_cache = kv_fp8.view(1, 1, tokens, 576)
    block_tables = torch.arange(topk, device="musa", dtype=torch.int32).view(1, 1, topk)
    seq_lens = torch.tensor([topk], device="musa", dtype=torch.int32)
    # workspace_buffer is retained only for FlashInfer call compatibility.
    # Prepare explicit metadata once while seq_lens and top-k stay unchanged.
    metadata = flashinfer.decode.get_batch_decode_metadata_mla(
        query=query,
        seq_lens=seq_lens,
        sparse_mla_top_k=topk,
    )

    q_descale = 1.0 / q_quant_scale
    kv_descale = 1.0 / kv_quant_scale
    out, lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=None,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=tokens,
        sparse_mla_top_k=topk,
        bmm1_scale=attention_scale * q_descale * kv_descale,
        bmm2_scale=kv_descale,
        backend="trtllm-gen",
        skip_softmax_threshold_scale_factor=None,
        return_lse=True,
        metadata=metadata,
    )
    torch.musa.synchronize()

    assert out.shape == (1, 1, heads, 512)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, heads)
    print(f"query: {tuple(query.shape)} {query.dtype}")
    print(f"kv_cache: {tuple(kv_cache.shape)} {kv_cache.dtype}")
    print(f"out: {tuple(out.shape)} {out.dtype}; lse: {tuple(lse.shape)}")


if __name__ == "__main__":
    main()
