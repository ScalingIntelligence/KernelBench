"""TileLang backend for DeepSeek V4 MHC pre big-fuse.

This module contains only the TileLang part of MHC pre. Prenorm GEMM and
square-sum are handled by the public API layer, typically through
``mate.deep_gemm.tf32_hc_prenorm_gemm``.
"""

import functools
import math
import os
from typing import Optional, Tuple

import torch

__all__ = ["run_mhc_pre_big_fuse"]


def _tilelang_musa_compile_flags(profile: Optional[str] = None):
    profile = (profile or "").strip().lower()
    if profile in ("", "default", "none", "0"):
        return None
    if profile == "opt1":
        return [
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-mllvm",
            "-mtgpu-opt-level=1",
        ]
    if profile == "ls":
        return [
            "-fmusa-flush-denormals-to-zero",
            "-fno-signed-zeros",
            "-mllvm",
            "-mtgpu-opt-level=1",
            "-mllvm",
            "-mtgpu-load-store-opt=1",
            "-mllvm",
            "-mtgpu-fold-global-ldst=1",
            "-mllvm",
            "-mtgpu-load-cluster-mutation=1",
            "-mllvm",
            "-mtgpu-store-cluster-mutation=1",
            "-mllvm",
            "-mtgpu-memory-sched-mutation=1",
        ]
    raise ValueError(
        f"Unsupported MHC TileLang compile_profile={profile!r}; "
        "expected one of default,opt1,ls."
    )


def _mhc_pass_configs(
    tilelang, mode: str = "safe", compile_profile: Optional[str] = None
):
    mode = mode.lower()
    if mode == "none":
        return None

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    if mode in ("burst", "aggressive"):
        pass_configs.update(
            {
                tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
                tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
                tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
            }
        )
    if mode == "aggressive":
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS] = True
        if os.environ.get("MATE_MHC_TILELANG_DISABLE_INDEX_PROMOTION") == "1":
            pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    if mode not in ("safe", "burst", "aggressive"):
        raise ValueError(
            "MHC TileLang pass_config must be one of 'safe', 'burst', "
            f"'aggressive', or 'none', got {mode!r}."
        )

    compile_flags = _tilelang_musa_compile_flags(compile_profile)
    if compile_flags is not None:
        pass_configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = compile_flags
    return pass_configs


@functools.lru_cache(maxsize=None)
def _tilelang_mhc_pre_big_fuse_kernel(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    mhc_mult: int = 4,
    threads: int = 128,
    hidden_block: int = 512,
    pass_config: str = "safe",
    compile_profile: Optional[str] = None,
):
    import tilelang
    import tilelang.language as T

    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(hidden_block, hidden_size)
    assert threads in (128, 256)
    assert hidden_block > 0
    assert hidden_size % hidden_block == 0
    assert n_splits > 0
    assert sinkhorn_repeat > 0

    @tilelang.jit(
        pass_configs=_mhc_pass_configs(tilelang, pass_config, compile_profile)
    )
    def mhc_pre_big_fuse_kernel(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=threads) as pid:
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            T.sync_threads()

            if T.get_thread_binding() < 32:
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * mhc_scale[0] + mhc_base[j])
                        + mhc_pre_eps
                    )
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(
                            mixes_shared[j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )

                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]

            T.sync_threads()

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=1):
                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc_mult):
                    pre = pre_mix_shared[i_mhc]
                    for i1_h in T.Parallel(hidden_block):
                        h = i0_h * hidden_block + i1_h
                        ol[i1_h] += pre * T.cast(residual[pid, i_mhc, h], T.float32)

                T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse_kernel


def _validate_big_fuse_tensors(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if gemm_out_mul.dtype != torch.float32:
        raise ValueError(f"gemm_out_mul must be float32, got {gemm_out_mul.dtype}.")
    if gemm_out_sqrsum.dtype != torch.float32:
        raise ValueError(
            f"gemm_out_sqrsum must be float32, got {gemm_out_sqrsum.dtype}."
        )
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be bfloat16, got {residual.dtype}.")
    if mhc_scale.dtype != torch.float32 or mhc_base.dtype != torch.float32:
        raise ValueError("mhc_scale and mhc_base must be float32.")
    for name, tensor in (
        ("gemm_out_mul", gemm_out_mul),
        ("gemm_out_sqrsum", gemm_out_sqrsum),
        ("residual", residual),
        ("mhc_scale", mhc_scale),
        ("mhc_base", mhc_base),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if residual.dim() != 3:
        raise ValueError(
            f"residual must have shape [num_tokens, mhc_mult, hidden_size], "
            f"got {tuple(residual.shape)}."
        )
    if gemm_out_mul.dim() == 2:
        gemm_out_mul = gemm_out_mul.unsqueeze(0)
    if gemm_out_sqrsum.dim() == 1:
        gemm_out_sqrsum = gemm_out_sqrsum.unsqueeze(0)
    if gemm_out_mul.dim() != 3 or gemm_out_sqrsum.dim() != 2:
        raise ValueError("gemm outputs must be [S, M, N] and [S, M].")

    n_splits, num_tokens, mhc_mult3 = gemm_out_mul.shape
    mhc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    if residual.shape[0] != num_tokens:
        raise ValueError(
            f"residual token dim {residual.shape[0]} does not match GEMM {num_tokens}."
        )
    if gemm_out_sqrsum.shape != (n_splits, num_tokens):
        raise ValueError(
            f"gemm_out_sqrsum must have shape {(n_splits, num_tokens)}, "
            f"got {tuple(gemm_out_sqrsum.shape)}."
        )
    if mhc_mult3 != mhc_mult * (2 + mhc_mult):
        raise ValueError(
            f"gemm_out_mul last dim must be mhc_mult * (2 + mhc_mult), "
            f"got {mhc_mult3} for mhc_mult={mhc_mult}."
        )
    if mhc_scale.shape != (3,):
        raise ValueError(
            f"mhc_scale must have shape [3], got {tuple(mhc_scale.shape)}."
        )
    if mhc_base.shape != (mhc_mult3,):
        raise ValueError(
            f"mhc_base must have shape [{mhc_mult3}], got {tuple(mhc_base.shape)}."
        )
    return n_splits, num_tokens, mhc_mult, hidden_size


def run_mhc_pre_big_fuse(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    threads: int = 128,
    hidden_block: int = 512,
    pass_config: str = "safe",
    compile_profile: Optional[str] = None,
) -> None:
    if gemm_out_mul.dim() == 2:
        gemm_out_mul = gemm_out_mul.unsqueeze(0)
    if gemm_out_sqrsum.dim() == 1:
        gemm_out_sqrsum = gemm_out_sqrsum.unsqueeze(0)
    n_splits, num_tokens, mhc_mult, hidden_size = _validate_big_fuse_tensors(
        gemm_out_mul, gemm_out_sqrsum, residual, mhc_scale, mhc_base
    )
    if post_mix.shape != (num_tokens, mhc_mult):
        raise ValueError(
            f"post_mix must have shape {(num_tokens, mhc_mult)}, "
            f"got {tuple(post_mix.shape)}."
        )
    if not post_mix.is_contiguous():
        raise ValueError("post_mix must be contiguous.")
    if comb_mix.shape != (num_tokens, mhc_mult * mhc_mult):
        raise ValueError(
            f"comb_mix must have shape {(num_tokens, mhc_mult * mhc_mult)}, "
            f"got {tuple(comb_mix.shape)}."
        )
    if not comb_mix.is_contiguous():
        raise ValueError("comb_mix must be contiguous.")
    if layer_input.shape != (num_tokens, hidden_size):
        raise ValueError(
            f"layer_input must have shape {(num_tokens, hidden_size)}, "
            f"got {tuple(layer_input.shape)}."
        )
    if not layer_input.is_contiguous():
        raise ValueError("layer_input must be contiguous.")
    if sinkhorn_repeat <= 0:
        raise ValueError(f"sinkhorn_repeat must be > 0, got {sinkhorn_repeat}.")

    _tilelang_mhc_pre_big_fuse_kernel(
        hidden_size,
        float(rms_eps),
        float(mhc_pre_eps),
        float(mhc_sinkhorn_eps),
        float(mhc_post_mult_value),
        int(sinkhorn_repeat),
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        threads=int(threads),
        hidden_block=int(hidden_block),
        pass_config=pass_config,
        compile_profile=compile_profile,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )
