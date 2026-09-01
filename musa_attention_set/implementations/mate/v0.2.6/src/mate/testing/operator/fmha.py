"""FMHA operator definition shared by generated and direct-input workflows."""

from __future__ import annotations

import gc
import math
import os
import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Sequence

import torch
from einops import rearrange

from mate.execution_context import empty_if_dry_run, is_fake_mode
from mate.mha_interface import (
    flash_attn_combine,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from mate.testing.flash_attn import (
    _combine_cp_partials,
    apply_rotary_emb,
    attention_ref,
    generate_block_kvcache,
    generate_random_padding_mask,
    lse_ref_from_score,
    unpad_input,
)

from .operator import Operator


class FmhaApi(Enum):
    VARLEN = "varlen"
    KVCACHE = "kvcache"
    COMBINE = "combine"


class FmhaLayout(Enum):
    NORMAL = "normal"
    PADDED = "padded"
    RAGGED = "ragged"
    PAGED = "paged"


class FmhaInputMode(Enum):
    STANDARD = "standard"
    METADATA = "metadata"
    NONCONTIGUOUS = "noncontiguous"
    LARGE_STRIDE = "large-stride"
    ADVANCED = "advanced"


class FmhaVerifyMode(Enum):
    CLOSE = "close"
    ADVANCED = "advanced"


class UnsupportedFmhaWorkload(ValueError):
    """Raised when an FMHA workload describes an unsupported feature combination."""


@dataclass(frozen=True, kw_only=True)
class FmhaWorkload:
    api: FmhaApi
    batch_size: int
    seqlen_q: int
    num_heads_q: int
    head_dim_v: int

    seqlen_k: int | None = None
    num_heads_kv: int | None = None
    head_dim_qk: int | None = None
    dtype: torch.dtype = torch.bfloat16
    reference_dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "musa"
    seed: int | None = 666
    label: str | None = None

    input_mode: FmhaInputMode = FmhaInputMode.STANDARD
    verify_mode: FmhaVerifyMode = FmhaVerifyMode.CLOSE
    q_layout: FmhaLayout = FmhaLayout.NORMAL
    kv_layout: FmhaLayout = FmhaLayout.NORMAL
    q_lengths: Sequence[int] | None = None
    kv_lengths: Sequence[int] | None = None
    q_used_lengths: Sequence[int] | None = None
    kv_used_lengths: Sequence[int] | None = None
    page_size: int | None = None

    causal: bool = False
    window_size: tuple[int | None, int | None] = (None, None)
    attention_chunk: int = 0
    softcap: float = 0.0
    softmax_scale: float | None = None
    num_splits: int = 0
    pack_gqa: bool | None = None
    backend: str = "mutlass"
    return_softmax_lse: bool = True
    compare_lse: bool = True
    atol: float = 1.5e-2
    rtol: float = 1e-2

    cp_world_size: int = 1
    learnable_sink: bool = False
    expect_single_split: bool = False

    has_qv: bool = False
    only_qv: bool = False
    new_kv: bool = False
    seqlen_new_eq_seqlen_q: bool = False
    has_cache_batch_idx: bool = False
    has_cache_leftpad: bool = False
    rotary_fraction: float = 0.0
    rotary_interleaved: bool = True
    has_rotary_seqlens: bool = False

    randomize_splits: bool = False
    reuse_output: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "seqlen_q", "num_heads_q", "head_dim_v"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.backend not in ("auto", "mutlass", "mubin"):
            raise ValueError("backend must be 'auto', 'mutlass', or 'mubin'")
        if self.cp_world_size <= 0:
            raise ValueError("cp_world_size must be positive")
        if self.q_layout is FmhaLayout.PAGED:
            raise ValueError("query layout cannot be paged")
        self._validate_lengths("q_lengths", self.q_lengths, self.seqlen_q)
        self._validate_used_lengths("q_used_lengths", self.q_used_lengths)

        if self.api is FmhaApi.COMBINE:
            if self.num_splits <= 0:
                raise ValueError("combine workloads require num_splits > 0")
            if self.kv_layout is not FmhaLayout.NORMAL:
                raise ValueError("combine workloads do not use a KV layout")
            return

        if self.seqlen_k is None or self.seqlen_k <= 0:
            raise ValueError("seqlen_k must be positive for attention workloads")
        if self.num_heads_kv is None or self.num_heads_kv <= 0:
            raise ValueError("num_heads_kv must be positive for attention workloads")
        if self.head_dim_qk is None or self.head_dim_qk <= 0:
            raise ValueError("head_dim_qk must be positive for attention workloads")
        if self.num_heads_q % self.num_heads_kv != 0:
            raise ValueError("num_heads_q must be divisible by num_heads_kv")
        self._validate_lengths("kv_lengths", self.kv_lengths, self.seqlen_k)
        self._validate_used_lengths("kv_used_lengths", self.kv_used_lengths)

        if self.api is FmhaApi.VARLEN and self.kv_layout is FmhaLayout.PAGED:
            raise ValueError("paged KV uses the kvcache API")
        if self.api is FmhaApi.KVCACHE and self.kv_layout not in (
            FmhaLayout.NORMAL,
            FmhaLayout.PAGED,
        ):
            raise ValueError("kvcache KV layout must be normal or paged")
        if self.kv_layout is FmhaLayout.PAGED:
            if self.page_size is None or self.page_size <= 0:
                raise ValueError("paged KV workloads require a positive page_size")
        elif self.page_size is not None:
            raise ValueError("page_size is only valid for paged KV workloads")
        if self.only_qv and not self.has_qv:
            raise ValueError("only_qv requires has_qv")

    def _validate_lengths(
        self,
        name: str,
        lengths: Sequence[int] | None,
        maximum: int,
    ) -> None:
        if lengths is None:
            return
        if len(lengths) != self.batch_size:
            raise ValueError(f"{name} must contain one value per batch item")
        if any(length < 0 or length > maximum for length in lengths):
            raise ValueError(f"{name} values must be in [0, {maximum}]")

    def _validate_used_lengths(
        self,
        name: str,
        lengths: Sequence[int] | None,
    ) -> None:
        if lengths is not None and len(lengths) != self.batch_size:
            raise ValueError(f"{name} must contain one value per batch item")

    def __str__(self) -> str:
        if self.label is not None:
            return self.label
        parts = [
            self.api.value,
            self.input_mode.value,
            f"q-{self.q_layout.value}",
            f"kv-{self.kv_layout.value}",
            f"b{self.batch_size}",
            f"sq{self.seqlen_q}",
        ]
        if self.seqlen_k is not None:
            parts.append(f"sk{self.seqlen_k}")
        heads = f"h{self.num_heads_q}"
        if self.num_heads_kv is not None:
            heads += f"x{self.num_heads_kv}"
        dimensions = f"d{self.head_dim_v}"
        if self.head_dim_qk is not None:
            dimensions = f"d{self.head_dim_qk}x{self.head_dim_v}"
        parts.extend((heads, dimensions))
        if self.page_size is not None:
            parts.append(f"page{self.page_size}")
        if self.causal:
            parts.append("causal")
        elif self.window_size not in ((None, None), (-1, -1)):
            parts.append(f"local{self.window_size[0]}x{self.window_size[1]}")
        if self.attention_chunk > 0:
            parts.append(f"chunk{self.attention_chunk}")
        if self.softcap > 0:
            parts.append(f"softcap{self.softcap:g}")
        parts.append(f"split{self.num_splits}")
        if self.pack_gqa:
            parts.append("pack")
        if self.cp_world_size > 1:
            parts.append(f"cp{self.cp_world_size}")
        if self.api is FmhaApi.VARLEN:
            parts.append(self.backend)
        return "-".join(parts)


@dataclass(kw_only=True)
class FmhaInputs:
    api: FmhaApi
    workload: FmhaWorkload | None = None

    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    qv: torch.Tensor | None = None
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    k_cache_reference: torch.Tensor | None = None
    v_cache_reference: torch.Tensor | None = None
    out_partial: torch.Tensor | None = None
    lse_partial: torch.Tensor | None = None
    out: torch.Tensor | None = None

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    cu_seqlens_k_new: torch.Tensor | None = None
    seqused_q: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | int | None = None
    cache_batch_idx: torch.Tensor | None = None
    cache_leftpad: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    prepare_scheduler_metadata: bool = False

    rotary_cos: torch.Tensor | None = None
    rotary_sin: torch.Tensor | None = None
    rotary_cos_reference: torch.Tensor | None = None
    rotary_sin_reference: torch.Tensor | None = None
    rotary_seqlens: torch.Tensor | None = None
    q_descale: torch.Tensor | None = None
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    learnable_sink: torch.Tensor | None = None
    cp_tot_seqused_k: torch.Tensor | None = None

    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    max_seqlen_k_new: int = 0
    softmax_scale: float | None = None
    causal: bool = False
    window_size: tuple[int | None, int | None] = (None, None)
    attention_chunk: int = 0
    softcap: float = 0.0
    num_splits: int = 0
    pack_gqa: bool | None = None
    deterministic: bool = False
    sm_margin: int = 0
    backend: str = "mutlass"
    rotary_interleaved: bool = True
    cp_world_size: int = 1
    cp_rank: int = 0
    only_qv: bool = False
    return_softmax_lse: bool = True
    out_dtype: torch.dtype | None = None

    q_padded: torch.Tensor | None = None
    k_padded: torch.Tensor | None = None
    v_padded: torch.Tensor | None = None
    qv_padded: torch.Tensor | None = None
    query_padding_mask: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None
    key_new_padding_mask: torch.Tensor | None = None
    q_indices: torch.Tensor | None = None
    k_indices: torch.Tensor | None = None
    q_lengths: tuple[int, ...] | None = None
    kv_lengths: tuple[int, ...] | None = None
    rotary_dim: int = 0
    baseline_inputs: FmhaInputs | None = None


@dataclass(frozen=True, kw_only=True)
class FmhaOutputs:
    out: torch.Tensor
    lse: torch.Tensor | None = None
    out_accum: torch.Tensor | None = None
    lse_accum: torch.Tensor | None = None
    auxiliary: tuple[Any, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FmhaReference:
    out: torch.Tensor
    lse: torch.Tensor | None = None
    baseline_out: torch.Tensor | None = None
    expected_k_cache: torch.Tensor | None = None
    expected_v_cache: torch.Tensor | None = None


class FmhaOperator(Operator[FmhaWorkload, FmhaInputs, FmhaOutputs, FmhaReference]):
    """Generate, invoke, reference, and verify FMHA workloads."""

    def generate(self, workload: FmhaWorkload) -> FmhaInputs:
        unsupported = self._unsupported_reason(workload)
        if unsupported is not None:
            raise UnsupportedFmhaWorkload(unsupported)
        self._seed(workload.seed, workload.device)
        if workload.api is FmhaApi.COMBINE:
            return self._generate_combine(workload)
        if workload.input_mode is FmhaInputMode.NONCONTIGUOUS:
            return self._generate_noncontiguous(workload)
        if workload.input_mode is FmhaInputMode.LARGE_STRIDE:
            return self._generate_large_stride(workload)
        if workload.input_mode is FmhaInputMode.ADVANCED:
            return self._generate_advanced(workload)
        return self._generate_attention(workload)

    def call(self, inputs: FmhaInputs) -> FmhaOutputs:
        if inputs.api is FmhaApi.COMBINE:
            return self._call_combine(inputs)
        if inputs.cp_world_size > 1:
            return self._call_cp(inputs)
        return self._call_attention_once(inputs)

    def reference(self, inputs: FmhaInputs) -> FmhaReference:
        if inputs.baseline_inputs is not None:
            return FmhaReference(
                out=self._call_attention_once(inputs.baseline_inputs).out
            )
        if inputs.api is FmhaApi.COMBINE:
            return self._reference_combine(inputs)
        if (
            inputs.workload is not None
            and inputs.workload.input_mode is FmhaInputMode.ADVANCED
        ):
            return self._reference_advanced(inputs)
        return self._reference_attention(inputs)

    def verify(
        self,
        inputs: FmhaInputs,
        outputs: FmhaOutputs,
        reference: FmhaReference,
    ) -> None:
        workload = inputs.workload
        if workload is not None and workload.verify_mode is FmhaVerifyMode.ADVANCED:
            self._verify_advanced(inputs, outputs, reference)
            return
        self._verify_close(inputs, outputs, reference)

    def _call_attention_once(self, inputs: FmhaInputs) -> FmhaOutputs:
        metadata = inputs.scheduler_metadata
        if (
            metadata is None
            and inputs.prepare_scheduler_metadata
            and inputs.num_splits >= 0
        ):
            metadata = self._make_scheduler_metadata(inputs)
            inputs.scheduler_metadata = metadata

        if inputs.api is FmhaApi.VARLEN:
            if inputs.q is None or inputs.k is None or inputs.v is None:
                raise ValueError("varlen calls require q, k, and v")
            result = flash_attn_varlen_func(
                q=inputs.q,
                k=inputs.k,
                v=inputs.v,
                cu_seqlens_q=inputs.cu_seqlens_q,
                cu_seqlens_k=inputs.cu_seqlens_k,
                max_seqlen_q=inputs.max_seqlen_q,
                max_seqlen_k=inputs.max_seqlen_k,
                seqused_q=inputs.seqused_q,
                seqused_k=inputs.seqused_k,
                page_table=inputs.page_table,
                softmax_scale=inputs.softmax_scale,
                causal=inputs.causal,
                qv=inputs.qv,
                q_descale=inputs.q_descale,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                window_size=inputs.window_size,
                learnable_sink=inputs.learnable_sink,
                attention_chunk=inputs.attention_chunk,
                softcap=inputs.softcap,
                scheduler_metadata=metadata,
                num_splits=inputs.num_splits,
                pack_gqa=inputs.pack_gqa,
                deterministic=inputs.deterministic,
                sm_margin=inputs.sm_margin,
                return_softmax_lse=inputs.return_softmax_lse,
                backend=inputs.backend,
                cp_world_size=inputs.cp_world_size,
                cp_rank=inputs.cp_rank,
                cp_tot_seqused_k=inputs.cp_tot_seqused_k,
                out=inputs.out,
            )
        else:
            if inputs.q is None or inputs.k_cache is None or inputs.v_cache is None:
                raise ValueError("kvcache calls require q, k_cache, and v_cache")
            if inputs.only_qv and inputs.qv is None:
                raise ValueError("only_qv kvcache calls require qv")
            result = flash_attn_with_kvcache(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                k=inputs.k,
                v=inputs.v,
                qv=inputs.qv,
                rotary_cos=inputs.rotary_cos,
                rotary_sin=inputs.rotary_sin,
                cache_seqlens=inputs.cache_seqlens,
                cache_batch_idx=inputs.cache_batch_idx,
                cache_leftpad=inputs.cache_leftpad,
                page_table=inputs.page_table,
                cu_seqlens_q=inputs.cu_seqlens_q,
                cu_seqlens_k_new=inputs.cu_seqlens_k_new,
                max_seqlen_q=inputs.max_seqlen_q,
                rotary_seqlens=inputs.rotary_seqlens,
                q_descale=inputs.q_descale,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                softmax_scale=inputs.softmax_scale,
                causal=inputs.causal,
                window_size=inputs.window_size,
                learnable_sink=inputs.learnable_sink,
                attention_chunk=inputs.attention_chunk,
                softcap=inputs.softcap,
                rotary_interleaved=inputs.rotary_interleaved,
                scheduler_metadata=metadata,
                num_splits=inputs.num_splits,
                pack_gqa=inputs.pack_gqa,
                sm_margin=inputs.sm_margin,
                return_softmax_lse=inputs.return_softmax_lse,
                cp_world_size=inputs.cp_world_size,
                cp_rank=inputs.cp_rank,
                cp_tot_seqused_k=inputs.cp_tot_seqused_k,
                only_qv=inputs.only_qv,
            )
        return self._normalize_outputs(result, inputs.return_softmax_lse)

    def _call_cp(self, inputs: FmhaInputs) -> FmhaOutputs:
        if inputs.q is None or inputs.cu_seqlens_q is None:
            raise ValueError("CP workloads require ragged q and cu_seqlens_q")
        if inputs.kv_lengths is None:
            raise ValueError("CP workloads require concrete KV lengths")

        cp_tot_seqused_k = torch.tensor(
            inputs.kv_lengths, dtype=torch.int32, device=inputs.q.device
        )
        rank_inputs_keepalive: list[FmhaInputs] = []
        rank_outputs: list[FmhaOutputs] = []
        for cp_rank in range(inputs.cp_world_size):
            local_lengths = tuple(
                max(
                    0,
                    (length - cp_rank + inputs.cp_world_size - 1)
                    // inputs.cp_world_size,
                )
                for length in inputs.kv_lengths
            )
            if inputs.api is FmhaApi.VARLEN:
                if inputs.k_padded is None or inputs.v_padded is None:
                    raise ValueError("varlen CP requires padded source K and V")
                k_segments = [
                    inputs.k_padded[index, cp_rank : length : inputs.cp_world_size]
                    for index, length in enumerate(inputs.kv_lengths)
                ]
                v_segments = [
                    inputs.v_padded[index, cp_rank : length : inputs.cp_world_size]
                    for index, length in enumerate(inputs.kv_lengths)
                ]
                cu_seqlens_k = torch.tensor(
                    self._cumulative_lengths(local_lengths),
                    dtype=torch.int32,
                    device=inputs.q.device,
                )
                rank_inputs = replace(
                    inputs,
                    k=torch.cat(k_segments, dim=0),
                    v=torch.cat(v_segments, dim=0),
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_k=None,
                    max_seqlen_k=max(local_lengths),
                    scheduler_metadata=None,
                    cp_rank=cp_rank,
                    cp_tot_seqused_k=cp_tot_seqused_k,
                )
            else:
                if (
                    inputs.k_cache is None
                    or inputs.v_cache is None
                    or inputs.page_table is None
                    or inputs.workload is None
                    or inputs.workload.page_size is None
                ):
                    raise ValueError("paged CP requires paged K/V and a page table")
                k_local, v_local, page_table_local, local_seqlens = (
                    self._make_cp_paged_cache(
                        inputs,
                        cp_rank=cp_rank,
                        local_lengths=local_lengths,
                    )
                )
                rank_inputs = replace(
                    inputs,
                    k_cache=k_local,
                    v_cache=v_local,
                    page_table=page_table_local,
                    cache_seqlens=local_seqlens,
                    max_seqlen_k=max(local_lengths),
                    scheduler_metadata=None,
                    cp_rank=cp_rank,
                    cp_tot_seqused_k=cp_tot_seqused_k,
                )
            rank_inputs_keepalive.append(rank_inputs)
            rank_outputs.append(self._call_attention_once(rank_inputs))
            if not is_fake_mode() and hasattr(torch, "musa"):
                torch.musa.synchronize()

        if is_fake_mode():
            return rank_outputs[0]
        total_q = inputs.q.shape[0]
        combined_out = torch.zeros(
            total_q,
            inputs.q.shape[-2],
            rank_outputs[0].out.shape[-1],
            dtype=torch.float32,
            device=inputs.q.device,
        )
        combined_lse = torch.empty(
            inputs.q.shape[-2], total_q, dtype=torch.float32, device=inputs.q.device
        )
        cu_q = [int(value) for value in inputs.cu_seqlens_q.cpu().tolist()]
        for batch_index, (start, end) in enumerate(zip(cu_q, cu_q[1:])):
            if start == end:
                continue
            out_item, lse_item = _combine_cp_partials(
                [output.out[start:end].float() for output in rank_outputs],
                [output.lse[:, start:end].float() for output in rank_outputs],
            )
            combined_out[start:end] = out_item
            combined_lse[:, start:end] = lse_item
        return FmhaOutputs(out=combined_out.to(inputs.q.dtype), lse=combined_lse)

    def _make_cp_paged_cache(
        self,
        inputs: FmhaInputs,
        *,
        cp_rank: int,
        local_lengths: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            inputs.k_cache is None
            or inputs.v_cache is None
            or inputs.page_table is None
            or inputs.workload is None
            or inputs.workload.page_size is None
        ):
            raise ValueError("paged CP requires paged K/V and a page table")
        page_size = inputs.workload.page_size
        batch_size = len(local_lengths)
        max_local_length = max(local_lengths)
        pages_per_sequence = max(1, math.ceil(max_local_length / page_size))
        padded_length = pages_per_sequence * page_size
        k_dense = self._logical_cache(inputs.k_cache, inputs.page_table)
        v_dense = self._logical_cache(inputs.v_cache, inputs.page_table)
        k_local_dense = torch.zeros(
            batch_size,
            padded_length,
            *k_dense.shape[2:],
            dtype=k_dense.dtype,
            device=k_dense.device,
        )
        v_local_dense = torch.zeros(
            batch_size,
            padded_length,
            *v_dense.shape[2:],
            dtype=v_dense.dtype,
            device=v_dense.device,
        )
        if not is_fake_mode():
            assert inputs.kv_lengths is not None
            for batch_index, global_length in enumerate(inputs.kv_lengths):
                local_length = local_lengths[batch_index]
                k_local_dense[batch_index, :local_length] = k_dense[
                    batch_index, cp_rank : global_length : inputs.cp_world_size
                ]
                v_local_dense[batch_index, :local_length] = v_dense[
                    batch_index, cp_rank : global_length : inputs.cp_world_size
                ]
        k_local = k_local_dense.reshape(
            batch_size * pages_per_sequence,
            page_size,
            *k_dense.shape[2:],
        )
        v_local = v_local_dense.reshape(
            batch_size * pages_per_sequence,
            page_size,
            *v_dense.shape[2:],
        )
        page_table = torch.arange(
            batch_size * pages_per_sequence,
            dtype=torch.int32,
            device=k_dense.device,
        ).reshape(batch_size, pages_per_sequence)
        seqlens = torch.tensor(local_lengths, dtype=torch.int32, device=k_dense.device)
        return k_local, v_local, page_table, seqlens

    def _verify_close(
        self,
        inputs: FmhaInputs,
        outputs: FmhaOutputs,
        reference: FmhaReference,
    ) -> None:
        workload = inputs.workload
        atol = workload.atol if workload is not None else 1.5e-2
        rtol = workload.rtol if workload is not None else 1e-2
        out = outputs.out
        out_ref = reference.out.to(dtype=out.dtype)
        valid_queries: torch.Tensor | None = None
        if out.dim() == 4 and inputs.seqused_q is not None:
            valid_queries = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
            valid_queries = valid_queries < inputs.seqused_q.to(out.device).unsqueeze(1)
            out = out[valid_queries]
            out_ref = out_ref[valid_queries]
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

        compare_lse = workload.compare_lse if workload is not None else True
        if compare_lse and reference.lse is not None:
            if outputs.lse is None:
                raise AssertionError("FMHA did not return the requested softmax LSE")
            lse = outputs.lse
            lse_ref = reference.lse.to(dtype=lse.dtype)
            if valid_queries is not None and lse.dim() == 3:
                if inputs.api is FmhaApi.COMBINE:
                    valid_lse = valid_queries.unsqueeze(-1).expand_as(lse)
                else:
                    valid_lse = valid_queries.unsqueeze(1).expand_as(lse)
                lse = lse[valid_lse]
                lse_ref = lse_ref[valid_lse]
            torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)

        if (
            workload is not None
            and workload.reuse_output
            and outputs.out is not inputs.out
        ):
            raise AssertionError("FMHA did not reuse the supplied output tensor")
        if workload is not None and workload.expect_single_split:
            if inputs.scheduler_metadata is None:
                raise AssertionError("FMHA did not generate scheduler metadata")
            splits = inputs.scheduler_metadata[: workload.batch_size]
            torch.testing.assert_close(splits, torch.ones_like(splits), atol=0, rtol=0)

    def _verify_advanced(
        self,
        inputs: FmhaInputs,
        outputs: FmhaOutputs,
        reference: FmhaReference,
    ) -> None:
        if reference.baseline_out is None or inputs.workload is None:
            raise ValueError("advanced verification requires a baseline reference")
        out = outputs.out.float()
        out_ref = reference.out.float()
        baseline = reference.baseline_out.float()
        max_multiplier = 4 if inputs.workload.dtype == torch.float8_e4m3fn else 2
        mean_multiplier = 3 if inputs.workload.dtype == torch.float8_e4m3fn else 1.5
        diff = (out - out_ref).abs()
        baseline_diff = (baseline - out_ref).abs()
        if diff.max().item() > max_multiplier * baseline_diff.max().item() + 1e-5:
            raise AssertionError("FMHA maximum error exceeded the advanced threshold")
        if diff.mean().item() > mean_multiplier * baseline_diff.mean().item() + 1e-5:
            raise AssertionError("FMHA mean error exceeded the advanced threshold")

        if not inputs.workload.new_kv:
            return
        if reference.expected_k_cache is None or reference.expected_v_cache is None:
            raise ValueError("append-KV verification requires expected caches")
        k_actual = self._logical_cache(
            inputs.k_cache.to(inputs.workload.reference_dtype), inputs.page_table
        )
        v_actual = self._logical_cache(
            inputs.v_cache.to(inputs.workload.reference_dtype), inputs.page_table
        )
        if inputs.cache_batch_idx is not None:
            indices = inputs.cache_batch_idx.long()
            k_actual = k_actual[indices]
            v_actual = v_actual[indices]
        k_actual = k_actual[:, : inputs.workload.seqlen_k].to(
            inputs.workload.reference_dtype
        )
        v_actual = v_actual[:, : inputs.workload.seqlen_k].to(
            inputs.workload.reference_dtype
        )
        if inputs.workload.dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(
                v_actual, reference.expected_v_cache, atol=1e-3, rtol=1e-3
            )
        else:
            torch.testing.assert_close(
                v_actual, reference.expected_v_cache, atol=0, rtol=0
            )
        if inputs.rotary_dim == 0:
            atol = rtol = 0.0
        elif inputs.workload.dtype == torch.float8_e4m3fn:
            atol = rtol = 1e-1
        else:
            atol, rtol = 1e-2, 1.5e-2
        torch.testing.assert_close(
            k_actual, reference.expected_k_cache, atol=atol, rtol=rtol
        )

    @staticmethod
    def _seed(seed: int | None, device: str | torch.device) -> None:
        if seed is None:
            return
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.device(device).type == "musa" and hasattr(torch, "musa"):
            torch.musa.manual_seed(seed)

    @staticmethod
    def _unsupported_reason(workload: FmhaWorkload) -> str | None:
        local = workload.window_size not in ((None, None), (-1, -1))
        if workload.learnable_sink and not local:
            return "learnable sink is only supported by these workloads with local attention"
        if workload.input_mode is not FmhaInputMode.ADVANCED:
            return None
        dimensions = (workload.head_dim_qk, workload.head_dim_v)
        if not workload.has_qv and dimensions not in ((192, 128), (128, 128)):
            return "non-QV advanced attention only supports d192x128 and d128x128"
        if workload.seqlen_k is not None and workload.seqlen_q > workload.seqlen_k:
            return "append KV requires seqlen_q <= seqlen_k"
        if not workload.new_kv and (
            workload.rotary_fraction > 0
            or workload.seqlen_new_eq_seqlen_q
            or workload.has_rotary_seqlens
        ):
            return "rotary append-KV options require new_kv"
        if workload.rotary_fraction == 0 and (
            workload.has_rotary_seqlens or not workload.rotary_interleaved
        ):
            return "rotary layout options require rotary embeddings"
        if workload.dtype == torch.float8_e4m3fn and workload.has_qv:
            if dimensions not in ((64, 256), (64, 512)):
                return "float8 QV only supports d64x256 and d64x512"
            if workload.new_kv or workload.rotary_fraction > 0:
                return "float8 QV does not support append KV or rotary"
        return None

    def _generate_attention(self, workload: FmhaWorkload) -> FmhaInputs:
        assert workload.seqlen_k is not None
        assert workload.num_heads_kv is not None
        assert workload.head_dim_qk is not None
        q_lengths = self._resolve_lengths(
            workload.q_lengths, workload.batch_size, workload.seqlen_q
        )
        kv_lengths = self._resolve_lengths(
            workload.kv_lengths, workload.batch_size, workload.seqlen_k
        )
        q_used = self._resolve_used_lengths(workload.q_used_lengths, q_lengths)
        kv_used = self._resolve_used_lengths(workload.kv_used_lengths, kv_lengths)
        q, q_padded, q_mask, q_indices, cu_q, seqused_q = self._generate_layout_tensor(
            workload.q_layout,
            q_lengths,
            q_used,
            workload.seqlen_q,
            workload.num_heads_q,
            workload.head_dim_qk,
            workload.dtype,
            workload.device,
            use_seqused=workload.q_used_lengths is not None,
        )
        learnable_sink = (
            torch.randn(
                workload.num_heads_q, device=workload.device, dtype=workload.dtype
            )
            * 10
            if workload.learnable_sink
            else None
        )
        common = dict(
            api=workload.api,
            workload=workload,
            q=q,
            q_padded=q_padded,
            query_padding_mask=q_mask,
            q_indices=q_indices,
            q_lengths=q_used,
            cu_seqlens_q=cu_q,
            seqused_q=seqused_q,
            max_seqlen_q=(
                workload.seqlen_q if workload.q_layout is FmhaLayout.RAGGED else None
            ),
            max_seqlen_k=workload.seqlen_k,
            softmax_scale=(
                workload.softmax_scale
                if workload.softmax_scale is not None
                else workload.head_dim_qk**-0.5
            ),
            causal=workload.causal,
            window_size=workload.window_size,
            learnable_sink=learnable_sink,
            attention_chunk=workload.attention_chunk,
            softcap=workload.softcap,
            num_splits=workload.num_splits,
            pack_gqa=workload.pack_gqa,
            backend=workload.backend,
            cp_world_size=workload.cp_world_size,
            return_softmax_lse=workload.return_softmax_lse,
            prepare_scheduler_metadata=True,
            kv_lengths=kv_used,
        )
        if workload.api is FmhaApi.VARLEN:
            k, k_padded, k_mask, k_indices, cu_k, seqused_k = (
                self._generate_layout_tensor(
                    workload.kv_layout,
                    kv_lengths,
                    kv_used,
                    workload.seqlen_k,
                    workload.num_heads_kv,
                    workload.head_dim_qk,
                    workload.dtype,
                    workload.device,
                    use_seqused=workload.kv_used_lengths is not None,
                )
            )
            v, v_padded, _, _, _, _ = self._generate_layout_tensor(
                workload.kv_layout,
                kv_lengths,
                kv_used,
                workload.seqlen_k,
                workload.num_heads_kv,
                workload.head_dim_v,
                workload.dtype,
                workload.device,
                mask=k_mask,
            )
            return FmhaInputs(
                **common,
                k=k,
                v=v,
                k_padded=k_padded,
                v_padded=v_padded,
                key_padding_mask=k_mask,
                k_indices=k_indices,
                cu_seqlens_k=cu_k,
                seqused_k=seqused_k,
            )

        cache_seqlens = torch.tensor(kv_used, dtype=torch.int32, device=workload.device)
        if workload.kv_layout is FmhaLayout.PAGED:
            assert workload.page_size is not None
            k_dense, v_dense, page_table, k_cache, v_cache, _ = generate_block_kvcache(
                workload.seqlen_k,
                workload.page_size,
                workload.batch_size,
                workload.num_heads_kv,
                workload.head_dim_qk,
                workload.head_dim_v,
                workload.device,
                workload.dtype,
            )
        else:
            k_dense = torch.randn(
                workload.batch_size,
                workload.seqlen_k,
                workload.num_heads_kv,
                workload.head_dim_qk,
                dtype=workload.dtype,
                device=workload.device,
            )
            v_dense = torch.randn(
                workload.batch_size,
                workload.seqlen_k,
                workload.num_heads_kv,
                workload.head_dim_v,
                dtype=workload.dtype,
                device=workload.device,
            )
            k_cache, v_cache, page_table = k_dense, v_dense, None
        key_mask = self._length_mask(kv_used, workload.seqlen_k, workload.device)
        return FmhaInputs(
            **common,
            k_cache=k_cache,
            v_cache=v_cache,
            k_padded=k_dense,
            v_padded=v_dense,
            key_padding_mask=key_mask,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
        )

    def _generate_noncontiguous(self, workload: FmhaWorkload) -> FmhaInputs:
        assert workload.head_dim_qk is not None
        assert workload.num_heads_kv is not None
        total = sum(
            self._resolve_lengths(
                workload.q_lengths, workload.batch_size, workload.seqlen_q
            )
        )
        qkv = torch.randn(
            total,
            3,
            workload.num_heads_q,
            workload.head_dim_qk,
            device=workload.device,
            dtype=workload.dtype,
        )
        q, k, v = qkv.unbind(dim=1)
        lengths = self._resolve_lengths(
            workload.q_lengths, workload.batch_size, workload.seqlen_q
        )
        cu = torch.tensor(
            self._cumulative_lengths(lengths),
            dtype=torch.int32,
            device=workload.device,
        )
        inputs = FmhaInputs(
            api=FmhaApi.VARLEN,
            workload=workload,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=workload.seqlen_q,
            max_seqlen_k=workload.seqlen_q,
            backend=workload.backend,
            return_softmax_lse=False,
        )
        inputs.baseline_inputs = replace(inputs, k=k.contiguous(), baseline_inputs=None)
        return inputs

    def _generate_large_stride(self, workload: FmhaWorkload) -> FmhaInputs:
        assert workload.head_dim_qk is not None
        assert workload.num_heads_kv is not None

        def make_view(
            num_heads: int, head_dim: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            base = torch.randn(
                workload.seqlen_q,
                num_heads,
                head_dim,
                device=workload.device,
                dtype=workload.dtype,
            )
            view = base.as_strided(
                (1, workload.seqlen_q, num_heads, head_dim),
                (2**31 + 1024, num_heads * head_dim, head_dim, 1),
            )
            return view, base.unsqueeze(0).contiguous()

        q, q_regular = make_view(workload.num_heads_q, workload.head_dim_qk)
        k, k_regular = make_view(workload.num_heads_kv, workload.head_dim_qk)
        v, v_regular = make_view(workload.num_heads_kv, workload.head_dim_v)
        inputs = FmhaInputs(
            api=FmhaApi.VARLEN,
            workload=workload,
            q=q,
            k=k,
            v=v,
            backend=workload.backend,
            return_softmax_lse=False,
        )
        inputs.baseline_inputs = replace(
            inputs,
            q=q_regular,
            k=k_regular,
            v=v_regular,
            baseline_inputs=None,
        )
        return inputs

    def _generate_advanced(self, workload: FmhaWorkload) -> FmhaInputs:
        assert workload.seqlen_k is not None
        assert workload.num_heads_kv is not None
        assert workload.head_dim_qk is not None
        if hasattr(torch, "musa"):
            gc.collect()
            torch.musa.empty_cache()
        input_dtype = workload.dtype
        ref_dtype = workload.reference_dtype
        q_ref = (
            torch.randn(
                workload.batch_size,
                workload.seqlen_q,
                workload.num_heads_q,
                workload.head_dim_qk,
                device=workload.device,
                dtype=ref_dtype,
            )
            .to(input_dtype)
            .to(ref_dtype)
        )
        qv_ref = (
            torch.randn(
                workload.batch_size,
                workload.seqlen_q,
                workload.num_heads_q,
                workload.head_dim_v,
                device=workload.device,
                dtype=ref_dtype,
            )
            .to(input_dtype)
            .to(ref_dtype)
            if workload.has_qv
            else None
        )
        if workload.q_layout is FmhaLayout.RAGGED:
            q_mask = generate_random_padding_mask(
                workload.seqlen_q, workload.batch_size, workload.device, mode="random"
            )
            q_unpadded, q_indices, cu_q, max_q, *_ = unpad_input(q_ref, q_mask)
            q_actual = q_unpadded.to(input_dtype)
            qv_actual = (
                qv_ref.reshape(-1, *qv_ref.shape[2:])[q_indices].to(input_dtype)
                if qv_ref is not None
                else None
            )
        else:
            q_mask, q_indices, cu_q = None, None, None
            max_q = workload.seqlen_q
            q_actual = q_ref.to(input_dtype)
            qv_actual = qv_ref.to(input_dtype) if qv_ref is not None else None

        seqlen_new = (
            workload.seqlen_q
            if workload.seqlen_new_eq_seqlen_q
            else random.randint(1, workload.seqlen_q)
        )
        if workload.new_kv:
            k_new_ref = (
                torch.randn(
                    workload.batch_size,
                    seqlen_new,
                    workload.num_heads_kv,
                    workload.head_dim_qk,
                    device=workload.device,
                    dtype=ref_dtype,
                )
                .to(input_dtype)
                .to(ref_dtype)
            )
            v_new_ref = (
                torch.randn(
                    workload.batch_size,
                    seqlen_new,
                    workload.num_heads_kv,
                    workload.head_dim_v,
                    device=workload.device,
                    dtype=ref_dtype,
                )
                .to(input_dtype)
                .to(ref_dtype)
            )
            if workload.q_layout is FmhaLayout.RAGGED:
                key_new_mask = generate_random_padding_mask(
                    seqlen_new, workload.batch_size, workload.device, mode="random"
                )
                k_actual, k_indices, cu_k_new, *_ = unpad_input(k_new_ref, key_new_mask)
                v_actual, *_ = unpad_input(v_new_ref, key_new_mask)
            else:
                key_new_mask, k_indices, cu_k_new = None, None, None
                k_actual, v_actual = k_new_ref, v_new_ref
            k_actual = k_actual.to(input_dtype)
            v_actual = v_actual.to(input_dtype)
        else:
            k_new_ref = v_new_ref = None
            k_actual = v_actual = None
            key_new_mask = k_indices = cu_k_new = None

        batch_size_cache = (
            workload.batch_size * 2
            if workload.has_cache_batch_idx
            else workload.batch_size
        )
        if workload.kv_layout is FmhaLayout.PAGED:
            assert workload.page_size is not None
            _, _, page_table, k_cache, v_cache, num_pages = generate_block_kvcache(
                workload.seqlen_k,
                workload.page_size,
                batch_size_cache,
                workload.num_heads_kv,
                workload.head_dim_qk,
                workload.head_dim_v,
                workload.device,
                input_dtype,
                ref_dtype,
                torch.randn,
            )
        else:
            k_cache = (
                torch.randn(
                    batch_size_cache,
                    workload.seqlen_k,
                    workload.num_heads_kv,
                    workload.head_dim_qk,
                    device=workload.device,
                    dtype=ref_dtype,
                )
                .to(input_dtype)
                .to(ref_dtype)
            )
            v_cache = (
                torch.randn(
                    batch_size_cache,
                    workload.seqlen_k,
                    workload.num_heads_kv,
                    workload.head_dim_v,
                    device=workload.device,
                    dtype=ref_dtype,
                )
                .to(input_dtype)
                .to(ref_dtype)
            )
            page_table, num_pages = None, 0

        rotary_dim = (
            math.floor(workload.rotary_fraction * workload.head_dim_qk / 16) * 16
        )
        effective_local = (
            workload.window_size not in ((None, None), (-1, -1))
            or workload.attention_chunk > 0
        )
        cache_upper = (
            workload.seqlen_k
            - (
                workload.seqlen_q
                if (workload.causal or effective_local) and rotary_dim > 1
                else seqlen_new
            )
            + 1
        )
        cache_seqlens = torch.randint(
            0 if workload.new_kv else 1,
            max(cache_upper if workload.new_kv else workload.seqlen_k + 1, 1),
            (workload.batch_size,),
            dtype=torch.int32,
            device=workload.device,
        )
        if workload.has_cache_leftpad:
            if is_fake_mode():
                cache_leftpad = torch.empty(
                    workload.batch_size, dtype=torch.int32, device=workload.device
                )
            else:
                cache_leftpad = torch.cat(
                    [
                        torch.randint(
                            0,
                            int(cache_seqlens[index].item()),
                            (1,),
                            dtype=torch.int32,
                            device=workload.device,
                        )
                        if cache_seqlens[index].item() > 0
                        else torch.zeros(1, dtype=torch.int32, device=workload.device)
                        for index in range(workload.batch_size)
                    ]
                )
        else:
            cache_leftpad = None
        cache_batch_idx = (
            torch.randperm(batch_size_cache, dtype=torch.int32, device=workload.device)[
                : workload.batch_size
            ]
            if workload.has_cache_batch_idx
            else None
        )
        arange = torch.arange(workload.seqlen_k, device=workload.device).reshape(1, -1)
        cache_expanded = cache_seqlens.reshape(-1, 1)
        if workload.new_kv:
            new_lengths: int | torch.Tensor = (
                key_new_mask.sum(-1, keepdim=True)
                if key_new_mask is not None
                else seqlen_new
            )
            key_mask = arange < cache_expanded + new_lengths
        else:
            key_mask = arange < cache_expanded
        if cache_leftpad is not None:
            key_mask = key_mask & (arange >= cache_leftpad.unsqueeze(-1))

        rotary_seqlens = (
            cache_seqlens // 2 if workload.has_rotary_seqlens else cache_seqlens
        )
        if rotary_dim > 0:
            angle = (
                torch.rand(
                    workload.seqlen_k
                    if workload.page_size is None
                    else num_pages * workload.page_size,
                    rotary_dim // 2,
                    device=workload.device,
                )
                * 2
                * math.pi
            )
            cos_ref = torch.cos(angle).to(ref_dtype).to(input_dtype).to(ref_dtype)
            sin_ref = torch.sin(angle).to(ref_dtype).to(input_dtype).to(ref_dtype)
            cos_actual, sin_actual = cos_ref.to(input_dtype), sin_ref.to(input_dtype)
        else:
            cos_ref = sin_ref = cos_actual = sin_actual = None

        if input_dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [
                torch.rand(
                    workload.batch_size,
                    workload.num_heads_kv,
                    device=workload.device,
                    dtype=torch.float32,
                )
                * 2
                for _ in range(3)
            ]
        else:
            q_descale = k_descale = v_descale = None
        return FmhaInputs(
            api=FmhaApi.KVCACHE,
            workload=workload,
            q=q_actual,
            q_padded=q_ref,
            qv=qv_actual,
            qv_padded=qv_ref,
            k=k_actual,
            v=v_actual,
            k_padded=k_new_ref,
            v_padded=v_new_ref,
            k_cache=k_cache.to(input_dtype),
            v_cache=v_cache.to(input_dtype),
            k_cache_reference=k_cache,
            v_cache_reference=v_cache,
            query_padding_mask=q_mask,
            key_padding_mask=key_mask,
            key_new_padding_mask=key_new_mask,
            q_indices=q_indices,
            k_indices=k_indices,
            cu_seqlens_q=cu_q,
            cu_seqlens_k_new=cu_k_new,
            max_seqlen_q=max_q,
            max_seqlen_k=workload.seqlen_k,
            max_seqlen_k_new=seqlen_new if workload.new_kv else 0,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            page_table=page_table,
            rotary_cos=cos_actual,
            rotary_sin=sin_actual,
            rotary_cos_reference=cos_ref,
            rotary_sin_reference=sin_ref,
            rotary_seqlens=(
                rotary_seqlens
                if (not workload.has_qv or workload.has_rotary_seqlens)
                else None
            ),
            rotary_dim=rotary_dim,
            rotary_interleaved=workload.rotary_interleaved,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            causal=workload.causal,
            window_size=workload.window_size,
            attention_chunk=workload.attention_chunk,
            softcap=workload.softcap,
            num_splits=workload.num_splits,
            pack_gqa=workload.pack_gqa,
            only_qv=workload.only_qv,
            return_softmax_lse=True,
            prepare_scheduler_metadata=True,
        )

    def _generate_combine(self, workload: FmhaWorkload) -> FmhaInputs:
        if workload.q_layout is FmhaLayout.RAGGED:
            q_lengths = tuple(
                random.randint(1, workload.seqlen_q) for _ in range(workload.batch_size)
            )
            total_q = sum(q_lengths)
            shape_out = (
                workload.num_splits,
                1,
                total_q,
                workload.num_heads_q,
                workload.head_dim_v,
            )
            cu_q = torch.tensor(
                self._cumulative_lengths(q_lengths),
                dtype=torch.int32,
                device=workload.device,
            )
            seqused_q = None
        else:
            q_lengths = (workload.seqlen_q,) * workload.batch_size
            shape_out = (
                workload.num_splits,
                workload.batch_size,
                workload.seqlen_q,
                workload.num_heads_q,
                workload.head_dim_v,
            )
            cu_q = None
            seqused_q = (
                torch.tensor(
                    [
                        random.randint(1, workload.seqlen_q)
                        for _ in range(workload.batch_size)
                    ],
                    dtype=torch.int32,
                    device=workload.device,
                )
                if workload.q_layout is FmhaLayout.PADDED
                else None
            )
        out_partial = torch.randn(
            shape_out, dtype=torch.float32, device=workload.device
        )
        lse_storage_shape = shape_out[:2] + (shape_out[3], shape_out[2])
        lse_partial = torch.randn(
            lse_storage_shape, dtype=torch.float32, device=workload.device
        ).transpose(2, 3)
        valid_splits = (
            tuple(
                random.randint(2, workload.num_splits)
                for _ in range(workload.batch_size)
            )
            if workload.randomize_splits
            else (workload.num_splits,) * workload.batch_size
        )
        if not is_fake_mode():
            for batch_index, split_count in enumerate(valid_splits):
                if workload.q_layout is FmhaLayout.RAGGED:
                    assert cu_q is not None
                    start = int(cu_q[batch_index].item())
                    end = int(cu_q[batch_index + 1].item())
                    out_partial[split_count:, 0, start:end] = 0
                    lse_partial[split_count:, 0, start:end] = float("-inf")
                else:
                    out_partial[split_count:, batch_index] = 0
                    lse_partial[split_count:, batch_index] = float("-inf")
            if seqused_q is not None:
                for batch_index, length in enumerate(seqused_q.cpu().tolist()):
                    out_partial[:, batch_index, length:] = 0
                    lse_partial[:, batch_index, length:] = float("-inf")
        out = (
            torch.empty(
                shape_out[1:],
                dtype=workload.dtype,
                device=workload.device,
            )
            if workload.reuse_output
            else None
        )
        return FmhaInputs(
            api=FmhaApi.COMBINE,
            workload=workload,
            out_partial=out_partial.contiguous(),
            lse_partial=lse_partial,
            out=out,
            out_dtype=workload.dtype,
            num_splits=workload.num_splits,
            cu_seqlens_q=cu_q,
            seqused_q=seqused_q,
            q_lengths=q_lengths,
        )

    @staticmethod
    def _generate_layout_tensor(
        layout: FmhaLayout,
        lengths: tuple[int, ...],
        used_lengths: tuple[int, ...],
        maximum: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
        *,
        mask: torch.Tensor | None = None,
        use_seqused: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        random_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
        padded = torch.randn(
            len(lengths),
            maximum,
            num_heads,
            head_dim,
            dtype=random_dtype,
            device=device,
        ).to(dtype)
        if layout is FmhaLayout.NORMAL:
            return padded, padded, None, None, None, None
        if mask is None:
            mask = FmhaOperator._length_mask(used_lengths, maximum, device)
        seqused = torch.tensor(used_lengths, dtype=torch.int32, device=device)
        if layout is FmhaLayout.PADDED:
            return padded, padded, mask, None, None, seqused
        if layout is not FmhaLayout.RAGGED:
            raise ValueError(f"unsupported tensor layout: {layout.value}")
        unpadded, indices, cu_seqlens, _, *_ = unpad_input(padded, mask)
        return (
            unpadded,
            padded,
            mask,
            indices,
            cu_seqlens,
            seqused if use_seqused else None,
        )

    @staticmethod
    def _resolve_lengths(
        lengths: Sequence[int] | None,
        batch_size: int,
        maximum: int,
    ) -> tuple[int, ...]:
        return (
            (maximum,) * batch_size
            if lengths is None
            else tuple(int(length) for length in lengths)
        )

    @staticmethod
    def _resolve_used_lengths(
        used_lengths: Sequence[int] | None,
        lengths: tuple[int, ...],
    ) -> tuple[int, ...]:
        if used_lengths is None:
            return lengths
        return tuple(
            length if used < 0 else min(length, int(used))
            for length, used in zip(lengths, used_lengths)
        )

    @staticmethod
    def _length_mask(
        lengths: Sequence[int], maximum: int, device: str | torch.device
    ) -> torch.Tensor:
        return torch.arange(maximum, device=device).reshape(1, -1) < torch.tensor(
            lengths, dtype=torch.int32, device=device
        ).reshape(-1, 1)

    @staticmethod
    def _cumulative_lengths(lengths: Sequence[int]) -> list[int]:
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + int(length))
        return cumulative

    @staticmethod
    def _call_combine(inputs: FmhaInputs) -> FmhaOutputs:
        if inputs.out_partial is None or inputs.lse_partial is None:
            raise ValueError("combine calls require out_partial and lse_partial")
        out, lse = flash_attn_combine(
            inputs.out_partial,
            inputs.lse_partial,
            out=inputs.out,
            out_dtype=inputs.out_dtype,
        )
        return FmhaOutputs(out=out, lse=lse)

    @staticmethod
    def _normalize_outputs(result: Any, return_lse: bool) -> FmhaOutputs:
        if not return_lse:
            if not isinstance(result, torch.Tensor):
                raise TypeError("FMHA returned an unexpected output value")
            return FmhaOutputs(out=result)
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError("FMHA did not return the requested output and LSE")
        out, lse, *auxiliary = result
        return FmhaOutputs(
            out=out,
            lse=lse,
            out_accum=auxiliary[0] if len(auxiliary) > 0 else None,
            lse_accum=auxiliary[1] if len(auxiliary) > 1 else None,
            auxiliary=tuple(auxiliary),
        )

    @staticmethod
    def _make_scheduler_metadata(inputs: FmhaInputs) -> torch.Tensor:
        if inputs.q is None:
            raise ValueError("scheduler metadata requires q")
        if inputs.q.dim() == 4:
            batch_size, max_seqlen_q = inputs.q.shape[:2]
        elif inputs.cu_seqlens_q is not None and inputs.max_seqlen_q is not None:
            batch_size = inputs.cu_seqlens_q.shape[0] - 1
            max_seqlen_q = inputs.max_seqlen_q
        else:
            raise ValueError("ragged q requires cu_seqlens_q and max_seqlen_q")
        kv = inputs.k if inputs.api is FmhaApi.VARLEN else inputs.k_cache
        vv = inputs.v if inputs.api is FmhaApi.VARLEN else inputs.v_cache
        if kv is None or vv is None:
            raise ValueError("scheduler metadata requires K and V")
        max_seqlen_k = inputs.max_seqlen_k or kv.shape[1]
        page_size = kv.shape[1] if inputs.page_table is not None else None
        batch_rounded = (batch_size + 3) // 4 * 4
        empty_metadata = torch.empty(
            4 * batch_rounded, dtype=torch.int32, device=inputs.q.device
        )
        return empty_if_dry_run(get_scheduler_metadata, empty_values=empty_metadata)(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=inputs.q.shape[-2],
            num_heads_kv=kv.shape[-2],
            headdim=inputs.q.shape[-1],
            seqused_q=inputs.seqused_q,
            seqused_k=(
                inputs.seqused_k
                if inputs.api is FmhaApi.VARLEN
                else inputs.cache_seqlens
            ),
            qkv_dtype=inputs.q.dtype,
            headdim_v=vv.shape[-1],
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            cu_seqlens_k_new=inputs.cu_seqlens_k_new,
            cache_leftpad=inputs.cache_leftpad,
            page_size=page_size,
            max_seqlen_k_new=inputs.max_seqlen_k_new,
            causal=inputs.causal,
            window_size=inputs.window_size,
            attention_chunk=inputs.attention_chunk,
            has_softcap=inputs.softcap != 0.0,
            num_splits=inputs.num_splits,
            pack_gqa=inputs.pack_gqa,
            has_qv=inputs.qv is not None,
            mp_margin=0,
        )

    def _reference_attention(self, inputs: FmhaInputs) -> FmhaReference:
        if (
            inputs.q_padded is None
            or inputs.k_padded is None
            or inputs.v_padded is None
        ):
            raise ValueError("attention reference requires padded Q, K, and V")
        out_ref, _, score_ref = attention_ref(
            inputs.q_padded,
            inputs.k_padded,
            inputs.v_padded,
            query_padding_mask=inputs.query_padding_mask,
            key_padding_mask=inputs.key_padding_mask,
            causal=inputs.causal,
            window_size=inputs.window_size,
            attention_chunk=inputs.attention_chunk,
            learnable_sink=inputs.learnable_sink,
            softcap=inputs.softcap,
        )
        lse_unpadded, lse_padded = lse_ref_from_score(
            score_ref,
            is_causal=inputs.causal,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            seqused_q=inputs.seqused_q,
            seqused_k=(
                inputs.seqused_k
                if inputs.api is FmhaApi.VARLEN
                else inputs.cache_seqlens
            ),
            learnable_sink=inputs.learnable_sink,
        )
        if inputs.q is not None and inputs.q.dim() == 3:
            if inputs.q_indices is None:
                raise ValueError("ragged reference requires q_indices")
            out_ref = out_ref.reshape(-1, *out_ref.shape[2:])[inputs.q_indices]
            lse_ref = lse_unpadded
        else:
            lse_ref = lse_padded
        self._register_dump_reference(out_ref, lse_ref)
        return FmhaReference(out=out_ref, lse=lse_ref)

    def _reference_advanced(self, inputs: FmhaInputs) -> FmhaReference:
        if inputs.workload is None or inputs.q_padded is None:
            raise ValueError("advanced reference requires its workload and padded q")
        workload = inputs.workload
        k_cache_source = (
            inputs.k_cache_reference
            if inputs.k_cache_reference is not None
            else inputs.k_cache
        )
        v_cache_source = (
            inputs.v_cache_reference
            if inputs.v_cache_reference is not None
            else inputs.v_cache
        )
        k_cache = self._logical_cache(k_cache_source, inputs.page_table).to(
            workload.reference_dtype
        )[:, : workload.seqlen_k]
        v_cache = self._logical_cache(v_cache_source, inputs.page_table).to(
            workload.reference_dtype
        )[:, : workload.seqlen_k]
        if inputs.cache_batch_idx is not None:
            indices = inputs.cache_batch_idx.long()
            k_cache = k_cache[indices].clone()
            v_cache = v_cache[indices].clone()
        else:
            k_cache = k_cache[: workload.batch_size].clone()
            v_cache = v_cache[: workload.batch_size].clone()

        q_ref = inputs.q_padded
        k_new = inputs.k_padded
        v_new = inputs.v_padded
        if inputs.rotary_dim > 0:
            if (
                inputs.rotary_cos_reference is None
                or inputs.rotary_sin_reference is None
                or inputs.rotary_seqlens is None
            ):
                raise ValueError("rotary reference requires cos, sin, and seqlens")
            effective_local = (
                inputs.window_size not in ((None, None), (-1, -1))
                or inputs.attention_chunk > 0
            )
            if inputs.causal or effective_local:
                q_ref = apply_rotary_emb(
                    q_ref,
                    inputs.rotary_cos_reference,
                    inputs.rotary_sin_reference,
                    seqlen_offsets=inputs.rotary_seqlens,
                    interleaved=inputs.rotary_interleaved,
                )
            else:
                q_ref = rearrange(
                    apply_rotary_emb(
                        rearrange(q_ref, "b s h d -> b 1 (s h) d"),
                        inputs.rotary_cos_reference,
                        inputs.rotary_sin_reference,
                        seqlen_offsets=inputs.rotary_seqlens,
                        interleaved=inputs.rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=workload.seqlen_q,
                )
            if k_new is not None:
                k_new = apply_rotary_emb(
                    k_new,
                    inputs.rotary_cos_reference,
                    inputs.rotary_sin_reference,
                    seqlen_offsets=inputs.rotary_seqlens,
                    interleaved=inputs.rotary_interleaved,
                )

        if workload.new_kv:
            if (
                k_new is None
                or v_new is None
                or not isinstance(inputs.cache_seqlens, torch.Tensor)
            ):
                raise ValueError(
                    "append-KV reference requires new K/V and cache lengths"
                )
            arange = torch.arange(workload.seqlen_k, device=q_ref.device).reshape(1, -1)
            cache_lengths = inputs.cache_seqlens.reshape(-1, 1)
            new_lengths: int | torch.Tensor = (
                inputs.key_new_padding_mask.sum(-1, keepdim=True)
                if inputs.key_new_padding_mask is not None
                else inputs.max_seqlen_k_new
            )
            update_mask = (cache_lengths <= arange) & (
                arange < cache_lengths + new_lengths
            )
            k_update = k_new.reshape(-1, *k_new.shape[2:])
            v_update = v_new.reshape(-1, *v_new.shape[2:])
            if inputs.k_indices is not None:
                k_update = k_update[inputs.k_indices]
                v_update = v_update[inputs.k_indices]
            k_cache[update_mask] = k_update
            v_cache[update_mask] = v_update

        out_ref, _, _ = attention_ref(
            q_ref,
            k_cache,
            v_cache,
            query_padding_mask=inputs.query_padding_mask,
            key_padding_mask=inputs.key_padding_mask,
            key_leftpad=inputs.cache_leftpad,
            causal=inputs.causal,
            qv=inputs.qv_padded,
            q_descale=inputs.q_descale,
            k_descale=inputs.k_descale,
            v_descale=inputs.v_descale,
            window_size=inputs.window_size,
            attention_chunk=inputs.attention_chunk,
            softcap=inputs.softcap,
            only_qv=inputs.only_qv,
        )
        baseline_out, _, _ = attention_ref(
            q_ref,
            k_cache,
            v_cache,
            query_padding_mask=inputs.query_padding_mask,
            key_padding_mask=inputs.key_padding_mask,
            key_leftpad=inputs.cache_leftpad,
            causal=inputs.causal,
            qv=inputs.qv_padded,
            q_descale=inputs.q_descale,
            k_descale=inputs.k_descale,
            v_descale=inputs.v_descale,
            window_size=inputs.window_size,
            attention_chunk=inputs.attention_chunk,
            softcap=inputs.softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=(
                workload.dtype if workload.dtype == torch.float8_e4m3fn else None
            ),
            only_qv=inputs.only_qv,
        )
        if inputs.q_indices is not None:
            out_ref = out_ref.reshape(-1, *out_ref.shape[2:])[inputs.q_indices]
            baseline_out = baseline_out.reshape(-1, *baseline_out.shape[2:])[
                inputs.q_indices
            ]
        return FmhaReference(
            out=out_ref,
            baseline_out=baseline_out,
            expected_k_cache=k_cache.to(workload.dtype).to(workload.reference_dtype),
            expected_v_cache=v_cache.to(workload.dtype).to(workload.reference_dtype),
        )

    @staticmethod
    def _logical_cache(
        cache: torch.Tensor | None, page_table: torch.Tensor | None
    ) -> torch.Tensor:
        if cache is None:
            raise ValueError("cache tensor is required")
        if page_table is None:
            return cache
        batch_size, blocks = page_table.shape
        return cache[page_table.long().reshape(-1)].reshape(
            batch_size, blocks * cache.shape[1], *cache.shape[2:]
        )

    @staticmethod
    def _reference_combine(inputs: FmhaInputs) -> FmhaReference:
        if inputs.out_partial is None or inputs.lse_partial is None:
            raise ValueError("combine reference requires out_partial and lse_partial")
        lse_partial = inputs.lse_partial.clone(memory_format=torch.contiguous_format)
        lse = torch.logsumexp(lse_partial, dim=0)
        weights = torch.exp(lse_partial - lse.unsqueeze(0))
        weights = torch.where(torch.isfinite(weights), weights, 0.0)
        out = (weights.unsqueeze(-1) * inputs.out_partial).sum(dim=0)
        return FmhaReference(out=out, lse=lse)

    @staticmethod
    def _register_dump_reference(out: torch.Tensor, lse: torch.Tensor) -> None:
        if os.environ.get("MATE_JIT_DUMP_REFERENCE", "0") != "1" or not os.environ.get(
            "MATE_JIT_DUMP_DIR"
        ):
            return
        from mate.jit.dump import register_dump_reference

        register_dump_reference("out", out, atol=1.5e-2, rtol=1e-2)
        register_dump_reference("lse", lse, atol=1.5e-2, rtol=1e-2)


__all__ = [
    "FmhaApi",
    "FmhaInputMode",
    "FmhaInputs",
    "FmhaLayout",
    "FmhaOperator",
    "FmhaOutputs",
    "FmhaReference",
    "FmhaVerifyMode",
    "FmhaWorkload",
    "UnsupportedFmhaWorkload",
]
