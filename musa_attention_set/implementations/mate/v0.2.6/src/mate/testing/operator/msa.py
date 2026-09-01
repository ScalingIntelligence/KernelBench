"""MSA operator definition shared by generated and direct-input workflows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch

from mate import msa_interface as msa
from mate.testing.operator import Operator

_FP8_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)
_DTYPE_NAMES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
if _FP8_E4M3_DTYPE is not None:
    _DTYPE_NAMES[_FP8_E4M3_DTYPE] = "fp8e4m3"

_SUPPORTED_DTYPES = tuple(_DTYPE_NAMES)
_SPARSE_PATTERNS = {"first", "tail", "rolling", "rolling_holes", "sink_local"}
_PAGE_MODES = {"identity", "reverse"}


class MsaApi(Enum):
    MAXSCORE = "maxscore"
    SPARSE_TOPK = "sparse-topk"
    SPARSE_FWD = "sparse-fwd"


class MsaKernelMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class MsaKvLayout(Enum):
    DENSE = "dense"
    PAGED = "paged"


class UnsupportedMsaWorkload(ValueError):
    """Raised when an MSA workload describes an unsupported feature combination."""


@dataclass(frozen=True, kw_only=True)
class MsaWorkload:
    api: MsaApi
    num_q_heads: int
    num_kv_heads: int

    q_lengths: Sequence[int] | None = None
    kv_lengths: Sequence[int] | None = None

    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    causal: bool = True
    device: str | torch.device = "musa"
    seed: int = 666
    label: str | None = None

    kv_layout: MsaKvLayout = MsaKvLayout.PAGED
    page_size: int = 128
    sparse_block_size: int = 128
    topk: int = 16
    pattern: str = "first"
    page_mode: str = "identity"
    kernel_mode: MsaKernelMode = MsaKernelMode.PREFILL

    softmax_scale: float | None = None
    k_scale: float = 1.0
    v_scale: float = 1.0

    # MAXSCORE options.
    preallocated_max_score: bool = False

    # SPARSE_TOPK options.
    total_q: int | None = None
    max_k_tiles: int | None = None
    num_valid_pages: int | None = None
    score_pattern: str = "random"
    spike_indices: tuple[int, ...] = ()
    spike_values: tuple[float, ...] = ()
    dip_indices: tuple[int, ...] = ()
    dip_value: float = -100.0
    invalid_after: int | None = None
    force_begin_blocks: int = 0
    force_end_blocks: int = 0
    force_blocks_count_in_topk: bool = True
    query_positions: Sequence[int] | None = None
    preallocated_output: bool = False

    def __post_init__(self) -> None:
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.page_size <= 0 or self.sparse_block_size <= 0:
            raise ValueError("page_size and sparse_block_size must be positive")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"unsupported MSA dtype: {self.dtype}")
        if self.pattern not in _SPARSE_PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(_SPARSE_PATTERNS)}")
        if self.page_mode not in _PAGE_MODES:
            raise ValueError(f"page_mode must be one of {sorted(_PAGE_MODES)}")
        if self.score_pattern not in {"random", "zeros"}:
            raise ValueError("score_pattern must be 'random' or 'zeros'")
        if len(self.spike_indices) != len(self.spike_values):
            raise ValueError("spike_indices and spike_values must have equal length")

        if self.api is MsaApi.SPARSE_TOPK:
            if self.total_q is None or self.total_q <= 0:
                raise ValueError("sparse-topk workloads require a positive total_q")
            if self.max_k_tiles is None or self.max_k_tiles <= 0:
                raise ValueError("sparse-topk workloads require a positive max_k_tiles")
            if self.num_valid_pages is not None and not (
                0 < self.num_valid_pages <= self.max_k_tiles
            ):
                raise ValueError(
                    "num_valid_pages must be in (0, max_k_tiles] when provided"
                )
            return

        if not self.q_lengths or not self.kv_lengths:
            raise ValueError("MSA execution workloads require q_lengths and kv_lengths")
        if len(self.q_lengths) != len(self.kv_lengths):
            raise ValueError("q_lengths and kv_lengths must describe the same batch")
        if any(length <= 0 for length in self.q_lengths) or any(
            length <= 0 for length in self.kv_lengths
        ):
            raise ValueError("sequence lengths must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        if self.api is MsaApi.SPARSE_FWD:
            if self.num_q_heads // self.num_kv_heads not in (8, 16):
                raise ValueError("MSA forward requires a local Hq/Hkv ratio of 8 or 16")
            if self.topk != 16:
                raise ValueError("MSA forward currently requires topk == 16")
            if self.kv_layout is not MsaKvLayout.PAGED:
                raise ValueError("MSA forward requires paged KV")
            if self.page_size != 128 or self.sparse_block_size != 128:
                raise ValueError("MSA forward currently requires page/block size 128")

        if self.api is MsaApi.MAXSCORE and self.kv_layout is MsaKvLayout.PAGED:
            if self.page_size != 128:
                raise ValueError("paged MSA maxscore requires page_size == 128")

    def __str__(self) -> str:
        if self.label is not None:
            return self.label
        parts = [
            self.api.value,
            _DTYPE_NAMES[self.dtype],
            f"h{self.num_q_heads}x{self.num_kv_heads}",
        ]
        if self.api is MsaApi.SPARSE_TOPK:
            parts.append(f"t{self.topk}")
            parts.append(f"k{self.max_k_tiles}")
            if self.num_valid_pages is not None:
                parts.append(f"v{self.num_valid_pages}")
            if self.force_begin_blocks or self.force_end_blocks:
                parts.append(f"force{self.force_begin_blocks}x{self.force_end_blocks}")
            if not self.force_blocks_count_in_topk:
                parts.append("outside-topk")
            if self.score_pattern == "zeros":
                parts.append("zeros")
            if self.query_positions:
                parts.append("qpos")
        else:
            parts.append(f"b{len(self.q_lengths)}")
            if self.causal:
                parts.append("causal")
            else:
                parts.append("noncausal")
            if self.api is MsaApi.MAXSCORE:
                parts.append(self.kv_layout.value)
                if self.preallocated_max_score:
                    parts.append("prealloc")
            else:
                parts.append(self.kernel_mode.value)
                parts.append(self.pattern)
                parts.append(self.page_mode)
        return "-".join(parts)


@dataclass(kw_only=True)
class MsaInputs:
    workload: MsaWorkload | None = None

    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    q_lens: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    kv_indices: torch.Tensor | None = None
    kv_block_indexes: torch.Tensor | None = None
    plan_info: tuple | None = None

    max_score: torch.Tensor | None = None
    out: torch.Tensor | None = None
    query_positions: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class MsaOutputs:
    out: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    max_score: torch.Tensor | None = None
    selected_blocks: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class MsaReference:
    out: torch.Tensor | None = None
    out_fp32: torch.Tensor | None = None
    max_score: torch.Tensor | None = None
    selected_blocks: torch.Tensor | None = None


class MsaOperator(Operator[MsaWorkload, MsaInputs, MsaOutputs, MsaReference]):
    """Generate, invoke, reference, and verify MSA workloads."""

    def generate(self, workload: MsaWorkload) -> MsaInputs:
        self._seed(workload.seed, workload.device)
        if workload.api is MsaApi.MAXSCORE:
            return self._generate_maxscore(workload)
        if workload.api is MsaApi.SPARSE_TOPK:
            return self._generate_sparse_topk(workload)
        return self._generate_sparse_fwd(workload)

    def call(self, inputs: MsaInputs) -> MsaOutputs:
        workload = inputs.workload
        if workload is None:
            raise ValueError("operator inputs must carry their workload")
        if workload.api is MsaApi.MAXSCORE:
            return self._call_maxscore(inputs)
        if workload.api is MsaApi.SPARSE_TOPK:
            return self._call_sparse_topk(inputs)
        return self._call_sparse_fwd(inputs)

    def reference(self, inputs: MsaInputs) -> MsaReference:
        workload = inputs.workload
        if workload is None:
            raise ValueError("operator inputs must carry their workload")
        if workload.api is MsaApi.MAXSCORE:
            return self._reference_maxscore(inputs)
        if workload.api is MsaApi.SPARSE_TOPK:
            return self._reference_sparse_topk(inputs)
        return self._reference_sparse_fwd(inputs)

    def verify(
        self,
        inputs: MsaInputs,
        outputs: MsaOutputs,
        reference: MsaReference,
    ) -> None:
        workload = inputs.workload
        if workload is None:
            raise ValueError("operator inputs must carry their workload")
        if workload.api is MsaApi.MAXSCORE:
            if outputs.max_score is None or reference.max_score is None:
                raise ValueError("maxscore verification requires max_score outputs")
            if outputs.out is not None:
                raise AssertionError(
                    "maxscore workloads must not produce attention output"
                )
            if workload.preallocated_max_score:
                if outputs.max_score is not inputs.max_score:
                    raise AssertionError(
                        "MSA maxscore did not reuse the supplied output tensor"
                    )
            self._verify_maxscore(outputs.max_score, reference.max_score)
            return
        if workload.api is MsaApi.SPARSE_TOPK:
            if outputs.selected_blocks is None or reference.selected_blocks is None:
                raise ValueError(
                    "sparse-topk verification requires selected block outputs"
                )
            if (
                workload.preallocated_output
                and outputs.selected_blocks is not inputs.out
            ):
                raise AssertionError(
                    "MSA sparse topk did not reuse the supplied output tensor"
                )
            if not torch.equal(outputs.selected_blocks, reference.selected_blocks):
                raise AssertionError(
                    "MSA sparse topk output does not match the reference"
                )
            return
        if outputs.out is None or reference.out is None or reference.out_fp32 is None:
            raise ValueError("sparse-fwd verification requires forward outputs")
        self._verify_sparse_fwd(outputs.out, reference.out_fp32, reference.out)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_maxscore(self, workload: MsaWorkload) -> MsaInputs:
        assert workload.q_lengths is not None
        assert workload.kv_lengths is not None
        device = workload.device
        q_lens = torch.tensor(workload.q_lengths, dtype=torch.int32, device=device)
        kv_lens = torch.tensor(workload.kv_lengths, dtype=torch.int32, device=device)
        total_q = sum(workload.q_lengths)
        total_k = sum(workload.kv_lengths)
        q = self._small_randn(
            (total_q, workload.num_q_heads, workload.head_dim),
            device=device,
            dtype=workload.dtype,
            scale=0.2,
        )
        page_table = None
        kv_indices = None
        if workload.kv_layout is MsaKvLayout.PAGED:
            page_table, kv_indices = self._make_page_table_and_kv_indices(
                kv_lens, page_size=workload.page_size, mode=workload.page_mode
            )
            k = self._small_randn(
                (
                    int(kv_indices.numel()),
                    workload.num_kv_heads,
                    workload.page_size,
                    workload.head_dim,
                ),
                device=device,
                dtype=workload.dtype,
                scale=0.2,
            )
        else:
            k = self._small_randn(
                (total_k, workload.num_kv_heads, workload.head_dim),
                device=device,
                dtype=workload.dtype,
                scale=0.2,
            )
        v = torch.empty_like(k)
        max_k_tiles = self._maxscore_k_tiles(max(workload.kv_lengths))
        max_score = (
            torch.empty(
                (total_q, workload.num_q_heads, max_k_tiles),
                dtype=torch.float32,
                device=device,
            )
            if workload.preallocated_max_score
            else None
        )
        plan_info = msa._msa_plan_from_lengths(
            q_lens.cpu(),
            kv_lens.cpu(),
            workload.num_q_heads,
            num_kv_heads=workload.num_kv_heads,
            page_size=workload.page_size
            if workload.kv_layout is MsaKvLayout.PAGED
            else -1,
            causal=workload.causal,
            output_maxscore=True,
            split_prefill_decode=False,
        )
        return MsaInputs(
            workload=workload,
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            kv_lens=kv_lens,
            page_table=page_table,
            kv_indices=kv_indices,
            max_score=max_score,
            plan_info=plan_info,
        )

    def _generate_sparse_topk(self, workload: MsaWorkload) -> MsaInputs:
        assert workload.total_q is not None
        assert workload.max_k_tiles is not None
        shape = (workload.total_q, workload.num_q_heads, workload.max_k_tiles)
        if workload.score_pattern == "zeros":
            max_score = torch.zeros(shape, dtype=torch.float32, device=workload.device)
        else:
            max_score = torch.randn(shape, dtype=torch.float32, device=workload.device)
            for index, value in zip(workload.spike_indices, workload.spike_values):
                max_score[:, :, index] += value
            for index in workload.dip_indices:
                max_score[:, :, index] = workload.dip_value
            if workload.invalid_after is not None:
                max_score[:, :, workload.invalid_after :] = -torch.inf
        output_width = (
            workload.topk
            if workload.force_blocks_count_in_topk
            else workload.topk + workload.force_begin_blocks + workload.force_end_blocks
        )
        out = (
            torch.empty(
                (workload.total_q, workload.num_q_heads, output_width),
                dtype=torch.int32,
                device=workload.device,
            )
            if workload.preallocated_output
            else None
        )
        query_positions = (
            torch.tensor(
                list(workload.query_positions),
                dtype=torch.int64,
                device=workload.device,
            )
            if workload.query_positions is not None
            else None
        )
        return MsaInputs(
            workload=workload,
            max_score=max_score,
            out=out,
            query_positions=query_positions,
        )

    def _generate_sparse_fwd(self, workload: MsaWorkload) -> MsaInputs:
        assert workload.q_lengths is not None
        assert workload.kv_lengths is not None
        device = workload.device
        q_lens = torch.tensor(workload.q_lengths, dtype=torch.int32, device=device)
        kv_lens = torch.tensor(workload.kv_lengths, dtype=torch.int32, device=device)
        page_table, kv_indices = self._make_page_table_and_kv_indices(
            kv_lens, page_size=workload.page_size, mode=workload.page_mode
        )
        total_q = sum(workload.q_lengths)
        total_pages = int(kv_indices.numel())
        q = self._small_randn(
            (total_q, workload.num_q_heads, workload.head_dim),
            device=device,
            dtype=workload.dtype,
        )
        k = self._small_randn(
            (total_pages, workload.num_kv_heads, workload.page_size, workload.head_dim),
            device=device,
            dtype=workload.dtype,
        )
        v = self._small_randn(
            (total_pages, workload.num_kv_heads, workload.page_size, workload.head_dim),
            device=device,
            dtype=workload.dtype,
        )
        kv_block_indexes = self._make_kv_block_indexes(
            q_lens,
            kv_lens,
            num_kv_heads=workload.num_kv_heads,
            topk=workload.topk,
            sparse_block_size=workload.sparse_block_size,
            pattern=workload.pattern,
            device=device,
        )
        plan_info = msa._msa_plan_from_lengths(
            q_lens.cpu(),
            kv_lens.cpu(),
            workload.num_q_heads,
            num_kv_heads=workload.num_kv_heads,
            page_size=workload.page_size,
            sparse_block_size=workload.sparse_block_size,
            kv_block_num=workload.topk,
            sparse_kernel_mode=workload.kernel_mode.value,
            split_prefill_decode=False,
            causal=workload.causal,
        )
        return MsaInputs(
            workload=workload,
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            kv_lens=kv_lens,
            page_table=page_table,
            kv_indices=kv_indices,
            kv_block_indexes=kv_block_indexes,
            plan_info=plan_info,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @staticmethod
    def _call_maxscore(inputs: MsaInputs) -> MsaOutputs:
        if (
            inputs.q is None
            or inputs.k is None
            or inputs.v is None
            or inputs.plan_info is None
        ):
            raise ValueError("maxscore calls require q, k, v, and plan_info")
        out, max_score = msa.msa(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.plan_info,
            kv_indices=inputs.kv_indices,
            output_o=False,
            max_score=inputs.max_score,
        )
        return MsaOutputs(out=out, max_score=max_score)

    @staticmethod
    def _call_sparse_topk(inputs: MsaInputs) -> MsaOutputs:
        workload = inputs.workload
        assert workload is not None
        if inputs.max_score is None:
            raise ValueError("sparse-topk calls require max_score")
        selected = msa.sparse_topk_select(
            inputs.max_score,
            topk=workload.topk,
            num_valid_pages=workload.num_valid_pages,
            output=inputs.out,
            force_begin_blocks=workload.force_begin_blocks,
            force_end_blocks=workload.force_end_blocks,
            force_blocks_count_in_topk=workload.force_blocks_count_in_topk,
            query_positions=inputs.query_positions,
        )
        return MsaOutputs(selected_blocks=selected)

    @staticmethod
    def _call_sparse_fwd(inputs: MsaInputs) -> MsaOutputs:
        workload = inputs.workload
        assert workload is not None
        if (
            inputs.q is None
            or inputs.k is None
            or inputs.v is None
            or inputs.plan_info is None
        ):
            raise ValueError("sparse-fwd calls require q, k, v, and plan_info")
        if workload.kernel_mode is MsaKernelMode.PREFILL:
            out, lse = msa.sparse_msa(
                inputs.q,
                inputs.k,
                inputs.v,
                inputs.plan_info,
                kv_indices=inputs.kv_indices,
                kv_block_indexes=inputs.kv_block_indexes,
                sm_scale=workload.softmax_scale,
                k_scale=workload.k_scale,
                v_scale=workload.v_scale,
            )
            return MsaOutputs(out=out, lse=lse)
        out = msa.sparse_decode_atten_func(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.plan_info,
            kv_indices=inputs.kv_indices,
            kv_block_indexes=inputs.kv_block_indexes,
            sm_scale=workload.softmax_scale,
            k_scale=workload.k_scale,
            v_scale=workload.v_scale,
        )
        return MsaOutputs(out=out)

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_maxscore(inputs: MsaInputs) -> MsaReference:
        workload = inputs.workload
        assert workload is not None
        assert workload.q_lengths is not None
        if (
            inputs.q is None
            or inputs.k is None
            or inputs.q_lens is None
            or inputs.kv_lens is None
        ):
            raise ValueError("maxscore references require q, k, and lengths")
        max_k_tiles = MsaOperator._maxscore_k_tiles(max(workload.kv_lengths))
        max_score = MsaOperator._dense_msa_output_maxscore_reference(
            inputs.q,
            inputs.k,
            inputs.q_lens,
            inputs.kv_lens,
            workload.num_kv_heads,
            max_k_tiles,
            causal=workload.causal,
            page_table=inputs.page_table,
            page_size=workload.page_size,
        )
        return MsaReference(max_score=max_score)

    @staticmethod
    def _reference_sparse_topk(inputs: MsaInputs) -> MsaReference:
        workload = inputs.workload
        assert workload is not None
        if inputs.max_score is None:
            raise ValueError("sparse-topk references require max_score")
        selected = MsaOperator._sparse_topk_select_reference(
            inputs.max_score,
            topk=workload.topk,
            num_valid_pages=workload.num_valid_pages,
            force_begin_blocks=workload.force_begin_blocks,
            force_end_blocks=workload.force_end_blocks,
            force_blocks_count_in_topk=workload.force_blocks_count_in_topk,
            query_positions=inputs.query_positions,
        )
        return MsaReference(selected_blocks=selected)

    @staticmethod
    def _reference_sparse_fwd(inputs: MsaInputs) -> MsaReference:
        workload = inputs.workload
        assert workload is not None
        if (
            inputs.q is None
            or inputs.k is None
            or inputs.v is None
            or inputs.q_lens is None
            or inputs.kv_lens is None
            or inputs.page_table is None
            or inputs.kv_block_indexes is None
        ):
            raise ValueError("sparse-fwd references require full sparse inputs")
        softmax_scale = (
            workload.softmax_scale
            if workload.softmax_scale is not None
            else workload.head_dim**-0.5
        )
        common = dict(
            causal=workload.causal,
            softmax_scale=softmax_scale,
            q_lens=inputs.q_lens,
            kv_lens=inputs.kv_lens,
            sparse_block_size=workload.sparse_block_size,
        )
        out = MsaOperator._dense_sparse_prefill_reference(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.page_table,
            inputs.kv_block_indexes,
            **common,
        )
        out_fp32 = MsaOperator._dense_sparse_prefill_reference(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.page_table,
            inputs.kv_block_indexes,
            output_dtype=torch.float32,
            **common,
        )
        return MsaReference(out=out, out_fp32=out_fp32)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_maxscore(actual: torch.Tensor, expected: torch.Tensor) -> None:
        finite = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[finite],
            expected[finite],
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))

    @staticmethod
    def _verify_sparse_fwd(
        actual: torch.Tensor,
        ref_fp32: torch.Tensor,
        ref_low_precision: torch.Tensor,
    ) -> None:
        fwd_atol = 2 * (ref_fp32 + 0.3 - 0.3 - ref_fp32).abs().max().item()
        pt_diff = (ref_low_precision.float() - ref_fp32).abs().max().item()
        kernel_diff = (actual.float() - ref_fp32).abs().max().item()
        allowed = 1.25 * (2 * pt_diff + fwd_atol)
        if actual.dtype == torch.float16:
            allowed = max(allowed, 5e-3)
        if kernel_diff > allowed:
            raise AssertionError(
                "MSA forward tolerance failed: "
                f"kernel_diff={kernel_diff:.8g}, pt_diff={pt_diff:.8g}, "
                f"fwd_atol={fwd_atol:.8g}, allowed={allowed:.8g}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed(seed: int | None, device: str | torch.device) -> None:
        if seed is None:
            return
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.device(device).type == "musa" and hasattr(torch, "musa"):
            torch.musa.manual_seed(seed)

    @staticmethod
    def _small_randn(
        shape: tuple[int, ...],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        scale: float = 1.0,
    ) -> torch.Tensor:
        return (torch.randn(shape, dtype=torch.float32) * scale).to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _maxscore_k_tiles(max_seqlen_k: int) -> int:
        kv_tiles = (int(max_seqlen_k) + 127) // 128
        return ((kv_tiles + 127) // 128) * 128

    @staticmethod
    def _page_counts_from_lens(kv_lens: torch.Tensor, page_size: int) -> list[int]:
        return [int((int(length) + page_size - 1) // page_size) for length in kv_lens]

    @staticmethod
    def _make_page_table_and_kv_indices(
        kv_lens: torch.Tensor,
        *,
        page_size: int,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_device = kv_lens.device
        page_counts = MsaOperator._page_counts_from_lens(kv_lens.cpu(), page_size)
        max_pages = max(page_counts, default=0)
        total_pages = sum(page_counts)
        if mode == "identity":
            physical_pages = list(range(total_pages))
        elif mode == "reverse":
            physical_pages = list(reversed(range(total_pages)))
        else:
            raise ValueError(f"unsupported page table mode: {mode}")

        page_table = torch.zeros(
            (len(page_counts), max_pages), dtype=torch.int32, device=output_device
        )
        flat_indices = []
        cursor = 0
        for batch_idx, count in enumerate(page_counts):
            pages = physical_pages[cursor : cursor + count]
            if pages:
                page_table[batch_idx, :count] = torch.tensor(
                    pages, dtype=torch.int32, device=output_device
                )
                flat_indices.extend(pages)
            cursor += count
        return page_table, torch.tensor(
            flat_indices, dtype=torch.int32, device=output_device
        )

    @staticmethod
    def _select_sparse_blocks(
        *,
        pattern: str,
        q_local: int,
        page_count: int,
        topk: int,
    ) -> list[int]:
        if page_count <= 0:
            return []
        if pattern == "first":
            selected = list(range(page_count))
        elif pattern == "tail":
            selected = list(range(max(0, page_count - topk), page_count))
        elif pattern in {"rolling", "rolling_holes"}:
            selected = []
            cursor = q_local % page_count
            while len(selected) < min(topk, page_count):
                if cursor not in selected:
                    selected.append(cursor)
                cursor = (cursor + 1) % page_count
        elif pattern == "sink_local":
            middle_count = max(0, topk - 2)
            middle = list(range(1, max(1, page_count - 1)))[:middle_count]
            selected = [0, *middle]
            if page_count > 1:
                selected.append(page_count - 1)
        else:
            raise ValueError(f"unsupported sparse block pattern: {pattern}")
        selected = selected[: min(topk, page_count)]
        if pattern == "rolling_holes" and q_local % 3 == 0 and len(selected) > 1:
            selected = selected[:-1]
        return sorted(selected)

    @staticmethod
    def _make_kv_block_indexes(
        q_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        *,
        num_kv_heads: int,
        topk: int,
        sparse_block_size: int,
        pattern: str,
        device: str | torch.device,
    ) -> torch.Tensor:
        q_lens_cpu = [int(v) for v in q_lens.cpu().tolist()]
        kv_lens_cpu = [int(v) for v in kv_lens.cpu().tolist()]
        total_q = sum(q_lens_cpu)
        q2k = torch.full(
            (total_q, num_kv_heads, topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        q_abs = 0
        for q_len, kv_len in zip(q_lens_cpu, kv_lens_cpu):
            page_count = (int(kv_len) + sparse_block_size - 1) // sparse_block_size
            for q_local in range(q_len):
                selected = MsaOperator._select_sparse_blocks(
                    pattern=pattern,
                    q_local=q_local,
                    page_count=page_count,
                    topk=topk,
                )
                if selected:
                    q2k[q_abs, :, : len(selected)] = torch.tensor(
                        selected,
                        dtype=torch.int32,
                        device=device,
                    ).view(1, len(selected))
                q_abs += 1
        return q2k.contiguous()

    @staticmethod
    def _dense_msa_output_maxscore_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        qo_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        num_kv_heads: int,
        max_k_tiles: int,
        *,
        causal: bool,
        qo_offset: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        page_size: int | None = None,
    ) -> torch.Tensor:
        total_q, num_qo_heads, _ = q.shape
        qhead_per_kv = num_qo_heads // num_kv_heads
        result = torch.full(
            (total_q, num_qo_heads, max_k_tiles),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
        q_lens_cpu = [int(v) for v in qo_lens.cpu().tolist()]
        kv_lens_cpu = [int(v) for v in kv_lens.cpu().tolist()]
        q_abs = 0
        kv_abs = 0
        for batch_idx, (q_len, kv_len) in enumerate(zip(q_lens_cpu, kv_lens_cpu)):
            q_seq = q[q_abs : q_abs + q_len].float()
            if page_table is None:
                k_seq = k[kv_abs : kv_abs + kv_len].float()
            else:
                assert page_size is not None
                pages = []
                for logical_page in range((kv_len + page_size - 1) // page_size):
                    physical_page = int(page_table[batch_idx, logical_page].item())
                    page_begin = logical_page * page_size
                    page_end = min(page_begin + page_size, kv_len)
                    pages.append(
                        k[physical_page, :, : page_end - page_begin]
                        .permute(1, 0, 2)
                        .float()
                    )
                k_seq = torch.cat(pages, dim=0) if pages else k.new_empty((0, 0, 0))
            causal_off = (
                int(qo_offset[batch_idx].item())
                if qo_offset is not None
                else kv_len - q_len
            )
            for tile_idx in range((kv_len + 127) // 128):
                k_begin = tile_idx * 128
                k_end = min(k_begin + 128, kv_len)
                k_tile = k_seq[k_begin:k_end].repeat_interleave(qhead_per_kv, dim=1)
                scores = torch.einsum("qhd,khd->qhk", q_seq, k_tile)
                if causal:
                    q_pos = (
                        torch.arange(q_len, device=q.device).unsqueeze(1) + causal_off
                    )
                    k_pos = torch.arange(k_begin, k_end, device=q.device).unsqueeze(0)
                    valid_mask = q_pos >= k_pos
                    scores = scores.masked_fill(~valid_mask.unsqueeze(1), -float("inf"))
                    tile_max = scores.max(dim=-1).values
                    tile_max = torch.where(
                        valid_mask.any(dim=1).unsqueeze(1),
                        tile_max,
                        torch.full_like(tile_max, -float("inf")),
                    )
                else:
                    tile_max = scores.max(dim=-1).values
                result[q_abs : q_abs + q_len, :, tile_idx] = tile_max
            q_abs += q_len
            kv_abs += kv_len
        return result

    @staticmethod
    def _dense_sparse_prefill_reference(
        q: torch.Tensor,
        k_pages_hpd: torch.Tensor,
        v_pages_hpd: torch.Tensor,
        page_table: torch.Tensor,
        kv_block_indexes: torch.Tensor,
        *,
        causal: bool,
        softmax_scale: float,
        qo_offset: int | None = None,
        q_lens: torch.Tensor | None = None,
        kv_lens: torch.Tensor | None = None,
        sparse_block_size: int = 128,
        upcast: bool = True,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        num_pages, num_kv_heads, page_size, head_dim_qk = k_pages_hpd.shape
        _, _, _, head_dim_v = v_pages_hpd.shape
        total_q, num_qo_heads, _ = q.shape
        qhead_per_kv = num_qo_heads // num_kv_heads
        batch_size = int(page_table.shape[0])
        if q_lens is None:
            if batch_size != 1:
                raise ValueError("q_lens is required for multi-batch reference inputs")
            q_lens_cpu = [total_q]
        else:
            q_lens_cpu = [int(v) for v in q_lens.cpu().tolist()]
        if kv_lens is None:
            kv_lens_cpu = [int(page_table.shape[1]) * page_size] * batch_size
        else:
            kv_lens_cpu = [int(v) for v in kv_lens.cpu().tolist()]
        if sum(q_lens_cpu) != total_q:
            raise ValueError(f"q_lens sum mismatch: {sum(q_lens_cpu)} vs {total_q}")

        out_dtype = q.dtype if output_dtype is None else output_dtype
        out = torch.zeros(
            (total_q, num_qo_heads, head_dim_v),
            dtype=out_dtype,
            device=q.device,
        )
        q_abs = 0
        for batch_idx, q_len in enumerate(q_lens_cpu):
            kv_len = kv_lens_cpu[batch_idx]
            q_offset = qo_offset if qo_offset is not None else kv_len - q_len
            for q_local in range(q_len):
                for head in range(num_kv_heads):
                    hq_begin = head * qhead_per_kv
                    hq_end = hq_begin + qhead_per_kv
                    k_chunks = []
                    v_chunks = []
                    for kv_block_idx in kv_block_indexes[q_abs, head].tolist():
                        if kv_block_idx < 0:
                            continue
                        block_start = int(kv_block_idx) * sparse_block_size
                        if block_start >= kv_len:
                            continue
                        block_end = min(block_start + sparse_block_size, kv_len)
                        if causal:
                            block_end = min(block_end, q_local + q_offset + 1)
                        if block_end <= block_start:
                            continue
                        cursor = block_start
                        while cursor < block_end:
                            logical_page = cursor // page_size
                            page_begin = cursor - logical_page * page_size
                            chunk_end = min(block_end, (logical_page + 1) * page_size)
                            page_end = chunk_end - logical_page * page_size
                            physical_page = int(
                                page_table[batch_idx, logical_page].item()
                            )
                            if physical_page < 0 or physical_page >= num_pages:
                                raise ValueError(
                                    f"invalid physical page {physical_page}"
                                )
                            k_chunks.append(
                                k_pages_hpd[physical_page, head, page_begin:page_end]
                            )
                            v_chunks.append(
                                v_pages_hpd[physical_page, head, page_begin:page_end]
                            )
                            cursor = chunk_end
                    if not k_chunks:
                        continue
                    k_sel = torch.cat(k_chunks, dim=0)
                    v_sel = torch.cat(v_chunks, dim=0)
                    q_sel = q[q_abs, hq_begin:hq_end]
                    if upcast:
                        k_sel = k_sel.to(torch.float32)
                        v_sel = v_sel.to(torch.float32)
                        q_sel = q_sel.to(torch.float32)
                    scores = torch.matmul(q_sel, k_sel.transpose(0, 1)) * softmax_scale
                    probs = torch.softmax(scores, dim=-1)
                    out[q_abs, hq_begin:hq_end] = torch.matmul(probs, v_sel).to(
                        dtype=out_dtype
                    )
                q_abs += 1
        return out

    @staticmethod
    def _sparse_topk_select_reference(
        max_score: torch.Tensor,
        *,
        topk: int,
        num_valid_pages: int | None = None,
        force_begin_blocks: int = 0,
        force_end_blocks: int = 0,
        force_blocks_count_in_topk: bool = True,
        query_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_q, num_qo_heads, max_k_tiles = max_score.shape
        valid_pages = max_k_tiles if num_valid_pages is None else int(num_valid_pages)
        output_width = (
            topk
            if force_blocks_count_in_topk
            else topk + force_begin_blocks + force_end_blocks
        )
        out = torch.full(
            (total_q, num_qo_heads, output_width),
            -1,
            dtype=torch.int32,
            device=max_score.device,
        )
        for head in range(num_qo_heads):
            for q_abs in range(total_q):
                row_valid_pages = valid_pages
                if query_positions is not None:
                    position = int(query_positions[q_abs].item())
                    row_valid_pages = (
                        0 if position < 0 else min(valid_pages, position // 128 + 1)
                    )
                forced = set(range(min(force_begin_blocks, row_valid_pages)))
                force_end_start = max(0, row_valid_pages - force_end_blocks)
                forced.update(range(force_end_start, row_valid_pages))
                remaining = [idx for idx in range(row_valid_pages) if idx not in forced]
                remaining.sort(
                    key=lambda idx: (
                        -float(max_score[q_abs, head, idx].item()),
                        idx,
                    )
                )
                if force_blocks_count_in_topk:
                    candidates = sorted(forced)
                    candidates.extend(remaining[: max(0, topk - len(candidates))])
                    candidates = sorted(candidates[:topk])
                else:
                    candidates = sorted(forced.union(remaining[:topk]))
                if candidates:
                    out[q_abs, head, : len(candidates)] = torch.tensor(
                        candidates,
                        dtype=torch.int32,
                        device=max_score.device,
                    )
        return out
