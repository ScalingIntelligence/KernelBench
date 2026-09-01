from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from mate.testing.flash_attn import gen_input_tensor, generate_block_kvcache


@dataclass
class KVTensors:
    """Generated KV tensors for flash attention benchmark."""

    k: torch.Tensor
    v: torch.Tensor
    page_table: Optional[torch.Tensor] = None


@dataclass
class BenchmarkMetrics:
    """Computed benchmark metrics from config."""

    flops: int
    io_bytes: int
    bench_seqlens_q: List[int]
    bench_avg_seqlens_kv: List[float]
    total_seqlen_q: int
    total_seqlen_kv: float


@dataclass
class FlashAttnBenchConfig:
    """Configuration for a single flash attention benchmark instance."""

    # === Tensor Dimensions ===
    batch_size: int
    seqlen_q: Union[int, List[int]]
    seqlen_kv: Union[int, List[int]]
    head_q: int
    head_kv: int
    headdim_qk: int
    headdim_vo: int

    # === Attention Flags ===
    is_packgqa: bool
    is_causal: bool
    window_size: Tuple[Optional[int], Optional[int]]

    # === Additional Sequence Info ===
    seqused_q: Optional[List[int]] = None
    seqused_kv: Optional[List[int]] = None

    # === Execution Environment ===
    device: str = "musa"
    dtype: torch.dtype = torch.bfloat16
    page_size: Optional[int] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        # Validate seqlen_q matches batch_size
        if isinstance(self.seqlen_q, list):
            assert len(self.seqlen_q) == self.batch_size, (
                f"len(seqlen_q)={len(self.seqlen_q)} != batch_size={self.batch_size}"
            )

        # Validate seqlen_kv matches batch_size
        if isinstance(self.seqlen_kv, list):
            assert len(self.seqlen_kv) == self.batch_size, (
                f"len(seqlen_kv)={len(self.seqlen_kv)} != batch_size={self.batch_size}"
            )

        # Paged attention: seqused_kv always equals seqlen_kv
        if self.is_paged_kv:
            assert isinstance(self.seqlen_kv, list), (
                "seqlen_kv must be list for paged attention"
            )
            self.seqused_kv = list(self.seqlen_kv)

        assert self.head_q % self.head_kv == 0, (
            f"head_q ({self.head_q}) must be divisible by "
            f"head_kv ({self.head_kv}) for GQA"
        )

        if self.seqused_q is not None:
            assert len(self.seqused_q) == self.batch_size, (
                f"len(seqused_q)={len(self.seqused_q)} != batch_size={self.batch_size}"
            )

        if self.window_size != (None, None):
            assert (
                self.window_size[0] is not None and self.window_size[1] is not None
            ), (
                f"window_size must be either (None, None) or (left, right) with both integers, "
                f"got {self.window_size}"
            )
            assert isinstance(self.window_size[0], int) and isinstance(
                self.window_size[1], int
            ), f"window_size values must be integers, got {self.window_size}"

    @property
    def is_varlen_q(self) -> bool:
        return isinstance(self.seqlen_q, list)

    @property
    def is_varlen_kv(self) -> bool:
        return isinstance(self.seqlen_kv, list)

    @property
    def max_seqlen_q(self) -> int:
        return self.seqlen_q if isinstance(self.seqlen_q, int) else max(self.seqlen_q)

    @property
    def max_seqlen_kv(self) -> int:
        return (
            self.seqlen_kv if isinstance(self.seqlen_kv, int) else max(self.seqlen_kv)
        )

    @property
    def is_paged_kv(self) -> bool:
        return self.page_size is not None

    @property
    def is_local(self) -> bool:
        return self.window_size[0] is not None and self.window_size[1] is not None

    @property
    def dtype_gen(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == torch.float8_e4m3fn else self.dtype

    @property
    def elem_size(self) -> int:
        """Element size in bytes based on dtype."""
        return torch.finfo(self.dtype).bits // 8

    @property
    def cu_seqlens_q(self) -> Optional[torch.Tensor]:
        if not self.is_varlen_q:
            return None
        seqlen_tensor = torch.tensor(
            self.seqlen_q, dtype=torch.int32, device=self.device
        )
        return torch.nn.functional.pad(torch.cumsum(seqlen_tensor, dim=0), (1, 0)).to(
            torch.int32
        )

    @property
    def cu_seqlens_kv(self) -> Optional[torch.Tensor]:
        if not self.is_varlen_kv or self.is_paged_kv:
            return None
        seqlen_tensor = torch.tensor(
            self.seqlen_kv, dtype=torch.int32, device=self.device
        )
        return torch.nn.functional.pad(torch.cumsum(seqlen_tensor, dim=0), (1, 0)).to(
            torch.int32
        )

    @property
    def seqused_q_tensor(self) -> Optional[torch.Tensor]:
        if self.seqused_q is None:
            return None
        return torch.tensor(self.seqused_q, dtype=torch.int32, device=self.device)

    @property
    def seqused_kv_tensor(self) -> Optional[torch.Tensor]:
        if self.seqused_kv is None:
            return None
        return torch.tensor(self.seqused_kv, dtype=torch.int32, device=self.device)

    @property
    def q_tensor(self) -> torch.Tensor:
        """Generate Q tensor on demand (lazy)."""
        q, _, _ = gen_input_tensor(
            batch_size=self.batch_size,
            seqlens=self.seqlen_q if self.is_varlen_q else self.max_seqlen_q,
            num_head=self.head_q,
            headdim=self.headdim_qk,
            use_cu_seqlens=self.is_varlen_q,
            datatype=self.dtype_gen,
            device=self.device,
        )
        if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            q = q.to(self.dtype)
        return q

    @property
    def kv_tensor(self) -> KVTensors:
        """Generate KV tensors on demand (lazy)."""
        if self.is_paged_kv:
            _, _, page_table, k_paged, v_paged, _ = generate_block_kvcache(
                max_seqlen_kv=self.max_seqlen_kv,
                page_size=self.page_size,
                batch_size=self.batch_size,
                head_kv=self.head_kv,
                headdim_qk=self.headdim_qk,
                headdim_vo=self.headdim_vo,
                device=self.device,
                dtype=self.dtype_gen,
            )
            if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                k_paged = k_paged.to(self.dtype)
                v_paged = v_paged.to(self.dtype)
            return KVTensors(k=k_paged, v=v_paged, page_table=page_table)
        else:
            k, _, _ = gen_input_tensor(
                batch_size=self.batch_size,
                seqlens=self.seqlen_kv if self.is_varlen_kv else self.max_seqlen_kv,
                num_head=self.head_kv,
                headdim=self.headdim_qk,
                use_cu_seqlens=self.is_varlen_kv,
                datatype=self.dtype_gen,
                device=self.device,
            )
            v, _, _ = gen_input_tensor(
                batch_size=self.batch_size,
                seqlens=self.seqlen_kv if self.is_varlen_kv else self.max_seqlen_kv,
                num_head=self.head_kv,
                headdim=self.headdim_vo,
                use_cu_seqlens=self.is_varlen_kv,
                datatype=self.dtype_gen,
                device=self.device,
            )
            if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                k = k.to(self.dtype)
                v = v.to(self.dtype)
            return KVTensors(k=k, v=v)

    @property
    def bench_seqlens_q(self) -> List[int]:
        """Effective Q sequence lengths for benchmarking."""
        if self.is_varlen_q:
            return self.seqlen_q  # type: ignore
        else:
            return [self.seqlen_q] * self.batch_size  # type: ignore

    @property
    def bench_seqlens_kv(self) -> List[int]:
        """Effective KV sequence lengths for benchmarking."""
        if self.is_varlen_kv:
            return self.seqlen_kv  # type: ignore
        else:
            return [self.seqlen_kv] * self.batch_size  # type: ignore

    @property
    def bench_avg_seqlens_kv(self) -> List[float]:
        """Effective average KV sequence lengths for benchmarking."""
        bench_q = self.bench_seqlens_q
        bench_kv = self.bench_seqlens_kv

        if self.is_causal:
            return [(max(0, k - q) + k) / 2 for q, k in zip(bench_q, bench_kv)]
        elif self.is_local:
            result = []
            for b in range(self.batch_size):
                row_idx = torch.arange(bench_q[b])
                col_left = torch.maximum(
                    row_idx + bench_kv[b] - bench_q[b] - self.window_size[0],
                    torch.tensor(0),
                )
                col_right = torch.minimum(
                    row_idx + bench_kv[b] - bench_q[b] + self.window_size[1],
                    torch.tensor(bench_kv[b] - 1),
                )
                avg_seqlen = (col_right - col_left + 1).float().mean().item()
                result.append(avg_seqlen)
            return result
        else:
            if self.is_varlen_kv or self.is_paged_kv:
                return [float(s) for s in bench_kv]
            else:
                return [float(self.max_seqlen_kv) for _ in range(self.batch_size)]

    def get_metrics(self) -> BenchmarkMetrics:
        """Calculate FLOPs and I/O bytes from config."""
        bench_q = self.bench_seqlens_q
        bench_kv = self.bench_avg_seqlens_kv

        total_q = sum(bench_q)
        total_kv = sum(bench_kv)

        elem_size = self.elem_size
        elem_lse = 4

        io_bytes = (
            total_q * self.head_q * self.headdim_qk  # load q
            + total_kv * self.head_kv * self.headdim_qk  # load k
            + total_kv * self.head_kv * self.headdim_vo  # load v
            + total_q * self.head_q * self.headdim_vo  # write o
        ) * elem_size + total_q * self.head_q * elem_lse  # write lse

        flops = (
            2
            * self.head_q
            * (self.headdim_qk + self.headdim_vo)
            * sum(q * k for q, k in zip(bench_q, bench_kv))
        )

        return BenchmarkMetrics(
            flops=int(flops),
            io_bytes=int(io_bytes),
            bench_seqlens_q=bench_q,
            bench_avg_seqlens_kv=bench_kv,
            total_seqlen_q=total_q,
            total_seqlen_kv=total_kv,
        )


def test():
    """Test FlashAttnBenchConfig with simple examples."""

    cfg1 = FlashAttnBenchConfig(
        batch_size=4,
        seqlen_q=[1, 1, 1, 1],
        seqlen_kv=[4096, 4096, 4096, 4096],
        head_q=64,
        head_kv=4,
        headdim_qk=128,
        headdim_vo=128,
        is_packgqa=True,
        is_causal=True,
        window_size=(None, None),
        seqused_kv=[4000, 4096, 3500, 4096],
        device="musa",
        dtype=torch.bfloat16,
        page_size=64,
    )

    print("=== Example 1: Paged attention with varlen Q ===")
    print(f"batch_size: {cfg1.batch_size}")
    print(f"max_seqlen_q: {cfg1.max_seqlen_q}")
    print(f"max_seqlen_kv: {cfg1.max_seqlen_kv}")
    print(f"is_varlen_q: {cfg1.is_varlen_q}")
    print(f"is_varlen_kv: {cfg1.is_varlen_kv}")
    print(f"is_paged_kv: {cfg1.is_paged_kv}")
    print(f"is_local: {cfg1.is_local}")
    print(f"cu_seqlens_q: {cfg1.cu_seqlens_q}")
    print(f"seqused_kv_tensor: {cfg1.seqused_kv_tensor}")

    cfg2 = FlashAttnBenchConfig(
        batch_size=1,
        seqlen_q=4096,
        seqlen_kv=4096,
        head_q=32,
        head_kv=8,
        headdim_qk=128,
        headdim_vo=128,
        is_packgqa=False,
        is_causal=True,
        window_size=(None, None),
        page_size=None,
    )

    print("\n=== Example 2: Non-paged, non-varlen attention ===")
    print(f"is_varlen_q: {cfg2.is_varlen_q}")
    print(f"is_varlen_kv: {cfg2.is_varlen_kv}")
    print(f"is_paged_kv: {cfg2.is_paged_kv}")
    print(f"cu_seqlens_q: {cfg2.cu_seqlens_q}")

    print("\n=== Tensor Generation (Example 1) ===")
    print(f"q_tensor shape: {cfg1.q_tensor.shape}")
    kv1 = cfg1.kv_tensor
    print(f"kv_tensor.k shape: {kv1.k.shape}")
    print(f"kv_tensor.v shape: {kv1.v.shape}")
    if kv1.page_table is not None:
        print(f"kv_tensor.page_table shape: {kv1.page_table.shape}")

    print("\n=== Tensor Generation (Example 2) ===")
    print(f"q_tensor shape: {cfg2.q_tensor.shape}")
    kv2 = cfg2.kv_tensor
    print(f"kv_tensor.k shape: {kv2.k.shape}")
    print(f"kv_tensor.v shape: {kv2.v.shape}")
    if kv2.page_table is not None:
        print(f"kv_tensor.page_table shape: {kv2.page_table.shape}")

    print("\n=== Metrics (Example 1) ===")
    metrics1 = cfg1.get_metrics()
    print(f"flops: {metrics1.flops}")
    print(f"io_bytes: {metrics1.io_bytes}")
    print(f"bench_seqlens_q: {metrics1.bench_seqlens_q}")
    print(f"bench_avg_seqlens_kv: {metrics1.bench_avg_seqlens_kv}")
    print(f"total_seqlen_q: {metrics1.total_seqlen_q}")
    print(f"total_seqlen_kv: {metrics1.total_seqlen_kv}")

    print("\n=== Metrics (Example 2) ===")
    metrics2 = cfg2.get_metrics()
    print(f"flops: {metrics2.flops}")
    print(f"io_bytes: {metrics2.io_bytes}")
    print(f"bench_seqlens_q: {metrics2.bench_seqlens_q}")
    print(f"bench_avg_seqlens_kv: {metrics2.bench_avg_seqlens_kv}")
    print(f"total_seqlen_q: {metrics2.total_seqlen_q}")
    print(f"total_seqlen_kv: {metrics2.total_seqlen_kv}")


if __name__ == "__main__":
    test()
