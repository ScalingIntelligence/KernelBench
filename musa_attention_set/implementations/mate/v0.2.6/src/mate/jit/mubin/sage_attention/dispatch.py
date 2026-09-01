from dataclasses import asdict
import functools
import hashlib
import json
from pathlib import Path

import torch

from mate.artifacts import ensure_mubin_kernel_artifact, load_kernel_map

from ..common import (
    MP31_ARCH,
    get_asm_dtype_from_torch_dtype,
)
from .types import SageAttentionMubinId


@functools.cache
def get_sage_attention_mubin_id_hash(asm_id: SageAttentionMubinId) -> str:
    payload = asdict(asm_id)
    for dtype_key in ("q_dtype", "k_dtype", "v_dtype"):
        payload[dtype_key] = get_asm_dtype_from_torch_dtype(getattr(asm_id, dtype_key))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SageAttentionMubinDispatcher:
    def __init__(self, kernel_map_path: Path):
        self.kernel_map_path = Path(kernel_map_path)
        self.module_dir = self.kernel_map_path.parent
        self._kernel_hash_map = {
            entry.dispatch_hash: entry
            for entry in load_kernel_map(self.kernel_map_path)
        }

    @functools.cache
    def get_asm_id(
        self,
        q_dtype,
        k_dtype,
        v_dtype,
        is_causal: bool,
        is_kv_cache: bool,
        headdim_qk: int,
        quant_mode: int,
        fp8_output: bool,
    ) -> SageAttentionMubinId:
        is_qk_int8 = (
            q_dtype == torch.int8
            and k_dtype == torch.int8
            and v_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        )
        return SageAttentionMubinId(
            arch=MP31_ARCH,
            q_dtype=q_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            is_causal=is_causal,
            is_kv_cache=is_kv_cache,
            is_varlen=False,
            headdim_qk=128 if headdim_qk <= 128 else headdim_qk,
            quant_mode=quant_mode,
            is_qk_int8=is_qk_int8,
            fp8_output=fp8_output,
        )

    @functools.cache
    def resolve_kernel_entry(self, asm_id: SageAttentionMubinId):
        asm_id_hash = get_sage_attention_mubin_id_hash(asm_id)
        entry = self._kernel_hash_map.get(asm_id_hash)
        if entry is None:
            raise ValueError(
                f"No SageAttention mubin kernel found for hash {asm_id_hash}"
            )
        return entry

    @functools.cache
    def resolve_kernel_path(self, asm_id: SageAttentionMubinId) -> Path:
        entry = self.resolve_kernel_entry(asm_id)
        return ensure_mubin_kernel_artifact("sage_attention", self.module_dir, entry)


@functools.cache
def get_sage_attention_mubin_dispatcher(kernel_map_path: Path):
    return SageAttentionMubinDispatcher(kernel_map_path)
