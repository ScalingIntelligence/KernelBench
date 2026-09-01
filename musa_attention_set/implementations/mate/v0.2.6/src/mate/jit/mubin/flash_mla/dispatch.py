from dataclasses import asdict
import functools
import hashlib
import json
from pathlib import Path

from mate.artifacts import ensure_mubin_kernel_artifact, load_kernel_map

from ..common import (
    MP31_ARCH,
    get_asm_dtype_from_torch_dtype,
)
from .types import FlashMLAMubinId


@functools.cache
def get_flash_mla_mubin_id_hash(asm_id: FlashMLAMubinId) -> str:
    payload = asdict(asm_id)
    payload["dtype"] = get_asm_dtype_from_torch_dtype(asm_id.dtype)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FlashMLAMubinDispatcher:
    def __init__(self, kernel_map_path: Path):
        self.kernel_map_path = Path(kernel_map_path)
        self.module_dir = self.kernel_map_path.parent
        self._kernel_hash_map = {
            entry.dispatch_hash: entry
            for entry in load_kernel_map(self.kernel_map_path)
        }

    @functools.cache
    def get_asm_id(self, dtype, is_causal: bool, is_varlen_q: bool) -> FlashMLAMubinId:
        return FlashMLAMubinId(
            arch=MP31_ARCH,
            dtype=dtype,
            is_causal=is_causal,
            is_varlen_q=is_varlen_q,
        )

    @functools.cache
    def resolve_kernel_entry(self, asm_id: FlashMLAMubinId):
        asm_id_hash = get_flash_mla_mubin_id_hash(asm_id)
        entry = self._kernel_hash_map.get(asm_id_hash)
        if entry is None:
            raise ValueError(f"No FlashMLA mubin kernel found for hash {asm_id_hash}")
        return entry

    @functools.cache
    def resolve_kernel_path(self, asm_id: FlashMLAMubinId) -> Path:
        entry = self.resolve_kernel_entry(asm_id)
        return ensure_mubin_kernel_artifact("flash_mla", self.module_dir, entry)


@functools.cache
def get_flash_mla_mubin_dispatcher(kernel_map_path: Path):
    return FlashMLAMubinDispatcher(kernel_map_path)
