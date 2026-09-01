from .fmha_get_metadata import (
    _fmha_get_metadata,  # noqa: F401
    gen_fmha_metadata_specs,
    get_fmha_metadata_aot_configs,
    gen_fmha_metadata_aot,
)
from .fmha_fwd import (
    _fmha_fwd,  # noqa: F401
    gen_fmha_fwd_specs,
    get_fmha_fwd_aot_configs,
    gen_fmha_fwd_aot,
)

from .fmha_combine import (
    _fmha_fwd_combine,  # noqa: F401
    gen_fmha_fwd_combine_specs,
    get_fmha_fwd_combine_aot_configs,
    gen_fmha_fwd_combine_aot,
)
from ...flash_attention_ops import gen_flash_attention_ops_aot


def gen_fmha_aot(config_level: int = 0):
    if config_level == 0:
        return []

    specs = []
    specs.extend(gen_fmha_metadata_aot(config_level))
    specs.extend(gen_fmha_fwd_combine_aot(config_level))
    specs.extend(gen_fmha_fwd_aot(config_level))
    specs.extend(gen_flash_attention_ops_aot())
    return specs


__all__ = [
    "_fmha_get_metadata",
    "_fmha_fwd",
    "_fmha_fwd_combine",
    "gen_fmha_aot",
    "gen_fmha_metadata_specs",
    "gen_fmha_fwd_specs",
    "gen_fmha_fwd_combine_specs",
    "get_fmha_metadata_aot_configs",
    "get_fmha_fwd_aot_configs",
    "get_fmha_fwd_combine_aot_configs",
]
