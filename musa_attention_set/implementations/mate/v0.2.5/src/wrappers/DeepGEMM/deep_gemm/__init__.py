from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .interface import (
    bf16_m_grouped_gemm_nt_masked,
    bf16_gemm_nt,
    # legacy aliases
    fp8_einsum,
    fp8_m_grouped_gemm_nt_masked,
    fp8_gemm_nt,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from . import testing as testing
from . import utils as utils
from .utils import (
    get_col_major_tma_aligned_tensor as get_col_major_tma_aligned_tensor,
)
from .utils import (
    get_mk_alignment_for_contiguous_layout as get_mk_alignment_for_contiguous_layout,
)
from .utils import get_mn_major_tma_aligned_tensor as get_mn_major_tma_aligned_tensor
from .utils import get_num_sms as get_num_sms
from .utils import get_tc_util as get_tc_util
from .utils import set_num_sms as set_num_sms
from .utils import set_tc_util as set_tc_util

try:
    from ._build_meta import __git_version__ as __git_version__
except Exception:
    __git_version__ = "unknown"


def _load_version() -> str:
    try:
        return version("deep-gemm")
    except PackageNotFoundError:
        pass

    try:
        from ._build_meta import __version__ as build_version

        return build_version
    except Exception:
        pass

    try:
        return (
            (Path(__file__).resolve().parents[3] / "version.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


__version__ = _load_version()
# utilities
# "get_num_sms",
# "set_num_sms",
# "get_tc_util",
# "set_tc_util",
# "get_mk_alignment_for_contiguous_layout",
# "get_col_major_tma_aligned_tensor",
# "get_mn_major_tma_aligned_tensor",

__all__ = [
    "__git_version__",
    "__version__",
    "testing",
    "utils",
    # GEMM
    "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "fp8_gemm_nt",
    "fp8_einsum",
    # MQA
    "get_paged_mqa_logits_metadata",
    "fp8_paged_mqa_logits",
    "fp8_mqa_logits",
    # legacy aliases
    "fp8_m_grouped_gemm_nt_masked",
    "bf16_m_grouped_gemm_nt_masked",
]
