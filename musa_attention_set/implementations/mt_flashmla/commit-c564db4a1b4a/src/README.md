# MT-FlashMLA

MT-FlashMLA is an efficient MLA decoding kernel for MooreThreads GPU (Compute Capability 3.1), optimized for variable-length sequences serving.

Currently released:
- BF16, FP16
- Paged kvcache with block size of 64

## Quick start

### Install

```bash
python setup_musa.py install
```

### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    ...
```

## Requirements

- MooreThreads GPU (Compute Capability 3.1)
- MUSA 4.0.0 and above
- torch_musa 2.5.0 and above

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel}, 
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
