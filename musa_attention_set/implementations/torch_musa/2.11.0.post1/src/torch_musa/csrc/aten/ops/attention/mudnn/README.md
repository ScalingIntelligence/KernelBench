## Scaled-dot-product Attention Computing

Torch Musa provides `torch.nn.functional.scaled_dot_product_attention` as the same in PyTorch CUDA. Now there are two modes for computing attention behind this api:
* math mode
* flash attention mode


Math mode is suitable for all kinds of Attention computing but may be not the optimal one. 

FlashAttention mode now supports attention with head dim less than 64, 128 or equal to 160, dtype `half`. Now it doesn't support backward, thus if you use Flashattention mode in forward, it will fail in backward. So in training, please disable flashattention now.


You could follow the backends enable/disable methods same as [https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel](PyTorch CUDA). For example, disable flashattention with `sdp_kernel` context manager: 

```
python

# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
with torch.backends.cuda.sdp_kernel(enable_flash=False):
    F.scaled_dot_product_attention(query,key,value)

```
