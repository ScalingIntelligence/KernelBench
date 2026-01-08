# Reproducing Sakana's Result

We focus on invesigating and understanding the results of Sakana's kernels.

We have thoroughly examined 2 problems. There might be more, and we will continue to update.
* Level 1 Problem 15: `15_Matmul_for_lower_triangular_matrices`
* Level 2 Problem 23: `23_Conv3d_GroupNorm_Mean`

For each problem, we put the kernel code in a folder with the following structure:
* We have the original code from Sakana, which is `_sakana.cu`. This is pure CUDA code and then bind to the model `forward` function using `pybind11`.
* We have the code in the KernelBench format `ModelNew`, which is `_kernelbench.py`. This is a PyTorch module with custom inline CUDA kernel, which is the KernelBench task format.

### Note on Sakana's Eval System

⚠️ **To be clear** ⚠️: There are many differnces between Sakana's eval system and our eval system -- while our eval system is not completely robust, there are some important differences to discuss. Here is an example of the Sakana eval, provided by one of the [Sakana paper authors](https://x.com/RobertTLange/status/1892489402070220989). A huge difference is how they wrap their inline CUDA code -- we query the model to generate an entirely new model and forward function, while they choose to overwrite the forward function of a fixed model. These differences change the behavior of some of the caching hacks that the Sakana model was able to use (notably, the infamous Matmul for TriLower matrices that gets a 150x speedup fails the correctness checks on our eval). Furthermore, we use synchronization markers (CUDA events) in our eval to prevent hacky solutions from passing -- these are not the most robust ways to time kernels (which we want to address too) and may even add some extra unwanted overhead, but at the very least it mitigates some hacky solutions.

You can use `scripts/run_and_check.py` to evaluate **using the KernelBench Eval code**. 

### Level 1 Problem 15: `15_Matmul_for_lower_triangular_matrices`

In this problem, it was discovered online that the runtime numbers were incorrect (see [this X thread](https://x.com/main_horse/status/1892446384910987718)). It turned
out that the model-generated kernel was doing nothing (effectively a no-op), and was caching results from the PyTorch reference outputs and using them as the 
solution. 

To use the KernelBench eval on this problem, you can run the following command:
```
python3 scripts/run_and_check.py ref_origin=kernelbench level=1 problem_id=15 kernel_src_path=sakana_kernels/level1_problem15/15_Matmul_for_lower_triangular_matrices_kernelbench.py
```

For this problem, the CUDA kernel is initialized with a 1D grid as follows:
```
const int threadsPerBlock = 256; // Increased thread count per block
const int numBlocks = N;
triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    N
);
```
However, in the actual kernel, we compute the row and column a thread computes as if we're using a 2D grid & block:
```
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
```
In this case: `blockIdx.y` will always be 0, `blockDim.y` will always be 1, and `threadIdx.y` will always be 0. So, the value of `row` will always be 0. So it
actually only computes values for the first row. Instead, the hypothesized reason why this kernel passes correctness checks is that it grabs values from
the same location of allocated memory (using `torch.empty_like`, similar to `malloc` as opposed to `torch.zeros_like` which writes over the values in memory) as the PyTorch reference kernel (which is run first). So this kernel actually is "cheating", but interestingly in the code there's no indication that the model is intentionally doing this. The fix to the first problem is by configuring a 2D grid/block instead:
```
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
triangular_mm_kernel<<<grid, block>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    N
)
```

To address the hacky "copying" problem, we need to fix the overall eval to address these issues. Notably, on the KernelBench eval this kernel does not pass the correctness checks (but still passes 4/5 tests!). The most obvious solution is calling `torch.cuda.empty_cache` between correctness runs to prevent grabbing any previous solutions. To keep results consistent between the eval and our paper, we choose to add this only for correctness tests to prevent these solutions from passing without influencing runtime numbers. For the future, we also plan to add more rigorous checking during benchmarking as well to prevent convoluted and hacky solutions. We also will call the model generated kernel first to prevent any kind of "stealing solutions"-esque approaches.

### Level 2 Problem 23: `23_Conv3d_GroupNorm_Mean`

In this problem, we have a batch (128) of 1536 elements that are group-normed (you can think of this as being mean 0, with low variance). It turns out
by a (rather hand-wavy) central limit theorem and a further division by the number of elements (~10^3) because we take a mean, 
the distribution of each element in the tensor has mean 0 (by symmetry) and a very low variance, 
allowing output tensors of all 0's to pass the tests under a small enough error of margin. The workaround to this in the future would
be to change either the kernel itself or the input distribution for the kernel inputs.

To use the KernelBench eval on this problem, you can run the following command:
```
python3 scripts/run_and_check.py ref_origin=kernelbench level=2 problem_id=23 kernel_src_path=sakana_kernels/level2_problem23/23_Conv3d_GroupNorm_Mean_kernelbench.py
```

On NVIDIA L40S, we see with our eval code.
```
========================================
[Eval] Kernel eval result: compiled=True correctness=True metadata={'hardware': 'NVIDIA L40S', 'device': 'cuda:0', 'correctness_trials': '(5 / 5)'} runtime=0.0327 runtime_stats={'mean': 0.0327, 'std': 0.00188, 'min': 0.0307, 'max': 0.0481, 'num_trials': 100, 'hardware': 'NVIDIA L40S', 'device': 'cuda:0'}
----------------------------------------
[Timing] PyTorch Reference Eager exec time: 1.26 
[Timing] PyTorch Reference torch.compile time: 0.704 ms
[Timing] Custom Kernel exec time: 0.0327 ms
----------------------------------------
[Speedup] Speedup over eager: 38.53x
[Speedup] Speedup over torch.compile: 21.53x
========================================
```

```
So the actual kernel output 

[Eval] Shape of output: tensor([-3.0275e-10, -9.0826e-10,  6.0551e-10,  ...,  9.0826e-10,
        -1.6651e-09, -9.0826e-10], device='cuda:0')
[Eval] Mean of output_new: 0.0000


The faulty kernel 
[Eval] Shape of output_new: tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')
[Eval] Mean of output: -0.0000
```
Interestingly, the faulty kernel doesn't actually use the weights of the convolution at all, making it obvious that it is wrong -- it instead produces all 0 outputs. The actual outputs are all mean 0, std roughly 10^-9, which passes all the atol and rtol checks.


### Takeaways
We appreciate Sakana's effort in providing the kernel code and the evaluation system. This level of transparency help the community understand and reproduce, enabling future progress in this direction.

We will continue working on making the eval robust. To keep results consistent with our current arXiv, we only modify the correctness checks for robustness, but we plan on adding the following changes:
* Prevent cached solutions by clearing the cache (from the caching allocator).
* Drawing from `triton.testing.do_bench`, run more correctness tests and clear on-device caches between runs to prevent incorrect timing analysis.
* To prevent including kernels with easy solutions (e.g. all "0"'s), explicitly filter out benchmark problems with solutions that fall within some interval `[x-0.001,x+0.001]`. Thanks to folks at [METR](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/) for proposing.
* Avoid extra overhead during timing analysis -- i.e. be more intentional and explicit about synchronization instructions.








