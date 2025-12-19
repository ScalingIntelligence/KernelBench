# Evaluation
[WIP] Benchmarking Guide
To be updated more comprehensively with the benchmarking guide & blog that we have been working on this quarter.

You should be **extra careful!** 

If the model can reward hack, it will find ways to reward hack!

## Methodology


### Correctness

### Performance
We (are) implementing a few ways for timing and understand the tradeoffs.

### Unit Tests

We've included some unit tests for the eval script in `src/unit_tests/test_eval_adversarial.py`. These tests run adversarial kernels (see `src/unit_tests/test_kernels/`) that contain examples of reward hacking that we've seen from LLMs and ensures that the eval script catches them, either by failing their correctness checks or flagging them for excessive speedups. Examples include:
- Reusing computations cached during the PyTorch reference
- Modifying inputs to cheat correctness checks
- Moving computation to a non-default CUDA stream

We will continue to add more tests as we explore additional adversarial scenarios.

 
Note this is an ongoing community effort. 