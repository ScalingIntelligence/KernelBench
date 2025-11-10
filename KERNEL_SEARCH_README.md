# AIDE + KernelBench Integration

This integration combines **AIDE ML's tree search** framework with **KernelBench's kernel evaluation** pipeline to enable iterative CUDA kernel optimization through automated search.

## Overview

The integration enables:
- **Tree-based search** over kernel implementations (draft → improve → debug cycles)
- **Compilation checking** with detailed error feedback
- **Correctness validation** against PyTorch reference implementations
- **Performance profiling** with GPU timing measurements
- **Iterative improvement** guided by LLM with execution feedback

## Architecture

```
┌─────────────────────────────────────┐
│  KernelAgent (Tree Search)          │
│  • Draft initial kernels             │
│  • Improve working kernels           │
│  • Debug failed kernels              │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KernelInterpreter (Evaluation)     │
│  • Compile CUDA code                 │
│  • Check correctness                 │
│  • Measure performance               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Journal (Solution Tree)            │
│  • Track all attempts                │
│  • Maintain parent-child relations   │
│  • Store metrics and feedback        │
└─────────────────────────────────────┘
```

## Files Created

### Core Integration Files
- **`kernel_agent.py`**: AIDE Agent extended for CUDA kernel optimization
  - Custom prompts for kernel development
  - Error-type-specific debugging strategies
  - Performance-focused improvement suggestions

- **`kernel_interpreter.py`**: Interpreter wrapping KernelBench evaluation
  - Maps `KernelExecResult` → `ExecutionResult` for AIDE compatibility
  - Formats compilation/correctness/performance feedback
  - Handles GPU device management

- **`backend.py`**: LLM query interface for AIDE
  - Adapts KernelBench's `query_server` for AIDE
  - Converts structured prompts to markdown
  - Handles multiple LLM providers

- **`kernel_config.yaml`**: Configuration template
  - Problem specification (level, problem_id)
  - GPU settings (architecture, precision, backend)
  - Search parameters (num_drafts, debug_prob, steps)
  - LLM inference settings

### Entry Points
- **`run_kernel_search.py`**: Main orchestrator for kernel optimization
  - Loads reference architecture from KernelBench
  - Runs tree search loop with rich terminal UI
  - Saves results, visualizations, and reports

- **`run_kernel_search_simple.py`**: Command-line wrapper
  - Simplified argument parsing
  - Builds configuration from CLI args
  - Convenient for quick experiments

## Usage

### Quick Start

```bash
# Basic search on Level 1, Problem 1
python run_kernel_search_simple.py --level 1 --problem_id 1 --steps 10

# With specific settings
python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 19 \
    --backend cuda \
    --precision fp32 \
    --steps 20 \
    --num_drafts 3 \
    --server_type google
```

### Using Configuration File

```bash
# Edit kernel_config.yaml, then run:
python run_kernel_search.py

# Override specific settings:
python run_kernel_search.py \
    kernel.level=2 \
    kernel.problem_id=5 \
    agent.steps=30
```

### Command-Line Arguments

**Problem Selection:**
- `--level`: KernelBench level (1-4)
- `--problem_id`: Problem ID within level
- `--dataset_src`: "local" or "huggingface"

**GPU Settings:**
- `--backend`: cuda | triton | tilelang | cute
- `--precision`: fp32 | fp16 | bf16
- `--gpu_arch`: Ada, Ampere, Volta, etc.

**Search Configuration:**
- `--steps`: Total search steps (default: 20)
- `--num_drafts`: Initial implementations (default: 3)
- `--debug_prob`: Probability of debugging (default: 0.3)
- `--max_debug_depth`: Max consecutive debug attempts (default: 2)

**LLM Settings:**
- `--server_type`: google | openai | anthropic | deepseek | together | local
- `--model_name`: Specific model (uses preset if not specified)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Max generation length (default: 8192)

**Reasoning Models:**
- `--is_reasoning_model`: Enable reasoning model mode
- `--reasoning_effort`: low | medium | high (for o1/o3)
- `--budget_tokens`: Token budget for Claude thinking

## Search Strategy

The kernel agent uses AIDE's tree search with three operation types:

### 1. Draft (Initial Implementations)
- Generates `num_drafts` initial kernel implementations
- Focus on correctness over aggressive optimization
- Each draft explores different optimization strategies

### 2. Improve (Performance Optimization)
- Takes a **working** kernel and proposes a single optimization
- Atomic changes for experimental evaluation:
  - Operator fusion (e.g., matmul + relu → single kernel)
  - Memory optimization (shared memory, coalescing)
  - Algorithmic improvements (tiling, online algorithms)
  - Thread configuration tuning

### 3. Debug (Error Fixing)
- Fixes **compilation errors**:
  - CUDA syntax issues
  - PyTorch extension API misuse
  - Type mismatches
  
- Fixes **correctness errors**:
  - Logic bugs in kernel implementation
  - Memory access violations
  - Synchronization issues
  - Numerical precision problems

### Search Policy
```python
if num_draft_nodes < num_drafts:
    action = DRAFT  # Generate initial implementations
elif random() < debug_prob and has_buggy_leaf_nodes:
    action = DEBUG  # Fix a buggy implementation
elif has_good_nodes:
    action = IMPROVE  # Optimize best working kernel
else:
    action = DRAFT  # No good nodes, try new approach
```

## Evaluation Pipeline

Each kernel goes through three stages:

### Stage 1: Compilation
- Compiles CUDA code via `torch.utils.cpp_extension.load_inline()`
- **Pass**: Code compiles successfully
- **Fail**: Syntax errors, API misuse, compilation errors
- Feedback includes compiler error messages

### Stage 2: Correctness
- Runs kernel with multiple random inputs (`num_correct_trials`)
- Compares outputs against PyTorch reference using `torch.allclose()`
- **Pass**: All trials match within tolerance
- **Fail**: Output mismatch, runtime errors, or crashes
- Feedback includes error type, max/avg differences, trial results

### Stage 3: Performance (if correct)
- Measures runtime over `num_perf_trials` using CUDA events
- Records mean, std, min, max, and per-trial times
- **Metric**: Mean runtime in milliseconds (lower is better)

## Output Files

After running, the log directory contains:

```
logs/kernel_search/<exp-name>/
├── journal.json              # Complete solution tree
├── config.yaml              # Configuration used
├── reference_architecture.py # Original PyTorch model
├── best_kernel.py           # Best performing kernel
├── best_solution.py         # Same as best_kernel.py
├── tree_plot.html          # Interactive visualization
└── optimization_report.md   # LLM-generated report
```

### Visualization (`tree_plot.html`)
- Interactive tree showing all kernel attempts
- Nodes colored by success/failure
- Click nodes to see code, errors, and metrics
- Best solution highlighted in bold

## Integration with Existing KernelBench

The integration is **non-invasive** and works alongside existing KernelBench scripts:

- **Uses existing evaluation**: Calls `eval_kernel_against_ref()` directly
- **Compatible with all backends**: cuda, triton, tilelang, cute
- **Reuses prompt construction**: Leverages `prompt_constructor.py`
- **Shares dataset loading**: Uses `construct_kernelbench_dataset()`

You can still use the original single-shot generation script:
```bash
python scripts/generate_and_eval_single_sample.py \
    dataset_src="local" \
    level=1 \
    problem_id=1 \
    server_type=google
```

## Example Session

```bash
$ python run_kernel_search_simple.py --level 1 --problem_id 1 --steps 15

================================================================================
🚀 CUDA Kernel Optimization with AIDE Tree Search
================================================================================
Experiment: 0-kernel-search-matmul
Level 1, Problem 1
Backend: cuda, Precision: fp32
Search steps: 15
================================================================================

✓ Loaded problem: 1_Square_matrix_multiplication_.py

[Step 1/15] Drafting initial implementation...
  ✓ Compilation: PASSED
  ✓ Correctness: PASSED (5/5 trials)
  ✓ Performance: 2.453 ms

[Step 2/15] Drafting initial implementation...
  ✗ Compilation: FAILED
  CompilationError: invalid kernel launch configuration

[Step 3/15] Improving best kernel (2.453 ms)...
  ✓ Compilation: PASSED
  ✓ Correctness: PASSED (5/5 trials)
  ✓ Performance: 1.832 ms  [25% faster!]

...

================================================================================
🏁 Kernel Search Complete
================================================================================
Total nodes explored: 15
Successful kernels: 8
Failed kernels: 7

🏆 Best kernel performance: 1.127 ms
   Plan: Implement tiled matrix multiplication with 32x32 tiles...

✓ Best kernel saved to: logs/kernel_search/0-kernel-search-matmul/best_kernel.py

📊 Visualization: logs/kernel_search/0-kernel-search-matmul/tree_plot.html
📁 Full results: logs/kernel_search/0-kernel-search-matmul
================================================================================
```

## Key Design Decisions

1. **Feedback without LLM review**: Unlike AIDE's default which uses an LLM to review execution results, we directly map evaluation results to node properties. This is more reliable for structured kernel evaluation.

2. **Atomic improvements**: Each improvement step makes a single, focused change. This enables experimental attribution of performance changes.

3. **Error-type-specific debugging**: Compilation errors get different debugging prompts than correctness errors, with targeted guidance.

4. **Backend flexibility**: Supports CUDA, Triton, TileLang, and CuTe through the same interface.

5. **Precision awareness**: Handles fp32, fp16, and bf16 with appropriate tolerances for correctness checking.

## Extending the Integration

### Adding New Prompt Strategies

Edit `kernel_agent.py`:
```python
def _draft(self) -> Node:
    # Add few-shot examples
    prompt["Examples"] = self._get_few_shot_examples()
    
    # Add hardware-specific info
    prompt["Hardware"] = self._get_gpu_specs()
    
    return Node(plan=plan, code=code)
```

### Custom Search Policies

Override `search_policy()` in `KernelAgent`:
```python
def search_policy(self) -> Node | None:
    # Custom logic, e.g.:
    # - Prioritize recent failures
    # - Explore diverse optimization strategies
    # - Focus on specific performance bottlenecks
    return selected_node
```

### Additional Metrics

Extend `KernelExecResult` in `src/eval.py`:
```python
@dataclass
class KernelExecResult:
    compiled: bool
    correctness: bool
    runtime: float
    memory_usage: float  # Add new metric
    flops: float  # Add FLOPS measurement
```

Then update `kernel_interpreter.py` to expose these in the `ExecutionResult`.

## Troubleshooting

**Import errors**: Ensure you're running from the KernelBench root directory:
```bash
cd /scratch/sa6740/KernelBench
python run_kernel_search_simple.py ...
```

**CUDA out of memory**: Reduce batch sizes in eval or use smaller problems:
```bash
python run_kernel_search_simple.py --level 1 --problem_id 1  # Start small
```

**Compilation hangs**: Check for lock file issues. The interpreter handles these but you may need to clean up:
```bash
rm -rf workspaces/kernel_search/*/cuda_build/.lock*
```

**No successful kernels**: Try:
- Increase `--num_drafts` for more initial attempts
- Increase `--steps` for more search
- Use a stronger model (e.g., GPT-4 or Claude)
- Check the journal.json to see common error patterns

## Performance Notes

- **First compilation**: Takes longer due to CUDA compilation (~30-60s)
- **Subsequent runs**: Cached if code unchanged
- **Memory usage**: Each evaluation uses GPU memory; old allocations are cleared
- **Parallelization**: Not yet implemented; runs sequentially

## Future Enhancements

- [ ] Parallel evaluation of independent nodes
- [ ] Cached compilation across runs
- [ ] Multi-objective optimization (speed + memory)
- [ ] Automatic baseline comparison
- [ ] Integration with profiling tools (nsight, pytorch profiler)
- [ ] Support for multi-GPU kernels
- [ ] Curriculum learning (start simple, increase complexity)

## Citation

If you use this integration in research, please cite:

**AIDE:**
```
@article{...}  # Add AIDE paper citation
```

**KernelBench:**
```
@article{...}  # Add KernelBench paper citation
```
