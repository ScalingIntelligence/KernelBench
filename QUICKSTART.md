# Quick Start Guide: AIDE + KernelBench Integration

## 🚀 Run Your First Kernel Search in 60 Seconds

### Prerequisites
- CUDA-capable GPU
- Conda environment `kernel-bench` activated
- API key for LLM provider (Google, OpenAI, etc.)

### Step 1: Activate Environment
```bash
cd /scratch/sa6740/KernelBench
conda activate kernel-bench
```

### Step 2: Set API Key (if using Google Gemini)
```bash
# Add to .env file or export
export GOOGLE_API_KEY="your-api-key-here"
```

### Step 3: Run a Simple Search
```bash
# Level 1, Problem 1 (Square Matrix Multiplication), 5 search steps
python run_kernel_search_simple.py \
    --level 1 \
    --problem_id 1 \
    --steps 5 \
    --server_type google
```

### Step 4: View Results
```bash
# Check the log directory
ls logs/kernel_search/

# Open the visualization in a browser
firefox logs/kernel_search/0-*/tree_plot.html

# View the best kernel
cat logs/kernel_search/0-*/best_kernel.py
```

---

## 📊 What Happens During the Search?

```
Step 1/5: Drafting initial implementation...
  ✓ Compilation: PASSED
  ✓ Correctness: PASSED (5/5 trials)
  ✓ Performance: 2.453 ms

Step 2/5: Drafting initial implementation...
  ✗ Compilation: FAILED
  CompilationError: invalid kernel launch configuration

Step 3/5: Debugging failed kernel...
  ✓ Compilation: PASSED
  ✓ Correctness: PASSED (5/5 trials)
  ✓ Performance: 2.201 ms

Step 4/5: Improving best kernel (2.201 ms)...
  ✓ Compilation: PASSED
  ✓ Correctness: PASSED (5/5 trials)
  ✓ Performance: 1.832 ms  [17% faster!]

Step 5/5: Improving best kernel (1.832 ms)...
  ✓ Compilation: PASSED
  ✗ Correctness: FAILED
  Output mismatch detected

🏆 Best kernel: 1.832 ms
```

---

## 🎯 Try Different Problems

### Easy (Level 1):
```bash
# ReLU activation
python run_kernel_search_simple.py --level 1 --problem_id 19 --steps 10

# Sigmoid activation  
python run_kernel_search_simple.py --level 1 --problem_id 21 --steps 10

# Element-wise operations
python run_kernel_search_simple.py --level 1 --problem_id 37 --steps 10
```

### Medium (Level 2):
```bash
# Fused operations
python run_kernel_search_simple.py --level 2 --problem_id 1 --steps 15
```

---

## ⚙️ Common Options

```bash
python run_kernel_search_simple.py \
    --level 1              # KernelBench level (1-4)
    --problem_id 1         # Problem number
    --steps 20             # Number of search iterations
    --num_drafts 3         # Initial implementations
    --backend cuda         # cuda | triton | tilelang | cute
    --precision fp32       # fp32 | fp16 | bf16
    --server_type google   # LLM provider
    --temperature 0.7      # Sampling temperature
    --num_correct_trials 5 # Correctness validation runs
    --num_perf_trials 100  # Performance measurement runs
```

---

## 📂 Output Files

After a search, check `logs/kernel_search/<exp-name>/`:

- **`best_kernel.py`** - Best performing kernel code
- **`tree_plot.html`** - Interactive tree visualization
- **`journal.json`** - Complete search history
- **`config.yaml`** - Configuration used
- **`optimization_report.md`** - LLM-generated summary

---

## 🐛 Quick Troubleshooting

### "No module named 'omegaconf'"
```bash
conda activate kernel-bench
pip install omegaconf coolname igraph humanize black genson dataclasses-json shutup
```

### "CUDA out of memory"
```bash
# Try a simpler problem or reduce batch size
python run_kernel_search_simple.py --level 1 --problem_id 1 --steps 5
```

### "Import Error"
```bash
# Make sure you're in the KernelBench directory
cd /scratch/sa6740/KernelBench
python run_kernel_search_simple.py ...
```

---

## 📚 Learn More

- **Full Documentation**: `KERNEL_SEARCH_README.md`
- **Implementation Details**: `INTEGRATION_SUMMARY.md`
- **Test Suite**: Run `python test_integration.py`

---

## 💡 Pro Tips

1. **Start small**: Use `--steps 5-10` for initial experiments
2. **Use reasoning models**: Add `--is_reasoning_model` for o1/o3
3. **Check the tree**: Open `tree_plot.html` to see all attempts
4. **Learn from failures**: Review `journal.json` to see error patterns
5. **Iterate on config**: Edit `kernel_config.yaml` for complex setups

---

## 🎉 You're Ready!

Now you can optimize CUDA kernels using automated search with LLM guidance. The system will:
- Generate multiple implementations
- Test for correctness
- Measure performance
- Learn from failures
- Iteratively improve

Happy kernel optimization! 🚀
