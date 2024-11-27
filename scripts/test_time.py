import torch
import numpy as np
from src import eval

"""
Test various timing methodology
"""

device = torch.device("cuda:5")
# curr_device = torch.cuda.current_device()

x = torch.randn(1000, 128 * 128).cuda(device=device)
y = torch.randn(128 * 128, 1000).cuda(device=device)

num_trials = 10
elapsed_times = eval.time_execution_with_cuda_event(
    torch.matmul, x, y, verbose=True, device=device
)
stats = eval.get_timing_stats(elapsed_times)

print(stats)
