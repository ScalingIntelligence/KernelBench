import os
import sys
import re

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.dataset import construct_kernelbench_dataset
from src.utils import read_file

# Template that ensures proper ordering
TEMPLATE = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# SHINKA_EVOLVE_TEMPLATE
# This file is auto-generated to seed the evolution.

# EVOLVE-BLOCK-START
import torch
import torch.nn as nn

# --- INSERT CUSTOM CUDA KERNEL HERE ---
# (Define source strings and call load_inline BEFORE the class definition)

{content}

def get_inputs():
    # Helper to generate inputs on CUDA
    return [x.cuda() for x in {get_inputs_name}()]

def get_init_inputs():
    return {get_init_inputs_name}()
# EVOLVE-BLOCK-END
"""

def create_seed_file(level, problem_id, output_path):
    dataset = construct_kernelbench_dataset(level)
    # KernelBench uses 0-indexed list, problem_id is 1-indexed
    ref_path = dataset[problem_id - 1]
    ref_src = read_file(ref_path)
    
    # 1. Rename Class
    content = re.sub(r'class Model\s*\(', 'class ModelNew(', ref_src)
    
    # 2. Fix super() call to be generic (handles name change)
    content = re.sub(r'super\s*\([^\)]+\)\.__init__\(\)', 'super().__init__()', content)

    # 3. Extract just the class and the original helper functions
    # We want to wrap the helper functions to ensure .cuda()
    # (Simplified regex extraction - assumes standard KernelBench format)
    
    # Actually, the simplest way is to just use the content as is, 
    # but rename the helpers in the content so we can wrap them in the TEMPLATE
    content = content.replace("def get_inputs():", "def _original_get_inputs():")
    content = content.replace("def get_init_inputs():", "def _original_get_init_inputs():")
    
    seed_content = TEMPLATE.format(
        content=content,
        get_inputs_name="_original_get_inputs",
        get_init_inputs_name="_original_get_init_inputs"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(seed_content)
    
    print(f"Created seed file at {output_path}")

if __name__ == "__main__":
    # Example usage
    create_seed_file(1, 1, "runs/debug_seed/initial.py")