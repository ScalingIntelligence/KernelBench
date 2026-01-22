#!/usr/bin/env python3
"""
check_hardware.py
Simple script to verify CUDA, PyTorch and whether the device looks like an H100.
"""
import sys
try:
    import torch
except Exception as e:
    print("ERROR: torch not importable:", e, file=sys.stderr)
    sys.exit(2)

def main():
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No CUDA device found. Abort.")
        sys.exit(1)

    count = torch.cuda.device_count()
    print("CUDA device count:", count)
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {name}")
        print(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
        # heuristic check for H100 in the name
        if "H100" in name.upper() or "HBM3" in name.upper():
            print("  -> Looks like an H100-series GPU (good).")
        else:
            print("  -> GPU name does not mention H100. If you require H100, verify this node.")

if __name__ == '__main__':
    main()

