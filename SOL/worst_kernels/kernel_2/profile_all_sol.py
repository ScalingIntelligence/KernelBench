#!/bin/bash

# Directory where KernelBench levels are located
SOURCE_DIR="/root/KernelBench/level1"
OUTPUT_DIR="./ncu_reports"
mkdir -p $OUTPUT_DIR

echo "Starting NCU Speed of Light Profiling on H100..."

# Loop through the first 100 levels (adjust glob if needed)
for script in $SOURCE_DIR/[1-9][0-9]_*.py $SOURCE_DIR/[1-9]_*.py; do
    filename=$(basename "$script")
    echo "Profiling $filename..."
    
    # Run NCU and output to CSV
    # sm__throughput.avg.pct_of_peak_sustained_elapsed = Compute SOL
    # gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed = Memory SOL
    ncu --csv --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-count 1 \
    python3 "$script" > "$OUTPUT_DIR/${filename%.py}.csv" 2>/dev/null
done

echo "Profiling complete. Results saved in $OUTPUT_DIR"

