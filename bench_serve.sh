#!/bin/bash

# Define parameter sweeps
REQUEST_RATES=("70" "130")

# Fixed parameters
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET="custom"
NUM_PROMPTS=2000
OUTPUT_LEN=1
DATASET_PATH="./synthetic_prompts_sharegpt.jsonl"

# Path to vLLM scheduler log
SCHEDULER_LOG_SRC="/home/jovyan/test_scheduler_trace.log"
mkdir -p "scheduler_logs"

# Create a folder to store logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Iterate over request rates
for rate in "${REQUEST_RATES[@]}"; do
    echo "============================================="
    echo "Running with request-rate=${rate}"
    echo "============================================="

    python create_synthetic_dataset.py
    
    # Define log filenames
    POWER_LOG="${LOG_DIR}/power_util_rate_${rate}.csv"
    PCIE_LOG="${LOG_DIR}/pcie_util_rate_${rate}.csv"
    SCHEDULER_LOG="${LOG_DIR}/scheduler_trace_rate_${rate}.log"

    # Start NVIDIA-SMI monitoring (run in background)
    echo "Starting NVIDIA-SMI monitors..."
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw \
        --format=csv --loop-ms=200 > "$POWER_LOG" &
    NSMI_PID1=$!

    nvidia-smi dmon -s t -d 1 --format=csv > "$PCIE_LOG" &
    NSMI_PID2=$!

    # Run vLLM benchmark
    vllm bench serve \
      --model "$MODEL" \
      --dataset-name "$DATASET" \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate "$rate" \
      --custom-output-len "$OUTPUT_LEN"

    # Kill NVIDIA-SMI monitors after run completes
    echo "Stopping NVIDIA-SMI monitors..."
    kill $NSMI_PID1 $NSMI_PID2

    # Move scheduler trace log if it exists
    if [ -f "$SCHEDULER_LOG_SRC" ]; then
        mv "$SCHEDULER_LOG_SRC" "$SCHEDULER_LOG"
        rm "$SCHEDULER_LOG_SRC"
        echo "Scheduler log moved to: $SCHEDULER_LOG"
    else
        echo "⚠️ Scheduler trace log not found at $SCHEDULER_LOG_SRC"
    fi

    echo "Logs saved to:"
    echo "  - $POWER_LOG"
    echo "  - $PCIE_LOG"
    echo "  - $SCHEDULER_LOG"
    echo "---------------------------------------------"
done

echo "✅ All benchmarks complete. Logs in $LOG_DIR/"
