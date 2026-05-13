#!/bin/bash

# Define parameter sweeps
REQUEST_RATES=("inf")
DATASETS=("sharegpt") 
RUNS=(1 2 3 4 5)

# Fixed parameters
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
DATASET="custom"
NUM_PROMPTS=5000
OUTPUT_LEN=1

# Path to vLLM scheduler log
SCHEDULER_LOG_SRC="/home/ec2-user/test_scheduler_trace.log"

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for run in "${RUNS[@]}"; do
    echo "============================================="
    echo "🔁 Starting RUN $run"
    echo "============================================="

    for PROMPT_DATASET in "${DATASETS[@]}"; do
        DATASET_PATH="./synthetic_prompts/synthetic_prompts_${PROMPT_DATASET}.jsonl"

        for rate in "${REQUEST_RATES[@]}"; do
            echo "---------------------------------------------"
            echo "Run=${run} | rate=${rate}"
            echo "---------------------------------------------"

            # Log filenames (include run)
            POWER_LOG="${LOG_DIR}/power_util_run_${run}_rate_${rate}_${PROMPT_DATASET}_full.csv"
            PCIE_LOG="${LOG_DIR}/pcie_util_run_${run}_rate_${rate}_${PROMPT_DATASET}_full.csv"
            SCHEDULER_LOG="${LOG_DIR}/scheduler_trace_run_${run}_rate_${rate}_${PROMPT_DATASET}_full.log"
            RESULT_LOG="${LOG_DIR}/benchmark_results_run_${run}_${PROMPT_DATASET}_full.log"

            # Start NVIDIA-SMI monitoring
            echo "Starting NVIDIA-SMI monitors..."
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw \
                --format=csv --loop-ms=200 > "$POWER_LOG" &
            NSMI_PID1=$!

            nvidia-smi dmon -s t -d 1 --format=csv > "$PCIE_LOG" &
            NSMI_PID2=$!

            # Run benchmark
            vllm bench serve \
            --model "$MODEL" \
            --dataset-name "$DATASET" \
            --dataset-path "$DATASET_PATH" \
            --num-prompts "$NUM_PROMPTS" \
            --no-oversample \
            --request-rate "$rate" \
            --custom-output-len "$OUTPUT_LEN" \
            --result-dir "${LOG_DIR}" \
            --result-filename "$(basename "$RESULT_LOG")" \
            --save-result

            # Stop monitors
            echo "Stopping NVIDIA-SMI monitors..."
            kill $NSMI_PID1 $NSMI_PID2

            # Move scheduler trace log
            if [ -f "$SCHEDULER_LOG_SRC" ]; then
                mv "$SCHEDULER_LOG_SRC" "$SCHEDULER_LOG"
                echo "Scheduler log moved to: $SCHEDULER_LOG"
            else
                echo "⚠️ Scheduler trace log not found"
            fi

            echo "Logs saved:"
            echo "  - $POWER_LOG"
            echo "  - $PCIE_LOG"
            echo "  - $SCHEDULER_LOG"
            echo "  - $RESULT_LOG"
        done
    done

    # Optional: cooldown between runs
    echo "⏳ Cooling down before next run..."
    sleep 5
done

echo "✅ All benchmarks complete. Logs in $LOG_DIR/"