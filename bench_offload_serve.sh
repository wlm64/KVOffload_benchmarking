#!/bin/bash

# Define parameter sweeps
models=("Qwen/Qwen3-235B-A22B-Instruct-2507") # "Qwen/Qwen3-235B-A22B-Instruct-2507" "deepseek-ai/DeepSeek-V2")
context_tokens=("128" "1024" "2048" "4096" "8192" "12800" "16384" "32768" "65536" "131072")
prefill_tokens=("128" "512")
DATASET="custom"
NUM_PROMPTS=200
OUTPUT_LEN=1

# Iterate over request rates
for MODEL in "${models[@]}"; do
    # Base log directory
    LOG_DIR="kv_offload_serve/${MODEL}/"

    for context in "${context_tokens[@]}"; do
        for prefill in "${prefill_tokens[@]}"; do
            # Run benchmark
            vllm bench serve \
                --model "$MODEL" \
                --dataset-name "$DATASET" \
                --dataset-path "offload_datasets/Qwen/Qwen3-235B-A22B-Instruct-2507/synthetic_prompts_${context}_${prefill}.jsonl" \
                --num-prompts "$NUM_PROMPTS" \
                --result-dir "${LOG_DIR}" \
                --result-filename "benchmark_results_rate_${context}_${prefill}.log" \
                --custom-output-len "$OUTPUT_LEN" \
                --percentile-metrics "ttft" \
                --metric-percentiles ".1,1,2.3,15.9,50,84.1,97.7,99,.1" \
                --save-result
                #
        done
    done
done

echo "✅ All benchmarks complete"
