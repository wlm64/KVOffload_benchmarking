#!/usr/bin/env bash
set -euo pipefail

# Start vLLM *separately* using --enable-prefix-caching and no LMCache connector.
# Example: vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
#   --tensor-parallel-size 8 --enable-prefix-caching --port 8000

MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
DATASET_ROOT="${DATASET_ROOT:-offload_datasets}"
LOG_ROOT="${LOG_ROOT:-kv_no_offload_warm}"
# Override via Bash arrays below for your model's max context and GPU capacity.
context_tokens=(0 1024 2048 4096 8192 16384 32768 65536)
prefill_tokens=(64 128 256 512 1024 2048)

for K in "${context_tokens[@]}"; do
  for T in "${prefill_tokens[@]}"; do
    echo "----- K=${K} cached, T=${T} new -----"
    python create_offload_datasets_exact.py \
      --model "$MODEL" --context-tokens "$K" --prefill-tokens "$T" \
      --num-prompts "$NUM_PROMPTS" --block-size "$BLOCK_SIZE" \
      --output-root "$DATASET_ROOT"

    PREFIX_FILE="${DATASET_ROOT}/${MODEL}/synthetic_prompts_${K}_${T}.prefix.txt"
    DATASET_FILE="${DATASET_ROOT}/${MODEL}/synthetic_prompts_${K}_${T}.jsonl"
    LOG_DIR="${LOG_ROOT}/${MODEL}"
    mkdir -p "$LOG_DIR"

    # Warm the prefix after generation. If K=0, this is a no-op.
    python warm_kv_prefix.py --model "$MODEL" --prefix-file "$PREFIX_FILE" \
      --base-url "$BASE_URL" --api-key "$API_KEY" \
      --expected-prefix-tokens "$K"

    # Metrics snapshots allow you to verify that ~K tokens/request were
    # actually reused from the HBM prefix cache, not accidentally recomputed.
    curl -fsS "${BASE_URL}/metrics" > "${LOG_DIR}/metrics_${K}_${T}_before.txt" || true

    vllm bench serve \
      --model "$MODEL" --base-url "$BASE_URL" --backend openai \
      --dataset-name custom --dataset-path "$DATASET_FILE" \
      --num-prompts "$NUM_PROMPTS" --custom-output-len 1 \
      --request-rate inf --disable-shuffle \
      --percentile-metrics ttft --metric-percentiles 50,95,99 \
      --result-dir "$LOG_DIR" \
      --result-filename "benchmark_results_${K}_${T}.json" \
      --save-result --save-detailed

    curl -fsS "${BASE_URL}/metrics" > "${LOG_DIR}/metrics_${K}_${T}_after.txt" || true
    python summarize_prefix_hits.py \
      --before "${LOG_DIR}/metrics_${K}_${T}_before.txt" \
      --after "${LOG_DIR}/metrics_${K}_${T}_after.txt" \
      --K "$K" --num-prompts "$NUM_PROMPTS" \
      | tee "${LOG_DIR}/prefix_hit_report_${K}_${T}.txt"
  done
done

echo "Completed warm-GPU (no LMCache) benchmark."
