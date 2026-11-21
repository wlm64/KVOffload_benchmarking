# Warm-resident vLLM KV-cache prefill benchmark

This bundle implements the no-offload control for requests of the form
`<K-token cached prefix><T new prefill tokens>`.

Files:
- `create_offload_datasets_exact.py`: generate `.jsonl` with an exact K-token common prefix, T-token distinct suffixes, and a companion `.prefix.txt` file.
- `warm_kv_prefix.py`: submit that prefix once via `/v1/completions`, checking the server-reported prompt token count.
- `bench_no_offload_serve_warm.sh`: sequential single-request benchmark sweep, separate prefix warmup per K/T, and metrics snapshots.
- `summarize_prefix_hits.py`: report observed GPU-prefix-cache reuse for each sweep point.
- `estimate_kv_overlap.py`: process measured latency data into an *effective* overlap factor (see caveats below).

## Requirements

The scripts are intended for the **same environment as vLLM** and need `transformers`, `vllm`, `curl`, and Python 3.8+. Start a serving engine separately **without** any LMCache KV connector, with `--enable-prefix-caching`, for example:

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 8 --enable-prefix-caching --port 8000
```

Use a `--max-model-len` setting and GPU KV budget that can handle `max(K+T)` and K respectively. When switching model, match `MODEL` in the benchmark shell script or export it.

Place all bundle scripts together and run from that directory:

```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  NUM_PROMPTS=200 bash bench_no_offload_serve_warm.sh
```

Edit the `context_tokens` and `prefill_tokens` Bash arrays as needed. The custom JSONL and warmup prefix will be generated under `offload_datasets/<MODEL>/`.

The scripts use raw `/v1/completions`, not `/v1/chat/completions`, to keep the shared prefix stable and prevent unexpected chat-template tokens. Each full prompt is re-tokenized with `add_special_tokens=False` to check length and common prefix. The warmup script additionally checks server-reported prompt tokens, failing if special tokens have changed the intended K. Ensure the benchmark and server use the same tokenizer/configuration.

The vLLM `/metrics` snapshots are compared using `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries`. The desired per-request cached tokens is approximately K (subject to full-block granularity); see `prefix_hit_report_*.txt`. If the metrics endpoint or counter is unavailable, the report indicates it rather than claiming a hit.

## Experimental cautions

- Use `--max-concurrency 1` for *isolated* warm-prefill latencies. Run another sweep with higher concurrency when studying batching and *inter-request* transfer/computation overlap.
- K should be a multiple of the configured vLLM KV block size (default 16 here). Keep the models, TP size, dtype, max batched tokens and other engine settings matched across no-offload/offload experiments.
- In the *offloaded* condition, verify K was loaded from **CPU**, rather than served from the GPU prefix cache. Simply reusing the same prompt with LMCache configured can still produce HBM hits. Force HBM eviction or isolate L0 state, while retaining K in LMCache L1; verify tier-sourced cached tokens and transfer bytes.
- Raw TTFT includes serving overhead, queueing and first-token output, so it is not equivalent to a CUDA kernel runtime. Use GPU traces for fine-grained overlap.
- Offload path can also run GPU-to-CPU stores in the background; distinguish stores from CPU-to-GPU retrieval in measurements.

## Effective overlap analysis

Prepare `measurements.csv` with columns `K,T,gpu_ms,io_ms,offloaded_ms` where `gpu_ms` is warm-resident wall time, `io_ms` is measured **isolated** transfer duration (on the comparable boundary), and `offloaded_ms` is the offloaded wall time:

```bash
python estimate_kv_overlap.py measurements.csv overlap_estimates.csv
```

The script calculates `alpha = (gpu_ms + io_ms - offloaded_ms) / min(gpu_ms,io_ms)`.

**Do not interpret alpha as pure kernel/DMA execution overlap without a timeline:** extra scheduling and synchronization overhead, HBM/SM contention, and measurement mismatches also affect the result. alpha outside [0,1] is preserved rather than clipped.

Use NVIDIA Nsight Systems to measure actual H2D DMA intervals overlapping with inference kernels. For the MP connector, LMCache's `lmcache_mp.transfer_phase_bytes/elapsed` counters and `lmcache_mp.transfer_staging_throughput` describe the stages, but do not by themselves reveal their concurrent execution against inference.
