#!/usr/bin/env python3
"""Compare Prometheus vLLM prefix-cache counters before/after one benchmark."""
import argparse
from pathlib import Path


def get_counter(path, metric):
    result = 0.0
    seen = False
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in (metric, metric + "_total"):
            continue
        try:
            result += float(line.rsplit(" ", 1)[-1])
            seen = True
        except ValueError:
            pass
    return result if seen else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--K", required=True, type=int)
    p.add_argument("--num-prompts", required=True, type=int)
    args = p.parse_args()
    fields = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
    deltas = {}
    for field in fields:
        before = get_counter(args.before, field)
        after = get_counter(args.after, field)
        if before is not None and after is not None:
            deltas[field] = after - before
            print(f"{field}: +{deltas[field]:.0f} tokens")
        else:
            print(f"{field}: not available on this vLLM /metrics endpoint")
    if fields[0] in deltas:
        hits = deltas[fields[0]]
        avg = hits / args.num_prompts
        print(f"Observed cached tokens/request: {avg:.1f}; intended K={args.K}")
        if args.K and avg < args.K * 0.95:
            print("WARNING: fewer hits than expected. Check prefix eviction, "
                  "tokenization, block size, and request concurrency.")
        if args.K and avg > args.K * 1.05:
            print("WARNING: more hits than expected. Check whether suffixes or "
                  "another benchmark's prompts were already cached.")


if __name__ == "__main__":
    main()
