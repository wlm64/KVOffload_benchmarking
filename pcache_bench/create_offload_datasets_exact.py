#!/usr/bin/env python3
"""Create vLLM custom .jsonl prompts with an exactly reusable K-token prefix.

Example:
 python create_offload_datasets_exact.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
   --context-tokens 8192 --prefill-tokens 512 --num-prompts 200

The output directory gets a dataset JSONL, a prefix.txt for priming, and metadata.
Uses add_special_tokens=False: confirm that serving and benchmarking tokenize the
raw completion prompts consistently (Qwen3 completions normally do).
"""
import argparse
import json
import random
from pathlib import Path
from transformers import AutoTokenizer

CANDIDATE_PARTS = (
    " the", " cat", " dog", " tree", " river", " north", " red", " blue",
    " apple", " stone", " cloud", " garden", " paper", " orange", " green",
    " window", " moon", " star", " lake", " light", " music", " road",
    " house", " flower", " bird", " mountain", " book", " sky",
)


def token_ids(tok, value):
    return tok.encode(value, add_special_tokens=False)


def stable_parts(tok):
    """Pick text fragments behaving like exactly one token per repetition."""
    good = []
    for part in CANDIDATE_PARTS:
        if len(token_ids(tok, part)) == 1 and len(token_ids(tok, part * 32)) == 32:
            good.append(part)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-tokens", type=int, required=True, help="K, resident cached tokens")
    ap.add_argument("--prefill-tokens", type=int, required=True, help="T, new tokens")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default="offload_datasets")
    args = ap.parse_args()
    K, T = args.context_tokens, args.prefill_tokens
    if K < 0 or T < 1 or args.num_prompts < 1:
        ap.error("K must be >= 0, T and num-prompts must be >= 1")
    if K and K % args.block_size:
        ap.error(f"K={K} is not a multiple of vLLM block size {args.block_size}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    parts = stable_parts(tok)
    if len(parts) < 2:
        raise RuntimeError("Could not find >=2 stable one-token text fragments for this tokenizer")

    # Prefer a uniform prefix: stable and easy to reproduce in warmup.
    prefix = None
    for unit in parts:
        possible = unit * K
        if len(token_ids(tok, possible)) == K:
            prefix = possible
            break
    if prefix is None:
        raise RuntimeError("Could not construct an exact-K token prefix")
    prefix_ids = token_ids(tok, prefix)
    rng = random.Random(args.seed + 1000003 * K + T)
    outdir = Path(args.output_root) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    basename = f"synthetic_prompts_{K}_{T}"
    dataset_path = outdir / f"{basename}.jsonl"
    prefix_path = outdir / f"{basename}.prefix.txt"
    meta_path = outdir / f"{basename}.meta.json"

    # Test combinations with this prefix; some one-token fragments merge when
    # adjacent. Retain only fragments that work in all short boundary tests.
    valid = []
    for part in parts:
        probe = prefix + part * 32
        probe_ids = token_ids(tok, probe)
        if len(probe_ids) == K + 32 and probe_ids[:K] == prefix_ids:
            valid.append(part)
    if len(valid) < 2:
        raise RuntimeError("No stable suffix alphabet at the prefix boundary")

    fingerprints = set()
    with dataset_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_prompts):
            for attempt in range(100):
                # A different first block stops suffix-prefix reuse across requests.
                units = [rng.choice(valid) for _ in range(T)]
                if T >= args.block_size:
                    sig = tuple(units[:args.block_size])
                    if sig in fingerprints:
                        continue
                prompt = prefix + "".join(units)
                ids = token_ids(tok, prompt)
                if len(ids) != K + T or ids[:K] != prefix_ids:
                    continue
                if T >= args.block_size:
                    fingerprints.add(sig)
                f.write(json.dumps({"prompt": prompt, "output_tokens": 1}) + "\n")
                break
            else:
                raise RuntimeError(f"Failed exact-token validation for sample {i}; "
                                   "try a different tokenizer/text alphabet")

    prefix_path.write_text(prefix, encoding="utf-8")
    meta = {"model": args.model, "K": K, "T": T,
            "num_prompts": args.num_prompts, "block_size": args.block_size,
            "prefix_tokens": len(prefix_ids), "dataset": str(dataset_path),
            "prefix_path": str(prefix_path), "seed": args.seed,
            "tokenizer_add_special_tokens": False}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # vLLM's serving tokenizer may add special tokens in some configurations.
    if K and len(tok.encode(prefix, add_special_tokens=True)) != K:
        print("WARNING: model tokenizer inserts special tokens with default settings; "
              "confirm the raw completions endpoint tokenization and effective K.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
