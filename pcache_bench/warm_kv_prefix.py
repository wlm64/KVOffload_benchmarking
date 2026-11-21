#!/usr/bin/env python3
"""Populate vLLM's HBM prefix cache by completing a common prefix once."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prefix-file", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--expected-prefix-tokens", type=int)
    args = p.parse_args()
    prefix = Path(args.prefix_file).read_text(encoding="utf-8")
    if not prefix:
        print("K=0: no prefix to prime")
        return
    payload = {"model": args.model, "prompt": prefix,
               "max_tokens": 1, "temperature": 0, "stream": False, "add_special_tokens": False,}
    request = Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + args.api_key},
        method="POST",
    )
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    usage = result.get("usage", {})
    print("Primed KV cache for model:", args.model)
    actual = usage.get("prompt_tokens")
    print("Server prompt tokens:", actual if actual is not None else "unavailable")
    if args.expected_prefix_tokens is not None and actual is not None:
        if actual != args.expected_prefix_tokens:
            raise RuntimeError(
                f"Expected K={args.expected_prefix_tokens}, but server tokenized "
                f"the warmup prefix into {actual} tokens. Fix tokenizer/template settings."
            )
    print("Server completion tokens:", usage.get("completion_tokens", "unavailable"))
    print("Prefix warmup is excluded from benchmark metrics.")


if __name__ == "__main__":
    main()
