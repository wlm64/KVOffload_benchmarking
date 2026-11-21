#!/usr/bin/env python3
"""Estimate a *system-level effective* overlap factor from K,T latency data.

Input CSV required columns: K,T,gpu_ms,io_ms,offloaded_ms.
Use comparable, independently measured transfer durations for io_ms;
compute overall per-request or per-iteration end-to-end durations for the others.
Run: python estimate_kv_overlap.py measurements.csv overlap_estimates.csv
"""
import csv
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python estimate_kv_overlap.py INPUT.csv OUTPUT.csv")
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        gpu, io, off = (float(r[field]) for field in
                        ("gpu_ms", "io_ms", "offloaded_ms"))
        exposed = off - gpu
        overlap = (gpu + io - off) / min(gpu, io) if min(gpu, io) > 0 else float('nan')
        r["exposed_io_plus_overhead_ms"] = f"{exposed:.5f}"
        r["effective_overlap_alpha_raw"] = f"{overlap:.5f}"
        r["ideal_no_overlap_ms"] = f"{gpu + io:.5f}"
        r["ideal_full_overlap_ms"] = f"{max(gpu, io):.5f}"
    if not rows:
        sys.exit("Input CSV has no data")
    with open(sys.argv[2], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {sys.argv[2]}")
    print("Values below 0 or above 1 flag extra overhead, contention, "
          "measurement mismatch, or other model violations; do not clamp silently.")


if __name__ == "__main__":
    main()
