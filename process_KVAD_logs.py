# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:21:09 2026

@author: Willie
"""

import os
import json
import glob
from collections import defaultdict
import numpy as np

# =========================
# Paths
# =========================
b200_dir = r"C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/data/B200_KVAD_logs_131K"
h200_dir = r"C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/data/H200_KVAD_logs"
h200_compute_dir = r"C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/data/H200_KVAD_compute_logs"

# =========================
# Helper: parse log
# =========================
def extract_duration(filepath):
    with open(filepath, "r") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Empty log file: {filepath}")

    # Parse JSON
    data = json.loads(content)

    if "duration" not in data:
        raise ValueError(f"No 'duration' field found in {filepath}")

    return data["duration"]

# =========================
# Parse B200 logs
# =========================
b200_data = defaultdict(dict)  # {run: {kt: duration}, full: duration}

for file in glob.glob(os.path.join(b200_dir, "*.log")):
    if "benchmark_results" in file:
        fname = os.path.basename(file)
        try:
            duration = extract_duration(file)
        except:
            import pdb; pdb.set_trace()
    
        if "full" in fname:
            # benchmark_results_run_1_sharegpt_full.log
            run = int(fname.split("_run_")[1].split("_")[0])
            b200_data[run]["full"] = duration
        else:
            # benchmark_results_run_1_KT_4.log
            parts = fname.split("_")
            run = int(parts[3])
            kt = int(parts[-1].replace(".log", ""))
            if kt == 4 and run == 1:
                pass
            else:
                b200_data[run][kt] = duration

# import pdb; pdb.set_trace()
# =========================
# Parse B200 logs
# =========================
h200_full_data = defaultdict(dict)  # {run: {kt: duration}, full: duration}

for file in glob.glob(os.path.join(h200_compute_dir, "*.log")):
    if "benchmark_results" in file:
        fname = os.path.basename(file)
        try:
            duration = extract_duration(file)
        except:
            import pdb; pdb.set_trace()
    
        if "full" in fname:
            # benchmark_results_run_1_sharegpt_full.log
            run = int(fname.split("_run_")[1].split("_")[0])
            h200_full_data[run]["full"] = duration
        else:
            # benchmark_results_run_1_KT_4.log
            parts = fname.split("_")
            run = int(parts[3])
            kt = int(parts[-1].replace(".log", ""))
    
            h200_full_data[run][kt] = duration

# =========================
# Parse H200 logs
# =========================
h200_data = defaultdict(dict)  # {run: {kt: duration}}

for file in glob.glob(os.path.join(h200_dir, "*.log")):
    if "benchmark_results" in file:
        fname = os.path.basename(file)
        duration = extract_duration(file)
    
        # benchmark_results_run_1_KT_4.log
        parts = fname.split("_")
        run = int(parts[3])
        kt = int(parts[-1].replace(".log", ""))
    
        h200_data[run][kt] = duration

# =========================
# Comparison
# =========================
print("\n===== Duration Comparison =====\n")


# =========================
# Compute average per KT
# =========================
kts = [4, 105]
b200_cost = 114
h200_cost = 63
for kt in kts:
    h200_avg = np.mean([h200_data[run][kt] for run in h200_data])
    # b200_avg = np.mean([b200_data[run][kt] for run in b200_data])
    full_avg = np.mean([b200_data[run]['full'] for run in b200_data])
    h200_full_avg = np.mean([h200_full_data[run]['full'] for run in h200_full_data])
    h200_compute_avg =  np.mean([h200_full_data[run][kt] for run in h200_full_data])
    
    if kt == 4:
        b200_avg = np.mean([b200_data[run][kt] for run in b200_data if run != 1])
    else:
        b200_avg = np.mean([b200_data[run][kt] for run in b200_data])
        
    # h200_avg = np.mean([h200_data[run][kt] for run in h200_data if run != 1])
    # b200_avg = np.mean([b200_data[run][kt] for run in b200_data if run != 1])
    # full_avg = np.mean([b200_data[run]['full'] for run in b200_data if run != 1])
    # h200_full_avg = np.mean([h200_full_data[run]['full'] for run in h200_full_data])
    # h200_compute_avg = np.mean([h200_full_data[run][kt] for run in h200_full_data])

    combined = h200_avg + b200_avg
    
    h2_cost = h200_avg * h200_cost
    b2_cost = b200_avg * b200_cost
    b2_full_cost = full_avg * b200_cost
    h200_full_cost =  h200_full_avg * h200_cost
    KVAD_cost = h2_cost + b2_cost

    print(f"KT={kt}")
    print(f"  H200 avg: {h200_avg:.3f}")
    print(f"  B200 avg: {b200_avg:.3f}")
    print(f"  H200 compute avg: {h200_compute_avg:.3f}")
    print(f"  Sum H200+B200: {combined:.3f}")
    print(f"  B200 full avg: {full_avg:.3f}")
    print(f"  H200 Full avg: {h200_full_avg:.3f}")
    print(f"  Ratio (full/combined): {full_avg/combined:.3f}")
    print(f"  Cost Ratio: {KVAD_cost/b2_full_cost:.3f}")
    print(f"  Cost Ratio H200: {KVAD_cost/h200_full_cost:.3f}")
    import pdb; pdb.set_trace()

import re

json_list = ["C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/synthetic_prompts/synthetic_prompts_sharegpt_KT_4_compute.jsonl",
             "C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/synthetic_prompts/synthetic_prompts_sharegpt_KT_4_memory.jsonl",
             "C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/synthetic_prompts/synthetic_prompts_sharegpt_KT_105_compute.jsonl",
             "C:/Users/Willie/Documents/GitHub/KVOffload_benchmarking/synthetic_prompts/synthetic_prompts_sharegpt_KT_105_memory.jsonl",]

kt_counts = {}

for path in json_list:
    # Extract KT value
    kt = int(re.search(r"KT_(\d+)", path).group(1))
    
    # Determine type
    kind = "compute" if "compute" in path else "memory"
    
    # Count lines (entries)
    with open(path, "r") as f:
        count = sum(1 for _ in f)
    
    # Store
    if kt not in kt_counts:
        kt_counts[kt] = {}
    kt_counts[kt][kind] = count

print(kt_counts)

import matplotlib.pyplot as plt
import numpy as np

kts = sorted(kt_counts.keys())

perf_ratios = []
cost_ratios = []
compute_fracs = []
memory_fracs = []

for kt in kts:
    # recompute from your earlier logic
    h200_avg = np.mean([h200_data[run][kt] for run in h200_data])
    if kt == 4:
        b200_avg = np.mean([b200_data[run][kt] for run in b200_data if run != 1])
    else:
        b200_avg = np.mean([b200_data[run][kt] for run in b200_data])
    full_avg = np.mean([b200_data[run]['full'] for run in b200_data])

    combined = h200_avg + b200_avg

    h2_cost = h200_avg * h200_cost
    b2_cost = b200_avg * b200_cost
    b2_full_cost = full_avg * b200_cost
    KVAD_cost = h2_cost + b2_cost

    perf_ratios.append(full_avg / combined)
    cost_ratios.append(KVAD_cost / b2_full_cost)

    # request mix
    compute = kt_counts[kt]["compute"]
    memory = kt_counts[kt]["memory"]
    total = compute + memory

    compute_fracs.append(compute / total)
    memory_fracs.append(memory / total)

perf_ratios.append(full_avg / h200_full_avg)
cost_ratios.append(h200_full_cost / b2_full_cost)
kts.append("H200")
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
})

# =========================
# Plot 1: Performance & Cost Ratios
# =========================
x = np.arange(len(kts))
width = 0.35

# Mapping KT values to LaTeX labels
xtick_labels = {4: r"$K_{CM}$", 105: r"$K_{ISO}$", "H200": "H200"}

# =========================
# Plot 1: Performance & Cost Ratios
# =========================
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
bars1 = ax.bar(x - width/2, perf_ratios, width, label="Throughput")
bars2 = ax.bar(x + width/2, cost_ratios, width, label="Cost")

# import pdb; pdb.set_trace()
# Add numbers on top
for bar in bars1:
    height = bar.get_height()
    label = f"{height:.2f}".lstrip("0")
    ax.text(bar.get_x() + bar.get_width()/2, height/2, label, ha="center", va="center", color="black", fontweight='bold', fontsize = 18)
for bar in bars2:
    height = bar.get_height()
    label = f"{height:.2f}".lstrip("0")
    ax.text(bar.get_x() + bar.get_width()/2, height/2, label, ha="center", va="center", color="black", fontweight='bold', fontsize = 18)

ax.set_xticks(x)
ax.set_xticklabels([xtick_labels[kt] for kt in kts])
ax.set_ylabel("Normalized Ratio")
#ax.set_title("Performance and Cost Ratios")
ax.legend()
ax.set_ylim(0, max(max(perf_ratios), max(cost_ratios)) * 1.2)

plt.tight_layout()
plt.savefig('./KVAD_throughput_cost.png')
plt.show()

# =========================
# Plot 2: Compute & Memory Fraction (green/red)
# =========================
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
bars1 = ax.bar(x - width/2, compute_fracs, width, label="Compute", color='green')
bars2 = ax.bar(x + width/2, memory_fracs, width, label="Memory", color='red')

# Add numbers on top
for bar in bars1:
    height = bar.get_height()
    label = f"{height:.2f}".lstrip("0")
    ax.text(bar.get_x() + bar.get_width()/2, height/2, label, ha="center", va="center", color="black", fontweight='bold', fontsize = 20)
for bar in bars2:
    height = bar.get_height()
    label = f"{height:.2f}".lstrip("0")
    ax.text(bar.get_x() + bar.get_width()/2, height/2, label, ha="center", va="center", color="black", fontweight='bold', fontsize = 20)

ax.set_xticks(x)
kts = sorted(kt_counts.keys())
ax.set_xticklabels([xtick_labels[kt] for kt in kts])
ax.set_ylabel("Fraction")
#ax.set_title("Compute vs Memory Request Fraction")
ax.legend()
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig('./KVAD_request_split.png')
plt.show()