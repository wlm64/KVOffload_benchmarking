# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:24:17 2026

@author: Willie
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
root_dirs = ["./data/kv_offload_B200", "./data/kv_offload_H200",]
output_dir = f"./paper_plots"
os.makedirs(output_dir, exist_ok=True)

# Font settings
plt.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
})

all_records = []

for root_dir in root_dirs:
    gpu_type = "H200" if "H200" in root_dir else "B200"
    
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
    
        subfolders = [f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))]
        if not subfolders:
            continue
        
        model_folder = os.path.join(folder_path, subfolders[0])
    
        log_files = [f for f in os.listdir(model_folder)
                     if f.startswith("benchmark_results") and f.endswith(".log")]
        if not log_files:
            continue
    
        for log_file in log_files:
            kv_match = [int(s) for s in log_file.replace(".log","").split("_") if s.isdigit()]
            if len(kv_match) != 2:
                continue
            
            kv_size, prefill = kv_match
            log_path = os.path.join(model_folder, log_file)
            
            with open(log_path, "r") as f:
                data = pd.read_json(f.read(), typ="series")
                
                all_records.append({
                    "GPU": gpu_type,
                    "KV": kv_size,
                    "Prefill": prefill,
                    "duration": data.get("duration", np.nan),
                    "mean_ttft": data.get("mean_ttft_ms", np.nan),
                    "std_ttft": data.get("std_ttft_ms", np.nan)
                })

df = pd.DataFrame(all_records)

# Split
df_h200 = df[df["GPU"] == "H200"].copy()
df_b200 = df[df["GPU"] == "B200"].copy()

merged = pd.merge(
    df_h200,
    df_b200,
    on=["KV", "Prefill"],
    suffixes=("_h200", "_b200")
)

# Avoid divide-by-zero
merged = merged[merged["duration_b200"] > 0]

merged["rel_duration"] = merged["duration_h200"] / merged["duration_b200"]
merged["kv_ratio"] = merged["KV"] / merged["Prefill"]

merged.to_csv('./data/b200_h100_merged.csv')
for prefill in [128, 512]:
    sub = merged[merged["Prefill"] == prefill].copy()
    sub = sub.sort_values("kv_ratio")
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(sub["kv_ratio"], sub["rel_duration"], marker='o')
    
    plt.xlabel("KV / Prefill")
    plt.ylabel("Relative Duration (H@00 / B200)")
    plt.title(f"Relative Duration vs KV/Prefill (Prefill={prefill})")
    
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f"rel_duration_prefill_{prefill}.png"),
                bbox_inches='tight')
    plt.close()