# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:44:47 2026

@author: wlmen
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_h100 = r"C:\Users\wlmen\OneDrive - PennO365\KVOffload_Data\kv_offload_h100"
base_b200 = r"C:\Users\wlmen\OneDrive - PennO365\KVOffload_Data\kv_offload_b200"

output_dir = r"C:\Users\wlmen\OneDrive - PennO365\KVOffload_Data\kv_decomposed_plots"
os.makedirs(output_dir, exist_ok=True)

def find_all_csvs(base_dir):
    paths = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f == "kv_transfer_time_merged.csv":
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                paths[rel_path] = full_path
    return paths

h100_files = find_all_csvs(base_h100)
b200_files = find_all_csvs(base_b200)

common_keys = set(h100_files.keys()) & set(b200_files.keys())

for rel_path in common_keys:

    df_h100 = pd.read_csv(h100_files[rel_path], index_col=0)
    df_b200 = pd.read_csv(b200_files[rel_path], index_col=0)

    # enforce numeric
    df_h100.index = df_h100.index.astype(int)
    df_b200.index = df_b200.index.astype(int)
    df_h100.columns = df_h100.columns.astype(int)
    df_b200.columns = df_b200.columns.astype(int)

    # filter prefill >= 64
    df_h100 = df_h100.loc[:, df_h100.columns >= 64]
    df_b200 = df_b200.loc[:, df_b200.columns >= 64]

    if df_h100.shape[1] == 0:
        continue

    # sort
    df_h100 = df_h100.sort_index().sort_index(axis=1)
    df_b200 = df_b200.sort_index().sort_index(axis=1)

    if 0 not in df_h100.index:
        print(f"No KV=0 row: {rel_path}")
        continue

    # -------------------------
    # 1. PREFILL ANALYSIS
    # -------------------------
    prefill_h100 = df_h100.loc[0]
    prefill_b200 = df_b200.loc[0]

    rel_prefill = prefill_h100 / prefill_b200

    # save prefill csv
    prefill_out = os.path.join(output_dir, rel_path.replace(".csv", "_prefill.csv"))
    os.makedirs(os.path.dirname(prefill_out), exist_ok=True)
    rel_prefill.to_csv(prefill_out)

    # plot prefill
    plt.figure(figsize=(6,4))
    plt.plot(rel_prefill.index, rel_prefill.values, marker='o')
    plt.axhline(1.0, linestyle="--")

    model_name = "/".join(rel_path.split(os.sep)[:-1])

    plt.title(f"{model_name} (Prefill)")
    plt.xlabel("Prefill Tokens")
    plt.ylabel("Relative Prefill Time (H100 / B200)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, model_name.replace("/", "_") + "_prefill.png"), dpi=200)
    plt.close()

    # -------------------------
    # 2. PCIE ANALYSIS
    # -------------------------

    pcie_points = []

    for kv in df_h100.index:
        if kv == 0:
            continue

        for prefill in df_h100.columns:

            p_h100 = prefill_h100[prefill]
            p_b200 = prefill_b200[prefill]

            t_h100 = df_h100.loc[kv, prefill]
            t_b200 = df_b200.loc[kv, prefill]

            pcie_h100 = t_h100 - p_h100
            pcie_b200 = t_b200 - p_b200

            # avoid divide-by-zero / noise
            if pcie_b200 <= 0:
                continue

            rel_pcie = pcie_h100 / pcie_b200

            pcie_points.append((kv, rel_pcie))

    if len(pcie_points) == 0:
        continue

    df_pcie = pd.DataFrame(pcie_points, columns=["kv", "rel_pcie"])

    # save raw pcie points
    pcie_out = os.path.join(output_dir, rel_path.replace(".csv", "_pcie.csv"))
    df_pcie.to_csv(pcie_out, index=False)
    
    # plot raw PCIe points
    plt.figure(figsize=(6,4))
    plt.scatter(df_pcie["kv"], df_pcie["rel_pcie"], marker='o')  # use scatter for individual points
    plt.axhline(1.0, linestyle="--")
    
    plt.title(f"{model_name} (PCIe)")
    plt.xlabel("KV Tokens")
    plt.ylabel("Relative PCIe Time (H100 / B200)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, model_name.replace("/", "_") + "_pcie.png"), dpi=200)
    plt.close()

    print(f"Processed: {model_name}")