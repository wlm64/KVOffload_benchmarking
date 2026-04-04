# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:32:55 2026

@author: Willie
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# =========================================
# SYSTEM PARAMETERS
# =========================================

BW_PCIe_B200 = 64
BW_PCIe_H100 = 64
BW_PCIe_A100 = 32

A100_CEFF = 312
H100_CEFF = 1000
B200_CEFF = 2500

A100_COST = 27 / 3600
H100_COST = 55 / 3600
B200_COST = 114 / 3600

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})


# =========================================
# MODELS
# =========================================

models = {
    'LLama-70b': {'size': 70, 'kv': 0.0003},
    'Deepseek-V3': {'size': 37, 'kv': 0.00007},
    'Qwen3-235B-A22B': {'size': 22, 'kv': 0.0002},
}

# =========================================
# TTFT FUNCTION
# =========================================

def compute_ttft(K, T, model_size, kv_bytes, Ceff, BW_PCIe):
    F_pf = 2 * model_size * 1e9
    t_pcie = (K * kv_bytes * 1e12) / BW_PCIe
    t_prefill = (T * F_pf) / Ceff
    return t_pcie + t_prefill

# =========================================
# PARETO FUNCTION (WITH KT)
# =========================================

def pareto_frontier(costs, ttfts, kts):
    points = sorted(zip(costs, ttfts, kts))  # sort by cost

    pareto = []
    best_ttft = -np.inf

    for c, t, k in points:
        if t > best_ttft:
            pareto.append((c, t, k))
            best_ttft = t

    return np.array(pareto)

# =========================================
# GENERIC RUN FUNCTION
# =========================================

def run_pareto(df_base, thresholds,
               fast_name, fast_ceff, fast_cost, fast_pcie,
               slow_name, slow_ceff, slow_cost, slow_pcie,
               outfile):

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

    for (model_name, specs), color in zip(models.items(), colors):

        model_size = specs["size"]
        kv_bytes = specs["kv"]

        df = df_base.copy()

        # Compute TTFTs
        df["ttft_fast"] = compute_ttft(
            df["context_tokens"],
            df["question_tokens"],
            model_size,
            kv_bytes,
            fast_ceff,
            fast_pcie
        )

        df["ttft_slow"] = compute_ttft(
            df["context_tokens"],
            df["question_tokens"],
            model_size,
            kv_bytes,
            slow_ceff,
            slow_pcie
        )

        baseline_ttft = df["ttft_fast"].sum()
        baseline_cost = (df["ttft_fast"] * fast_cost).sum()

        costs = []
        ttfts = []
        kts = []

        for thresh in thresholds:

            use_fast = df["KT_ratio"] < thresh

            total_ttft = np.where(
                use_fast,
                df["ttft_fast"],
                df["ttft_slow"]
            ).sum()

            total_cost = np.where(
                use_fast,
                df["ttft_fast"] * fast_cost,
                df["ttft_slow"] * slow_cost
            ).sum()

            norm_ttft = baseline_ttft / total_ttft
            norm_cost = total_cost / baseline_cost

            costs.append(norm_cost)
            ttfts.append(norm_ttft)
            kts.append(thresh)

        costs = np.array(costs)
        ttfts = np.array(ttfts)
        kts = np.array(kts)

        # Scatter all points
        plt.scatter(costs, ttfts, color=color, alpha=0.2)

        # Pareto frontier
        pareto = pareto_frontier(costs, ttfts, kts)

        # Plot Pareto curve
        plt.plot(
            pareto[:, 0],
            pareto[:, 1],
            color=color,
            linewidth=3,
            label=model_name
        )

        # =========================================
        # BEST COST POINT (LEFTMOST)
        # =========================================
        best = pareto[0]

        plt.scatter(best[0], best[1], color=color, marker="o", zorder=6)

        plt.annotate(
            "$K_{CM}$" + f"={best[2]:.0f}",
            xy=(best[0], best[1]),
            xytext=(best[0] - .025, best[1] - 0.05),
            arrowprops=dict(arrowstyle="->"),
            fontsize=13
        )

        # =========================================
        # 95% TTFT POINT
        # =========================================
        mask = pareto[:, 1] >= 0.99

        if np.any(mask):
            idx = np.argmax(mask)
            pt = pareto[idx]

            plt.scatter(pt[0], pt[1], color=color, marker="s", zorder=6)

            plt.annotate(
                "$K_{ISO}$" + f"={pt[2]:.0f}",
                xy=(pt[0], pt[1]),
                xytext=(pt[0], pt[1] + 0.03),
                arrowprops=dict(arrowstyle="->"),
                fontsize=13
            )

    # Reference lines
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.axvline(1.0, linestyle="--", linewidth=1)

    plt.xlabel("Normalized Cost")
    plt.ylabel("Normalized Throughput")
    plt.legend(loc='lower right') #
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"✅ Saved {outfile}")

# =========================================
# MAIN
# =========================================

csv_list = ['./sharegpt_effective_prefill.csv']

for csv in csv_list:

    dataset = csv.split('/')[1].split('_')[0]

    if dataset == 'sharegpt':
        thresholds = np.linspace(0, 700, 201)
    else:
        thresholds = np.linspace(0, 10000, 501)

    dfs = [pd.read_csv(f) for f in glob.glob(csv)]
    df_base = pd.concat(dfs, ignore_index=True)

    df_base["KT_ratio"] = (
        df_base["context_tokens"] / df_base["question_tokens"]
    )

    # =========================================
    # A100 vs B200
    # =========================================
    run_pareto(
        df_base,
        thresholds,
        fast_name="B200",
        fast_ceff=B200_CEFF,
        fast_cost=B200_COST,
        fast_pcie=BW_PCIe_B200,
        slow_name="H100",
        slow_ceff=H100_CEFF,
        slow_cost=H100_COST,
        slow_pcie=BW_PCIe_H100,
        outfile=f"pareto_H100_B200_{dataset}.png"
    )

    # =========================================
    # A100 vs H100
    # =========================================
    run_pareto(
        df_base,
        thresholds,
        fast_name="H100",
        fast_ceff=H100_CEFF,
        fast_cost=H100_COST,
        fast_pcie=BW_PCIe_H100,
        slow_name="A100",
        slow_ceff=A100_CEFF,
        slow_cost=A100_COST,
        slow_pcie=BW_PCIe_A100,
        outfile=f"pareto_A100_H100_{dataset}.png"
    )