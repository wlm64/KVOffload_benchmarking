# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:29:34 2026

@author: Willie
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# USER INPUT
# =========================================

BW_PCIe = 64  # PCIe bandwidth (bytes/sec)

gpus = {
    "A100": 312,
    "H100": 2000,
    "B200": 4500
}

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

# ---- DEFINE YOUR MODELS HERE ----
# B_kv: bytes per token
# F_pf: FLOPs per token
models = {
    'LLama-405b': {'size': 405, 'kv': 0.0005},
    'LLama-70b': {'size': 70, 'kv': 0.0003},
     'Deepseek-V3': {'size': 37, 'kv': 0.00007},
    # 'Deepseek-V2-Lite': {'size': 2.4, 'kv': 0.00003},
    'Qwen3-235B-A22B': {'size': 22, 'kv': 0.0002},
    'Qwen3-30B-A3B': {'size': 3.3, 'kv': 0.000092},
}


# Sweep K/T ratio
kt_ratios = np.linspace(0, 2000, 101)
T = 1000
sharegpt_ratio = 100

# =========================================
# TTFT Function
# =========================================
def compute_ttft(K, T, size, B_kv, Ceff):
    F_pf = 2 * size * 1e9   # per your definition
    t_pcie = (K * B_kv * 1e12) / BW_PCIe
    t_prefill = (T * F_pf) / Ceff
    return t_pcie + t_prefill


# =========================================
# Plot H100 / B200 (normalized by B200)
# =========================================

plt.figure(figsize=(6, 4))

colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

qwen_share_y = None

for (model_name, specs), color in zip(models.items(), colors):

    size = specs["size"]
    B_kv = specs["kv"]

    ratio_curve = []

    for ratio in kt_ratios:
        K = ratio * T

        ttft_h100 = compute_ttft(K, T, size, B_kv, gpus["H100"])
        ttft_b200 = compute_ttft(K, T, size, B_kv, gpus["B200"])

        ratio_curve.append(ttft_h100 / ttft_b200)

    plt.plot(
        kt_ratios,
        ratio_curve,
        linewidth=2,
        label=model_name,
        color=color
    )

    # ShareGPT point
    K_share = sharegpt_ratio * T
    ttft_h100 = compute_ttft(K_share, T, size, B_kv, gpus["H100"])
    ttft_b200 = compute_ttft(K_share, T, size, B_kv, gpus["B200"])
    y_share = ttft_h100 / ttft_b200

    if model_name == "Qwen3-235B-A22B":
        qwen_share_y = y_share


# Vertical ShareGPT line
plt.axvline(sharegpt_ratio, linestyle="--", linewidth=1.5, color="black")
plt.text(
    sharegpt_ratio * 1.1,
    plt.ylim()[1] * 0.95,
    r"ShareGPT $\kappa_{\mathrm{ratio}}$= 100",
    verticalalignment="top"
)

# -------------------------------------------------------
# Annotate Qwen3-235B speedup at ShareGPT
# -------------------------------------------------------
if qwen_share_y is not None:

    # Highlight the point
    plt.scatter(
        sharegpt_ratio,
        qwen_share_y,
        color="black",
        s=40,
        zorder=10,
    )
    
    # Arrow from the point to the middle of the figure
    plt.annotate(
        f"{qwen_share_y:.2f}× TTFT\nSpeedup",
        xy=(sharegpt_ratio, qwen_share_y),      # arrow starts here
        xytext=(700, 1.7),                      # label location
        fontsize=13,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color="black",
            shrinkA=0,
            shrinkB=5,
        ),
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="black",
            alpha=0.9,
        ),
    )
plt.axhline(1.0, linestyle="--", linewidth=1)

plt.xlabel(r"$\kappa_{\mathrm{ratio}}$")
plt.ylabel("TTFT Improvement")
#plt.title("H100 vs B200 TTFT (Normalized by B200)")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("h100_vs_b200_sharegpt.png", dpi=300)
plt.show()
plt.close()

print("✅ Saved H100 vs B200 comparison with ShareGPT marker")