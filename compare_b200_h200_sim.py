# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:59:19 2026

@author: wlmen
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
})

# --------------------------------------------------
# Hardware assumptions
# --------------------------------------------------
# Edit H200 compute if you want to use a measured effective value.
# Using the same PCIe bandwidth means speedup -> 1 as kappa grows.
gpus = {
    "H200": {
        "compute_tflops": 2000,   # change if you want an empirical calibration
        "pcie_gbps": 64,
    },
    "B200": {
        "compute_tflops": 4500,
        "pcie_gbps": 64,
    },
}

# --------------------------------------------------
# Model parameters
# --------------------------------------------------
# N: active params
# B_kv_bytes: KV bytes per cached token
# A: 4 * n_layer * d_attn
models = {
    "LLaMA-3.1-70B": {
        "N": 70e9,
        "B_kv_bytes": 328 * 1000,
        "A": 2_621_440,
    },
    "Qwen3-235B-A22B": {
        "N": 22e9,
        "B_kv_bytes": 192 * 1000,
        "A": 3_080_192,
    },
    # If you want LLaMA-405B instead, uncomment this and swap below.
    # "LLaMA-3.1-405B": {
    #     "N": 405e9,
    #     "B_kv_bytes": 516 * 1000,
    #     "A": 8_257_536,
    # },
}

# --------------------------------------------------
# Analytical model
# --------------------------------------------------
def f_pf_per_token(K, T, N, A):
    """
    FLOPs per newly computed token:
      F_pf(K,T) = 2N + A*(K + T/2)
    """
    return 2 * N + A * (K + T / 2.0)

def ttft_seconds(K, T, N, A, B_kv_bytes, compute_tflops, pcie_gbps):
    """
    TTFT = t_gpu + t_pcie
      t_gpu  = T * F_pf(K,T) / C_eff
      t_pcie = K * B_kv / BW_pcie
    """
    C_eff = compute_tflops * 1e12   # TFLOP/s -> FLOP/s
    BW = pcie_gbps * 1e9            # GB/s -> B/s

    fpf = f_pf_per_token(K, T, N, A)
    t_gpu = T * fpf / C_eff
    t_pcie = K * B_kv_bytes / BW

    return t_gpu + t_pcie

# --------------------------------------------------
# Sweep setup
# --------------------------------------------------
T_values = [64, 128, 256, 512, 1024]
kappa_vals = np.linspace(0, 2000, 800)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300, sharey=True)
plt.subplots_adjust(top=0.82, wspace=0.14)

for ax, (model_name, p) in zip(axes, models.items()):
    for T in T_values:
        K_vals = kappa_vals * T

        ttft_h200 = np.array([
            ttft_seconds(
                K=K,
                T=T,
                N=p["N"],
                A=p["A"],
                B_kv_bytes=p["B_kv_bytes"],
                compute_tflops=gpus["H200"]["compute_tflops"],
                pcie_gbps=gpus["H200"]["pcie_gbps"],
            )
            for K in K_vals
        ])

        ttft_b200 = np.array([
            ttft_seconds(
                K=K,
                T=T,
                N=p["N"],
                A=p["A"],
                B_kv_bytes=p["B_kv_bytes"],
                compute_tflops=gpus["B200"]["compute_tflops"],
                pcie_gbps=gpus["B200"]["pcie_gbps"],
            )
            for K in K_vals
        ])

        # H200 baseline: >1 means B200 is faster
        speedup = ttft_h200 / ttft_b200

        ax.plot(kappa_vals, speedup, linewidth=3, label=f"T={T}")

    ax.axhline(1.0, linestyle="--", linewidth=2)
    # ax.set_title(model_name)
    ax.set_xlabel(r"$\kappa_{\mathrm{ratio}}$")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 200)

axes[0].set_ylabel("Relative TTFT Speedup")

# one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(T_values),
    frameon=False,
    columnspacing=0.9,
    handletextpad=0.4,
)

plt.savefig("./paper_plots/b200_vs_h200_ttft_speedup.png", dpi=300, bbox_inches="tight")
plt.show()