# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:49:52 2026

@author: Willie
"""


import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
root_dir = "./kv_offload_serve"
gpu = "h100"
output_dir = "./paper_plots"
os.makedirs(output_dir, exist_ok=True)

# 1.0 = scale the entire baseline TTFT by the FLOP ratio
# 0.0 = no attention correction
COMPUTE_FRACTION = 1.0

plt.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
})

# Model: layers, active parameters (billions),
# query heads, QK dimension/head, value dimension/head
MODELS = {
    "llama70": (80, 70.0, 64, 128, 128),
    "llama405": (126, 405.0, 128, 128, 128),
    "qwen30": (48, 3.3, 32, 128, 128),
    "qwen235": (94, 22.0, 64, 128, 128),
    "deepseekv3": (61, 37.0, 128, 192, 128),
}


def get_model(model_path):
    """Identify model from the directory name."""
    name = re.sub(r"[^a-z0-9]", "", model_path.lower())

    if "qwen" in name and "235b" in name:
        return MODELS["qwen235"]
    if "qwen" in name and "30b" in name:
        return MODELS["qwen30"]
    if "llama" in name and "405b" in name:
        return MODELS["llama405"]
    if "llama" in name and "70b" in name:
        return MODELS["llama70"]
    if "deepseek" in name and "v2" in name:
        return MODELS["deepseekv3"]
    if "gpt" in name:
        return MODELS["deepseekv3"]

    raise ValueError(f"Unknown model: {model_path}")


def flops_kaplan(model, total_context):
    """FLOPs per token for full causal prefill."""
    layers, active_b, q_heads, qk_dim, v_dim = model

    two_n = 2.0 * active_b * 1e9
    d_attn = q_heads * (qk_dim + v_dim) / 2.0

    return (
        two_n
        + 2.0 * layers * np.asarray(total_context)
        * d_attn
    )


def flops_reused_prefix(model, total_context, new_tokens):
    """FLOPs per new token when attending to cached KV."""
    layers, active_b, q_heads, qk_dim, v_dim = model

    two_n = 2.0 * active_b * 1e9
    d_attn = q_heads * (qk_dim + v_dim) / 2.0

    L = np.asarray(total_context, dtype=float)
    T = np.minimum(np.asarray(new_tokens, dtype=float), L)
    K = L - T

    return two_n + 4.0 * layers * d_attn * (K + T / 2.0)


# --- Iterate over folders ---
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    subfolders = [
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    if not subfolders:
        continue

    model_folder = os.path.join(folder_path, subfolders[0])
    model = get_model(model_folder)

    log_files = [
        f for f in os.listdir(model_folder)
        if f.startswith("benchmark_results") and f.endswith(".log")
    ]

    if not log_files:
        print(f"Skipping {folder}, no benchmark logs found")
        continue

    # --- Parse benchmark logs ---
    records = []

    for log_file in log_files:
        kv_match = [
            int(s)
            for s in log_file.replace(".log", "").split("_")
            if s.isdigit()
        ]

        if len(kv_match) != 2:
            continue

        kv_size, prefill = kv_match
        log_path = os.path.join(model_folder, log_file)

        with open(log_path, "r") as f:
            data = pd.read_json(f.read(), typ="series")

        records.append({
            "KV": kv_size,
            "Prefill": prefill,
            "duration": data.get("duration", np.nan),
            "mean_ttft": data.get("mean_ttft_ms", np.nan),
            "std_ttft": data.get("std_ttft_ms", np.nan),
        })

    df = pd.DataFrame(records)

    if df.empty:
        continue

    # --- Baseline measurements (KV=0) ---
    baseline_df = df[df["KV"] == 0].set_index("Prefill")

    baseline_ttft = baseline_df["mean_ttft"]
    baseline_duration = baseline_df["duration"]

    df_plot = df.copy()

    df_plot["baseline_ttft"] = df_plot["Prefill"].map(baseline_ttft)
    df_plot["baseline_duration"] = df_plot["Prefill"].map(
        baseline_duration
    )

    df_plot["K/T"] = df_plot["KV"] / df_plot["Prefill"]

    df_plot["normalized_ttft"] = (
        df_plot["mean_ttft"] / df_plot["baseline_ttft"]
    )

    df_plot["normalized_duration"] = (
        df_plot["duration"] / df_plot["baseline_duration"]
    )

    df_plot["std_normalized"] = (
        df_plot["std_ttft"] / df_plot["baseline_ttft"]
    )

    # ============================================
    # NEW: FLOP-aware baseline correction
    # ============================================

    K = df_plot["KV"].to_numpy(dtype=float)
    T = df_plot["Prefill"].to_numpy(dtype=float)

    # Baseline: full prefill of T tokens
    F_baseline = flops_kaplan(model, T)

    # Offloaded case: T new tokens attending to K cached tokens
    F_offloaded = flops_reused_prefix(model, K + T, T)

    # FLOP ratio between the two cases
    df_plot["flops_ratio"] = F_offloaded / F_baseline

    # Time scaling factor
    df_plot["time_scale"] = (
        1.0
        + COMPUTE_FRACTION * (df_plot["flops_ratio"] - 1.0)
    )

    # Scale the baseline TTFT
    df_plot["adjusted_baseline_ttft"] = (
        df_plot["baseline_ttft"] * df_plot["time_scale"]
    )

    # Original and corrected P_OH
    df_plot["poh_original"] = df_plot["normalized_ttft"] - 1.0

    df_plot["poh_adjusted"] = (
        df_plot["mean_ttft"]
        / df_plot["adjusted_baseline_ttft"]
        - 1.0
    )

    # Adjust error bars to use the scaled baseline
    df_plot["std_adjusted"] = (
        df_plot["std_ttft"] / df_plot["adjusted_baseline_ttft"]
    )

    # Optionally correct benchmark duration as well
    df_plot["adjusted_baseline_duration"] = (
        df_plot["baseline_duration"] * df_plot["time_scale"]
    )

    df_plot["normalized_duration_adjusted"] = (
        df_plot["duration"]
        / df_plot["adjusted_baseline_duration"]
    )

    # Save measurements for further analysis
    csv_path = os.path.join(
        output_dir, f"{folder}_attention_adjusted.csv"
    )
    df_plot.to_csv(csv_path, index=False)

    print(
        f"{folder}: FLOP ratio range = "
        f"{df_plot['flops_ratio'].min():.2f}x to "
        f"{df_plot['flops_ratio'].max():.2f}x"
    )

    # --- Prepare plotting data ---
    df_plot = df_plot.set_index("Prefill")
    df_plot = df_plot[df_plot["KV"] != 0].copy()

    kv_sizes = sorted(df_plot["KV"].unique())

    prefill_values = [
        x for x in sorted(df_plot.index.unique())
        if 32 < x <= 2048
    ]

    if not prefill_values:
        continue

    # ============================================
    # Plot 1: Original grouped bar chart
    # ============================================

    x = np.arange(len(kv_sizes))
    width = 0.8 / len(prefill_values)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for i, prefill in enumerate(prefill_values):
        df_prefill = (
            df_plot[df_plot.index == prefill]
            .set_index("KV")
        )

        heights = (
            df_prefill["poh_original"]
            .reindex(kv_sizes).values
        )

        errors = (
            df_prefill["std_normalized"]
            .reindex(kv_sizes).values
        )

        ax.bar(
            x + i * width,
            heights,
            width,
            label=f"Prefill={prefill}",
            edgecolor="black",
            yerr=errors,
            capsize=4,
            linewidth=1,
        )

    ax.set_xticks(
        x + width * (len(prefill_values) - 1) / 2
    )
    ax.set_xticklabels(
        [str(k) for k in kv_sizes],
        rotation=30,
        ha="center"
    )

    ax.set_xlabel("Offloaded KV")
    ax.set_ylabel(r"$P_{OH}$")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()

    out_path = os.path.join(
        output_dir, f"{folder}_normalized_ttft.png"
    )

    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    
    
    # ============================================
    # Plot 1B: Separate bar chart for each KV size
    # ============================================
    
    kv_output_dir = os.path.join(
        output_dir, f"{folder}_per_kv"
    )
    os.makedirs(kv_output_dir, exist_ok=True)
    
    for kv_size in kv_sizes:
    
        # Select measurements for one fixed offloaded KV size
        df_kv = df_plot[
            (df_plot["KV"] == kv_size) &
            (df_plot.index.isin(prefill_values))
        ].sort_index()
    
        if df_kv.empty:
            continue
    
        prefill_tokens = df_kv.index.to_numpy()
        x = np.arange(len(prefill_tokens))
        width = 0.35
    
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
        # Original P_OH
        ax.bar(
            x - width / 2,
            df_kv["poh_original"].values,
            width,
            yerr=df_kv["std_normalized"].values,
            label="Original",
            edgecolor="black",
            linewidth=1,
            capsize=4,
        )
    
        # Attention-adjusted P_OH
        ax.bar(
            x + width / 2,
            df_kv["poh_adjusted"].values,
            width,
            yerr=df_kv["std_adjusted"].values,
            label="Attention Adjusted",
            edgecolor="black",
            linewidth=1,
            capsize=4,
        )
    
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(t) for t in prefill_tokens],
            rotation=30,
            ha="center"
        )
    
        ax.set_xlabel("Prefill Tokens", fontsize=20)
        ax.set_ylabel(r"$P_{OH}$", fontsize=20)
    
        ax.set_title(
            f"Offloaded KV = {kv_size:,}",
            fontsize=20
        )
    
        ax.tick_params(axis="both", labelsize=18)
    
        ax.axhline(
            0, color="black", linewidth=1
        )
    
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )
        ax.set_axisbelow(True)
    
        ax.legend(fontsize=15)
    
        fig.tight_layout()
    
        out_path = os.path.join(
            kv_output_dir,
            f"{folder}_KV_{kv_size}_poh.png"
        )
    
        fig.savefig(out_path, dpi=300)
        plt.show()
        plt.close(fig)
    
        print(f"Saved: {out_path}")


    # ============================================
    # Plot 2: Original vs attention-adjusted P_OH
    # ============================================

    fig, axes = plt.subplots(
        1, 2,
        figsize=(20, 7.5),
        sharex=True,
        sharey=True
    )

    # colors = plt.cm.viridis(
    #     np.linspace(0.05, 0.95, len(prefill_values))
    # )

    for i, prefill in enumerate(prefill_values):

        df_prefill = (
            df_plot[df_plot.index == prefill]
            .sort_values("K/T")
        )

        x_vals = df_prefill["K/T"].values

        # Original P_OH
        axes[0].errorbar(
            x_vals,
            df_prefill["poh_original"].values,
            yerr=df_prefill["std_normalized"].values,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            # color=colors[i],
            label=f"Prefill={prefill}",
        )

        # Attention-adjusted P_OH
        axes[1].errorbar(
            x_vals,
            df_prefill["poh_adjusted"].values,
            yerr=df_prefill["std_adjusted"].values,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            # color=colors[i],
            label=f"Prefill={prefill}",
        )

    axes[0].set_title("Original")
    axes[1].set_title("Attention Adjusted")

    for ax in axes:
        ax.set_xlabel(r"$K_{\mathrm{ratio}} = K/T$")
        ax.set_xscale("log")
        ax.axhline(
            0, color="black", linestyle="--", linewidth=1
        )
        ax.grid(
            axis="both", linestyle="--", alpha=0.5
        )

    axes[0].set_ylabel(r"$P_{OH}$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(4, len(prefill_values)),
        frameon=False,
        fontsize=20,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    plt.ylim(0,20)
    out_path = os.path.join(
        output_dir,
        f"{folder}_poh_attention_comparison.png"
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Saved: {csv_path}")

print("All plots saved.")
