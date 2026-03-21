import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# =========================================
# SYSTEM PARAMETERS
# =========================================

BW_PCIe = 64  # bytes/sec

H100_CEFF = 1000
B200_CEFF = 2500

H100_COST = 55 / 3600   # $ per second
B200_COST = 114 / 3600   # $ per second



# =========================================
# MODELS
# =========================================

models = {
    'LLama-405b': {'size': 405, 'kv': 0.0005},
    'LLama-70b': {'size': 70, 'kv': 0.0003},
    'Deepseek-V3': {'size': 37, 'kv': 0.00007},
    'Qwen3-235B-A22B': {'size': 22, 'kv': 0.0002},
    'Qwen3-30B-A3B': {'size': 3.3, 'kv': 0.000092},
}

# =========================================
# LOAD TRACE
# =========================================
csv_list = ['./sharegpt_effective_prefill.csv', './narrativeqa_token_counts_all_splits.csv', './docfinqa_token_counts_all_splits.csv']
for csv in csv_list:
    dataset = csv.split('/')[1].split('_')[0]
    if dataset == 'sharegpt':
        thresholds = np.linspace(0, 500, 101)
    else:
        thresholds = np.linspace(0, 10000, 501)
    csv_files = glob.glob(csv)
    dfs = [pd.read_csv(f) for f in csv_files]
    df_base = pd.concat(dfs, ignore_index=True)
    
    df_base["KT_ratio"] = (
        df_base["context_tokens"] / df_base["question_tokens"]
    )
    
    # =========================================
    # TTFT FUNCTION
    # =========================================
    
    def compute_ttft(K, T, model_size, kv_bytes, Ceff):
        F_pf = 2 * model_size * 1e9
        t_pcie = (K * kv_bytes * 1e12) / BW_PCIe
        t_prefill = (T * F_pf) / Ceff
        return t_pcie + t_prefill
    
    
    # =========================================
    # PLOT
    # =========================================
    
    plt.figure(figsize=(10, 7))

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()
    
    # More distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    
    for (model_name, specs), color in zip(models.items(), colors):
    
        model_size = specs["size"]
        kv_bytes = specs["kv"]
    
        df = df_base.copy()
    
        df["ttft_h100"] = compute_ttft(
            df["context_tokens"],
            df["question_tokens"],
            model_size,
            kv_bytes,
            H100_CEFF
        )
    
        df["ttft_b200"] = compute_ttft(
            df["context_tokens"],
            df["question_tokens"],
            model_size,
            kv_bytes,
            B200_CEFF
        )
    
        baseline_ttft = df["ttft_b200"].sum()
        baseline_cost = (df["ttft_b200"] * B200_COST).sum()
    
        normalized_ttft = []
        normalized_cost = []
    
        for thresh in thresholds:
    
            use_b200 = df["KT_ratio"] < thresh
    
            total_ttft = np.where(
                use_b200,
                df["ttft_b200"],
                df["ttft_h100"]
            ).sum()
    
            total_cost = np.where(
                use_b200,
                df["ttft_b200"] * B200_COST,
                df["ttft_h100"] * H100_COST
            ).sum()
    
            normalized_ttft.append(baseline_ttft / total_ttft)
            normalized_cost.append(total_cost / baseline_cost)
    
        # TTFT on left axis (solid)
        ax1.plot(
            thresholds,
            normalized_ttft,
            color=color,
            linewidth=2.5,
            label=f"{model_name} TTFT"
        )
    
        # Cost on right axis (dashed)
        ax2.plot(
            thresholds,
            normalized_cost,
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f"{model_name} Cost"
        )
    
    # Reference lines
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    
    ax1.set_xlabel(r"$\kappa_{\mathrm{ratio}}$ Threshold")
    ax1.set_ylabel("Normalized TTFT")
    ax2.set_ylabel("Normalized Cost")
    
    #ax1.set_title("Dynamic B200/H100 Policy Across Models")
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2)
    
    ax1.grid(True, alpha=0.3)
    # if dataset == 'sharegpt':
    #     # Force both axes to start at 1
    #     ax1.set_ylim(bottom=.95)
    #     ax2.set_ylim(bottom=.95)

    plt.tight_layout()
    plt.savefig(f"multi_model_policy_tradeoff_{dataset}.png", dpi=300)
    plt.close()
    
    print("Saved multi_model_policy_tradeoff.png")