
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

# --- Load datasets ---
df2 = pd.read_csv("./mooncake_toolagent_effective_prefill.csv")
df1 = pd.read_csv("./docfinqa_token_counts_all_splits.csv")
df3 = pd.read_csv("./sharegpt_effective_prefill.csv")

datasets = [
    ("ShareGPT", df3),
    ("FinQA", df1),
    ("Mooncake ToolAgent", df2),
]

# --- Compute kappa ratio ---
for name, df in datasets:
    df["kappa_ratio"] = (
        df["context_tokens"] / df["question_tokens"].replace(0, np.nan)
    )
    df['prompt_length'] = df['context_tokens'] + df['question_tokens']
    
    print(np.mean(df['prompt_length']))
 
# --- Metrics to plot ---
metrics = [
    ("question_tokens", "Prefill Tokens"),
    ("context_tokens", "Reused KV"),
    ("kappa_ratio", r"$\kappa_{\mathrm{ratio}}$"),
]

# --- Plot setup: 3 rows x 3 columns ---
fig, axes = plt.subplots(3, 3, figsize=(14 , 8), dpi=300)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# --- Human-readable tick formatting ---
def human_format(x, pos = None):
    if x >= 1e6:
        return f"{x/1e6:g}M"
    elif x >= 1e3:
        return f"{x/1e3:g}k"
    return f"{x:g}"

# --- X-axis limits ---
x_limits = {
    ("ShareGPT", "question_tokens"): (0, 100),
    ("ShareGPT", "context_tokens"): (0, 32000),
    ("ShareGPT", "kappa_ratio"): (0, 2000),

    ("FinQA", "question_tokens"): (0, 30),
    ("FinQA", "context_tokens"): (50000, 450000),
    ("FinQA", "kappa_ratio"): (0, 40000),

    ("Mooncake ToolAgent", "question_tokens"): (0, 4000),
    ("Mooncake ToolAgent", "context_tokens"): (0, 10000),
    ("Mooncake ToolAgent", "kappa_ratio"): (0, 100),
}

# --- Plot CDFs ---
for row_idx, (ds_name, df) in enumerate(datasets):
    for col_idx, (col, col_label) in enumerate(metrics):
        ax = axes[row_idx, col_idx]

        data = np.sort(
            df[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        )

        if len(data) == 0:
            continue

        print(
            f"{ds_name}, {col_label}, "
            f"Mean: {np.mean(data):.2f}, "
            f"Median: {np.median(data):.2f}, "
            f"90%: {np.percentile(data, 90):.2f}, "
            f"95%: {np.percentile(data, 95):.2f}"
        )

        cdf = np.arange(1, len(data) + 1) / len(data)

        ax.plot(data, cdf, linestyle="-", label=ds_name)

        ax.set_xlabel(col_label)
        ax.set_ylabel("CDF")
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Apply per-plot x limits
        xmin, xmax = x_limits[(ds_name, col)]
        ax.set_xlim(xmin, xmax)

        # Human-readable ticks
        if col in ["context_tokens", "kappa_ratio"] or "Mooncake" in ds_name:
            ticks = np.linspace(xmin, xmax, num=5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([human_format(x) for x in ticks])
        
        # from matplotlib.ticker import MultipleLocator, FuncFormatter

        # if col in ["context_tokens", "kappa_ratio"] or "Mooncake" in ds_name:
        #     xmin, xmax = x_limits[(ds_name, col)]
        
        #     # Choose tick spacing in multiples of 1000
        #     spacing = max(1000, np.ceil((xmax - xmin) / 4000) * 1000)
        
        #     ax.xaxis.set_major_locator(MultipleLocator(spacing))
        #     ax.xaxis.set_major_formatter(FuncFormatter(human_format))

plt.tight_layout()
plt.savefig(
    "./paper_plots/sgpt_finqa_mooncake_cdf.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
