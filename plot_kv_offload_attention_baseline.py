
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration - CHANGE THESE FOR EACH MODEL
# ============================================================

# Original KV offloading benchmark directory
root_dir = "./kv_offload_serve/meta-llama"

# New attention / prefix-caching baseline directory
baseline_dir = "./kv_no_offload_warm_attention/meta-llama"

# Plot output directory
output_dir = "./paper_plots"
os.makedirs(output_dir, exist_ok=True)

# Plot settings
MIN_PREFILL = 32
MAX_PREFILL = 2048

KRATIO_XLIM = (0, 100)
POH_YLIM = (0, 20)

# Bar charts use total duration, line plots use TTFT.
# Set to "ttft" to use TTFT for both.
BAR_METRIC = "ttft"

# Font settings
plt.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
})

# ============================================================
# Directory discovery
#
# Supports:
# 1. Direct model folder containing benchmark logs
# 2. Parent folder containing multiple model folders
# ============================================================

def discover_model_dirs(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Offloading directory not found: {root}"
        )

    # Case 1: root directly contains benchmark files
    files = os.listdir(root)

    if any(
        f.startswith("benchmark_results_")
        and f.endswith(".log")
        for f in files
    ):
        return {
            os.path.basename(os.path.normpath(root)): root
        }

    # Case 2: root contains model subdirectories
    model_dirs = {}

    for folder in sorted(files):
        folder_path = os.path.join(root, folder)

        if not os.path.isdir(folder_path):
            continue

        if any(
            f.startswith("benchmark_results_")
            and f.endswith(".log")
            for f in os.listdir(folder_path)
        ):
            model_dirs[folder] = folder_path

    return model_dirs


def resolve_baseline_dir(model_name):
    """
    If baseline_dir directly contains JSON files,
    use it as the baseline directory.

    Otherwise, look for:
        baseline_dir / model_name
    """

    if not os.path.isdir(baseline_dir):
        return None

    files = os.listdir(baseline_dir)

    if any(
        f.startswith("benchmark_results_")
        and f.endswith(".json")
        for f in files
    ):
        return baseline_dir

    model_baseline = os.path.join(
        baseline_dir,
        model_name
    )

    if os.path.isdir(model_baseline):
        return model_baseline

    return None


# ============================================================
# Load original KV offloading benchmarks
# ============================================================

def load_offloading_logs(model_folder):

    records = []

    pattern = re.compile(
        r"benchmark_results_rate_(\d+)_(\d+)\.log$"
    )

    for filename in sorted(os.listdir(model_folder)):

        match = pattern.fullmatch(filename)

        if match is None:
            continue

        kv_size, prefill = map(int, match.groups())

        log_path = os.path.join(
            model_folder,
            filename
        )

        try:
            with open(log_path, "r") as f:
                data = json.load(f)

            records.append({
                "KV": kv_size,
                "Prefill": prefill,
                "duration": data.get(
                    "duration", np.nan
                ),
                "mean_ttft": data.get(
                    "mean_ttft_ms", np.nan
                ),
                "std_ttft": data.get(
                    "std_ttft_ms", np.nan
                )
            })

        except (OSError, ValueError) as e:
            print(f"Failed to read {log_path}: {e}")

    return pd.DataFrame(records)


# ============================================================
# Load NEW attention / prefix-caching baselines
#
# Baseline matches the same K and T as the offloaded run.
#
# Example:
# Offload:
#   benchmark_results_1024_64.log
#
# Baseline:
#   benchmark_results_1024_64.json
# ============================================================

def load_attention_baselines(folder):

    records = []

    pattern = re.compile(
        r"benchmark_results_(\d+)_(\d+)\.json$"
    )

    for filename in sorted(os.listdir(folder)):

        match = pattern.fullmatch(filename)

        if match is None:
            continue

        kv_size, prefill = map(int, match.groups())

        file_path = os.path.join(
            folder,
            filename
        )

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            records.append({
                "KV": kv_size,
                "Prefill": prefill,
                "baseline_duration": data.get(
                    "duration", np.nan
                ),
                "baseline_mean_ttft": data.get(
                    "mean_ttft_ms", np.nan
                ),
                "baseline_std_ttft": data.get(
                    "std_ttft_ms", np.nan
                )
            })

        except (OSError, ValueError) as e:
            print(f"Failed to read {file_path}: {e}")

    return pd.DataFrame(records)


# ============================================================
# Normalize offloading measurements using NEW baseline
# ============================================================

def normalize_data(df, baseline_df):

    if df.empty or baseline_df.empty:
        return pd.DataFrame()

    # Remove old KV=0 measurements
    # They are replaced by new attention baselines
    df = df[df["KV"] > 0].copy()

    # Match each offloaded measurement with baseline
    # sharing the exact same (K, T).
    df_plot = df.merge(
        baseline_df,
        on=["KV", "Prefill"],
        how="left",
        validate="many_to_one",
        indicator=True
    )

    # Report missing baseline combinations
    missing = df_plot[
        df_plot["_merge"] == "left_only"
    ]

    if not missing.empty:
        print("\nMissing baseline measurements:")

        for _, row in missing.iterrows():
            print(
                f"  K={int(row['KV'])}, "
                f"T={int(row['Prefill'])}"
            )

    df_plot = df_plot.drop(columns=["_merge"])

    # Convert measurement columns to numeric
    numeric_columns = [
        "duration",
        "mean_ttft",
        "std_ttft",
        "baseline_duration",
        "baseline_mean_ttft",
        "baseline_std_ttft"
    ]

    for col in numeric_columns:
        df_plot[col] = pd.to_numeric(
            df_plot[col], errors="coerce"
        )

    # Keep only measurements with valid TTFT baselines
    valid = (
        df_plot["mean_ttft"].notna()
        & df_plot["baseline_mean_ttft"].notna()
        & (df_plot["baseline_mean_ttft"] > 0)
    )

    if BAR_METRIC == "duration":
        valid = (
            valid
            & df_plot["duration"].notna()
            & df_plot["baseline_duration"].notna()
            & (df_plot["baseline_duration"] > 0)
        )

    df_plot = df_plot[valid].copy()

    if df_plot.empty:
        return df_plot

    # Calculate K_ratio
    df_plot["K/T"] = (
        df_plot["KV"] / df_plot["Prefill"]
    )

    # TTFT normalized against attention baseline
    df_plot["normalized_ttft"] = (
        df_plot["mean_ttft"]
        / df_plot["baseline_mean_ttft"]
    )

    # Duration normalized against attention baseline
    df_plot["normalized_duration"] = (
        df_plot["duration"]
        / df_plot["baseline_duration"]
    )

    # Original standard deviation normalization
    df_plot["std_normalized"] = (
        df_plot["std_ttft"]
        / df_plot["baseline_mean_ttft"]
    )

    # Calculate overhead directly
    df_plot["poh_ttft"] = (
        df_plot["normalized_ttft"] - 1
    )

    df_plot["poh_duration"] = (
        df_plot["normalized_duration"] - 1
    )

    return df_plot


# ============================================================
# Plot 1: Grouped bar chart
# P_OH vs Offloaded KV, grouped by prefill length
# ============================================================

def plot_grouped_bars(df_plot, model_name):

    kv_sizes = sorted(
        df_plot["KV"].unique()
    )

    prefill_values = sorted(
        p for p in df_plot["Prefill"].unique()
        if MIN_PREFILL < p <= MAX_PREFILL
    )

    if not prefill_values:
        return

    x = np.arange(len(kv_sizes))
    width = 0.8 / len(prefill_values)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for i, prefill in enumerate(prefill_values):

        df_prefill = (
            df_plot[
                df_plot["Prefill"] == prefill
            ]
            .set_index("KV")
        )

        if BAR_METRIC == "ttft":
            heights = (
                df_prefill["poh_ttft"]
                .reindex(kv_sizes)
                .values
            )

            errors = (
                df_prefill["std_normalized"]
                .reindex(kv_sizes)
                .values
            )

        else:
            heights = (
                df_prefill["poh_duration"]
                .reindex(kv_sizes)
                .values
            )

            # No duration SD is available in the logs.
            errors = None

        ax.bar(
            x + i * width,
            heights,
            width,
            label=f"Prefill={prefill}",
            edgecolor="black",
            yerr=errors,
            capsize=4,
            linewidth=1
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

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 1])

    out_path = os.path.join(
        output_dir,
        f"{model_name}_normalized_ttft.png"
    )

    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Saved: {out_path}")


# ============================================================
# Plot 2: P_OH vs K_ratio
# Uses normalized TTFT
# ============================================================

def plot_poh_vs_kratio(df_plot, model_name):

    prefill_values = sorted(
        p for p in df_plot["Prefill"].unique()
        if MIN_PREFILL < p <= MAX_PREFILL
    )

    if not prefill_values:
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for prefill in prefill_values:

        df_prefill = df_plot[
            df_plot["Prefill"] == prefill
        ].copy()

        df_prefill = df_prefill.sort_values("K/T")

        x_vals = df_prefill["K/T"].values

        y_vals = df_prefill["poh_ttft"].values

        errors = df_prefill["std_normalized"].values

        if np.all(np.isnan(errors)):
            errors = None
        else:
            errors = np.nan_to_num(
                errors, nan=0.0
            )

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=errors,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            label=f"Prefill={prefill}"
        )

    ax.set_xlabel(
        r"$K_{\mathrm{ratio}}$"
    )

    ax.set_ylabel(
        r"$P_{OH}$"
    )

    ax.grid(
        axis="both",
        linestyle="--",
        alpha=0.5
    )

    ax.legend()

    ax.set_xlim(*KRATIO_XLIM)
    ax.set_ylim(*POH_YLIM)

    fig.tight_layout(rect=[0, 0, 1, 1])

    out_path = os.path.join(
        output_dir,
        f"{model_name}_poh_vs_kratio.png"
    )

    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():

    model_dirs = discover_model_dirs(root_dir)

    if not model_dirs:
        print(
            f"No benchmark logs found in {root_dir}"
        )
        return

    # Prevent accidentally reusing one direct baseline
    # folder for multiple different models.
    if len(model_dirs) > 1:
        direct_json = any(
            f.startswith("benchmark_results_")
            and f.endswith(".json")
            for f in os.listdir(baseline_dir)
        ) if os.path.isdir(baseline_dir) else False

        if direct_json:
            raise ValueError(
                "Multiple offload models detected, but "
                "baseline_dir points to one model's JSON files. "
                "Set baseline_dir to the parent directory "
                "containing matching model subdirectories."
            )

    for model_name, model_folder in model_dirs.items():

        model_baseline = resolve_baseline_dir(
            model_name
        )

        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print(f"Offload: {model_folder}")
        print(f"Baseline: {model_baseline}")

        if model_baseline is None:
            print("No baseline directory found. Skipping.")
            continue

        df = load_offloading_logs(model_folder)

        baseline_df = load_attention_baselines(
            model_baseline
        )

        if df.empty or baseline_df.empty:
            print("Missing measurements. Skipping.")
            continue

        # Normalize using attention-enabled baseline
        df_plot = normalize_data(
            df, baseline_df
        )

        if df_plot.empty:
            print("No matching valid measurements.")
            continue

        print(
            f"Matched {len(df_plot)} measurements "
            f"to attention baselines."
        )

        # Save normalized results
        csv_path = os.path.join(
            output_dir,
            f"{model_name}_normalized.csv"
        )

        df_plot.to_csv(
            csv_path, index=False
        )

        print(f"Saved: {csv_path}")

        # Generate original plots
        plot_grouped_bars(
            df_plot, model_name
        )

        plot_poh_vs_kratio(
            df_plot, model_name
        )

    print("\nAll plots completed.")


if __name__ == "__main__":
    main()
