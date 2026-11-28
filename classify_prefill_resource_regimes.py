#!/usr/bin/env python3
"""
Classify requests as Compute-bound, Mixed, or Memory-bound for
LLaMA-3.1-70B and Qwen3-235B-A22B using an attention-adjusted kappa_crit.

Default: use dataset-level FLOP correction factors from the paper analysis.
Optional: --factor-mode request uses each request's (K,T) attention FLOPs.

Run:
    python classify_prefill_resource_regimes.py
    python classify_prefill_resource_regimes.py --input-dir . --outdir paper_plots
    python classify_prefill_resource_regimes.py --factor-mode request

Required columns in each CSV: context_tokens (K), question_tokens (T).
Outputs: normalized stacked bar plots (PNG/PDF), summary and thresholds
         CSVs, plus per-dataset classified request CSVs.

These labels reflect an approximate isolated-request compute/transfer balance,
not an observed bottleneck under overlap, batching, or partial cache residency.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Baseline kappa_crit values and architecture for optional per-request correction.
MODELS = {
    "LLaMA-3.1-70B": {
        "kcrit": 14.3,
        "active_parameters": 70e9,
        "n_layers": 80,
        "d_attn": 8192,
    },
    "Qwen3-235B-A22B": {
        "kcrit": 7.8,
        "active_parameters": 22e9,
        "n_layers": 94,
        "d_attn": 8192,
    },
}

# Factors for the datasets' representative context lengths. These are
# FLOP corrections to 2N, not corrections measured from end-to-end runtime.
DATASETS = {
    "ShareGPT": {
        "filename": "sharegpt_effective_prefill.csv",
        "factors": {"LLaMA-3.1-70B": 1.20, "Qwen3-235B-A22B": 1.78},
    },
    "FinQA": {
        "filename": "docfinqa_token_counts_all_splits.csv",
        "factors": {"LLaMA-3.1-70B": 5.00, "Qwen3-235B-A22B": 16.00},
    },
    "Mooncake ToolAgent": {
        "filename": "mooncake_toolagent_effective_prefill.csv",
        "factors": {"LLaMA-3.1-70B": 1.15, "Qwen3-235B-A22B": 1.56},
    },
}

REGIMES = ("Compute-bound", "Mixed", "Memory-bound")
REGIME_COLORS = {
    "Compute-bound": "tab:blue",
    "Mixed": "tab:green",
    "Memory-bound": "tab:orange",
}


def per_request_attention_factor(k, t, model):
    """Return F_pf(K,T) / (2N), with T new queries and K cached keys/values.

    F_pf(K,T) ~= 2N + 4*n_layers*d_attn*(K+(T+1)/2).
    Approximation assumes conventional (GQA) attention arithmetic.
    """
    two_n = 2.0 * model["active_parameters"]
    attention = 4.0 * model["n_layers"] * model["d_attn"] * (
        k + (t + 1.0) / 2.0
    )
    return (two_n + attention) / two_n


def classify_dataset(dataset_name, config, input_dir, outdir, factor_mode, save_rows):
    filename = input_dir / config["filename"]
    if not filename.is_file():
        raise FileNotFoundError(f"Missing dataset: {filename}")

    df = pd.read_csv(filename)
    for col in ("context_tokens", "question_tokens"):
        if col not in df.columns:
            raise ValueError(f"{filename} is missing required column: {col}")

    k = pd.to_numeric(df["context_tokens"], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df["question_tokens"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(k) & np.isfinite(t) & (k >= 0) & (t > 0)

    # Keep invalid rows (with missing categories) in the optional row-wise CSV.
    ratio = np.full(len(df), np.nan, dtype=float)
    ratio[valid] = k[valid] / t[valid]
    df["prompt_length"] = k + t
    df["kappa_ratio"] = ratio

    print(f"\n{dataset_name}: {int(valid.sum()):,} valid requests, "
          f"{int((~valid).sum()):,} excluded (T <= 0, negatives, or NaNs)")

    summaries = []
    thresholds = []
    for model_name, model in MODELS.items():
        factors = np.full(len(df), np.nan, dtype=float)
        if factor_mode == "dataset":
            factors[valid] = config["factors"][model_name]
        else:
            factors[valid] = per_request_attention_factor(k[valid], t[valid], model)

        kcrit_adjusted = model["kcrit"] * factors
        lower = 0.5 * kcrit_adjusted
        upper = 1.5 * kcrit_adjusted

        # Include equality at either cutoff in the Mixed range.
        categories = np.full(len(df), "Invalid", dtype=object)
        categories[valid & (ratio < lower)] = "Compute-bound"
        categories[valid & (ratio >= lower) & (ratio <= upper)] = "Mixed"
        categories[valid & (ratio > upper)] = "Memory-bound"

        prefix = "llama70b" if "LLaMA" in model_name else "qwen235b"
        df[f"{prefix}_attention_factor"] = factors
        df[f"{prefix}_kcrit_adjusted"] = kcrit_adjusted
        df[f"{prefix}_lower_cutoff"] = lower
        df[f"{prefix}_upper_cutoff"] = upper
        df[f"{prefix}_regime"] = categories

        valid_categories = pd.Series(categories[valid])
        counts = valid_categories.value_counts().reindex(REGIMES, fill_value=0)
        n = int(valid.sum())
        for category in REGIMES:
            count = int(counts[category])
            summaries.append({
                "dataset": dataset_name,
                "model": model_name,
                "regime": category,
                "count": count,
                "total_valid_requests": n,
                "percentage": 100.0 * count / n if n else np.nan,
                "factor_mode": factor_mode,
            })

        thresholds.append({
            "dataset": dataset_name,
            "model": model_name,
            "baseline_kcrit": model["kcrit"],
            "factor_mode": factor_mode,
            "attention_factor_mean": np.mean(factors[valid]) if n else np.nan,
            "adjusted_kcrit_mean": np.mean(kcrit_adjusted[valid]) if n else np.nan,
            "lower_cutoff_mean": np.mean(lower[valid]) if n else np.nan,
            "upper_cutoff_mean": np.mean(upper[valid]) if n else np.nan,
        })

        print(f"  {model_name}: " + ", ".join(
            f"{r} {int(counts[r]):,} ({100*counts[r]/n:.1f}%)" if n else f"{r} 0"
            for r in REGIMES
        ))

    if save_rows:
        stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in dataset_name)
        df.to_csv(outdir / f"{stem}_classified_requests.csv", index=False)

    return summaries, thresholds


def make_plot(summary_df, outdir, factor_mode):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 15,
    })

    names = list(DATASETS.keys())
    labels = ["ShareGPT", "FinQA", "Mooncake"]
    x = np.arange(len(names))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model_name in zip(axs, MODELS):
        bottom = np.zeros(len(names), dtype=float)
        for regime in REGIMES:
            vals = np.array([
                summary_df.loc[
                    (summary_df["dataset"] == ds)
                    & (summary_df["model"] == model_name)
                    & (summary_df["regime"] == regime),
                    "percentage",
                ].iloc[0]
                for ds in names
            ], dtype=float)
            ax.bar(
                x, vals, bottom=bottom, width=0.64,
                label=regime, color=REGIME_COLORS[regime],
                edgecolor="white", linewidth=0.8,
            )
            for idx, val in enumerate(vals):
                if np.isfinite(val) and val >= 4.5:
                    ax.text(x[idx], bottom[idx] + val / 2.0,
                            f"{val:.0f}%", ha="center", va="center",
                            color="black" if regime != "Mixed" else "black",
                            fontsize=10, weight="semibold")
            bottom += np.nan_to_num(vals)

        # ax.set_title(model_name)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Request Percentage")
    handles, legends = axs[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.77, bottom=0.20, wspace=0.13)

    stem = f"prefill_resource_regimes_{factor_mode}"
    for ext in (["png"]):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches = 'tight')
    plt.show()
    print(f"\nPlot: {outdir / (stem + '.png')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("paper_plots"))
    parser.add_argument("--factor-mode", choices=("dataset", "request"),
                        default="request",
                        help="Default uses the paper's dataset-average attention factors; "
                             "request computes attention correction from each K,T")
    parser.add_argument("--no-rows", action="store_true",
                        help="Skip saving per-request classification CSV files")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_rows, threshold_rows = [], []
    for name, config in DATASETS.items():
        summary, threshold = classify_dataset(
            name, config, args.input_dir, args.outdir,
            args.factor_mode, save_rows=not args.no_rows,
        )
        summary_rows.extend(summary)
        threshold_rows.extend(threshold)

    summary_df = pd.DataFrame(summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    summary_df.to_csv(args.outdir / f"resource_regime_summary_{args.factor_mode}.csv",
                      index=False)
    threshold_df.to_csv(args.outdir / f"resource_regime_thresholds_{args.factor_mode}.csv",
                        index=False)
    make_plot(summary_df, args.outdir, args.factor_mode)
    print(f"Other CSV outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
