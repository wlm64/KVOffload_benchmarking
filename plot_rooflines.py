#!/usr/bin/env python3
"""Attention-aware PCIe rooflines with empirical workload AI bands.

One figure per model. Each samples up to 1000 requests from each CSV.
Workloads: 5th-95th percentile translucent band + ONE dotted median line.
Hardware: direct labels on PCIe diagonals and GPU horizontal ceilings,
matching the style of the original roofline figure (no GPU crossovers).

Inputs in --data-dir (default '.'): 
  sharegpt_effective_prefill.csv
  docfinqa_token_counts_all_splits.csv
  mooncake_toolagent_effective_prefill.csv

Columns: question_tokens=T; context_tokens=K.

This plots theoretical roofline bounds, not observed throughput.
K=0 samples have infinite host-transfer arithmetic intensity and are
excluded from finite-intensity shading/median (reported on stdout).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
})

GPUS = {
    # TFLOP/s of modeled compute, GB/s of modeled PCIe bandwidth
    "A100": {"compute": 312.0, "pcie": 32.0, "color": "green"},
    "H100": {"compute": 2000.0, "pcie": 64.0, "color": "blue"},
    "B200": {"compute": 4500.0, "pcie": 64.0, "color": "black"},
}
MODELS = {
    "LLaMA-3.1-70B": {"N": 70e9, "A": 2621440.0, "Bkv": 328e3,
                      "filename": "roofline_LLaMA-3.1-70B_workload_bands"},
    "Qwen3-235B-A22B": {"N": 22e9, "A": 3080192.0, "Bkv": 192e3,
                         "filename": "roofline_Qwen3-235B-A22B_workload_bands"},
}
WORKLOADS = {
    "ShareGPT": {"filename": "sharegpt_effective_prefill.csv", "color": "#9467bd"},
    "FinQA": {"filename": "docfinqa_token_counts_all_splits.csv", "color": "#ff7f0e"},
    "Mooncake": {"filename": "mooncake_toolagent_effective_prefill.csv", "color": "#2ca02c"},
}


def load_requests(path: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["question_tokens", "context_tokens"])
    df = df.rename(columns={"question_tokens": "T", "context_tokens": "K"})
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["T"] > 0) & (df["K"] >= 0)]
    if df.empty:
        raise ValueError(f"No valid requests in {path}")
    return df.sample(n=min(n, len(df)), random_state=seed)


def request_ai(requests: pd.DataFrame, model: dict) -> tuple[np.ndarray, int]:
    """AI in FLOP/byte (K>0), from total F=T*[2N+A(K+T/2)], bytes=K*Bkv."""
    T = requests["T"].to_numpy(dtype=float)
    K = requests["K"].to_numpy(dtype=float)
    K = np.maximum(K, 1.0)
    valid = K > 0
    flops = T[valid] * (2.0 * model["N"] + model["A"] * (K[valid] + T[valid] / 2.0))
    bytes_moved = K[valid] * model["Bkv"]
    ai = flops / bytes_moved
    ai = ai[np.isfinite(ai) & (ai > 0)]
    if not len(ai):
        raise ValueError("No requests with positive K and finite arithmetic intensity")
    return ai, int(np.count_nonzero(~valid))


def diagonal_label(ax, xval: float, bw: float, label: str, color: str,
                   side: int, distance_pts: float = 13.0) -> None:
    """Align text with a PCIe slope, using symmetric offsets normal to the line.

    side=+1 labels above the diagonal; side=-1 labels below it.
    Labels for both PCIe generations can share the exact same xval, so their
    text begins at matching positions along their respective diagonals.
    """
    yval = bw * xval / 1000.0
    # Compute angle in *display* coordinates because log-log axes may not be
    # square: 45 degrees in data space is not necessarily 45 on the page.
    p1 = ax.transData.transform((xval, yval))
    p2 = ax.transData.transform((xval * 10.0, yval * 10.0))
    theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    print(theta)
    angle_deg = np.degrees(.78)

    # Perpendicular offset, in points, mirrored about each sloping line.
    dx_pts = -side * distance_pts * np.sin(theta)
    dy_pts = side * distance_pts * np.cos(theta)
    ax.annotate(label, xy=(xval, yval), xytext=(dx_pts, dy_pts),
                textcoords="offset points", color=color,
                rotation=angle_deg, rotation_mode="anchor",
                fontsize=18, fontweight="bold", ha="left", va="center",
                annotation_clip=False, zorder=5)


def plot_roofline(model_name: str, model: dict, requests: dict[str, pd.DataFrame],
                  outdir: Path, percentile_low: float, percentile_high: float) -> None:
    stats = {}
    for workload_name, df in requests.items():
        ai, omitted_k0 = request_ai(df, model)
        lo, med, hi = np.percentile(ai, [percentile_low, 50.0, percentile_high])
        stats[workload_name] = (lo, med, hi, len(ai), omitted_k0)
        print(f"{model_name:19s} | {workload_name:9s} | sampled={len(df):4d}, "
              f"finite_AI={len(ai):4d}, K=0 omitted={omitted_k0:4d} | "
              f"p{percentile_low:g}={lo:,.1f}, median={med:,.1f}, p{percentile_high:g}={hi:,.1f} FLOP/B")

    max_knee = max(1000.0 * p["compute"] / p["pcie"] for p in GPUS.values())
    largest = max(v[2] for v in stats.values())
    smallest = min(v[0] for v in stats.values())
    xmin = 10 ** np.floor(np.log10(min(1.0, max(smallest / 5.0, 1e-7))))
    xmax = 10 ** np.ceil(np.log10(max(1e6, largest * 2.0, 4.0 * max_knee)))
    ymin, ymax = 0.02, 12000

    fig, ax = plt.subplots(figsize=(6, 5), dpi=250)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # One translucent 5-95% (configurable) band and one dotted median per workload.
    for name, (p_lo, p_med, p_hi, *_rest) in stats.items():
        color = WORKLOADS[name]["color"]
        ax.axvspan(p_lo, p_hi, color=color, alpha=0.13, zorder=0.5, linewidth=0)
        ax.axvline(p_med, color=color, ls=":", lw=2.5, zorder=2.0)

    # Explicit bandwidth diagonals, not three redundant sloping rooflines.
    bw_colors = {"PCIe-5": "#ff7f0e", "PCIe-4": "#1f77b4"}
    # Extend each PCIe bandwidth diagonal until it meets the B200 compute ceiling.
    for bw_label, bw, knee in [
        ("PCIe-5", 64.0, 1000.0 * GPUS["B200"]["compute"] / 64.0),
        ("PCIe-4", 32.0, 1000.0 * GPUS["B200"]["compute"] / 32.0),
    ]:
        x = np.geomspace(xmin, knee, 400)
        y = bw * x / 1000.0
        ax.plot(x, y, lw=3.0, color=bw_colors[bw_label], solid_capstyle="round", zorder=3)

    # Horizontal compute ceilings begin at the corresponding machine-balance knee.
    for gpu_name, gpu in GPUS.items():
        knee = gpu["compute"] * 1000.0 / gpu["pcie"]
        ax.plot([knee, xmax], [gpu["compute"], gpu["compute"]],
                lw=3.0, color=gpu["color"], zorder=3, solid_capstyle="round")

    # Direct annotations like the original figure. No crossover guide lines.
    fig.canvas.draw()
    # Put both labels at the same horizontal anchor and mirror their
    # perpendicular offsets (PCIe-5 above, PCIe-4 below the slopes).
    label_x = max(xmin * 25, 25.0)
    diagonal_label(ax, label_x, 64.0, "PCIe-5 64 GB/s", bw_colors["PCIe-5"], side=+1)
    diagonal_label(ax, label_x, 32.0, "PCIe-4 32 GB/s", bw_colors["PCIe-4"], side=-1)
    # Hardware names directly above their horizontal ceilings.
    for gpu_name in ["B200", "H100", "A100"]:
        gpu = GPUS[gpu_name]
        knee = gpu["compute"] * 1000.0 / gpu["pcie"]
        label_x = 10 ** (np.log10(knee) + 0.28 * (np.log10(xmax) - np.log10(knee)))
        ax.text(label_x, gpu["compute"] * 1.02, gpu_name,
                color=gpu["color"], fontsize=17, weight="bold", va="bottom", zorder=5)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    from matplotlib.ticker import LogLocator
    
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(
            LogLocator(base=10, subs=(1.0,), numticks=20)
        )
        axis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10), numticks=100)
        )
    
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--",
            color="#999999", alpha=0.7, linewidth=0.9)
    ax.grid(True, which="minor", linestyle=":",
            color="#b3b3b3", alpha=0.33, linewidth=0.5)
    ax.set_axisbelow(True)

    # One legend entry per workload: shaded range and dotted median conveyed together.
    handles = [Line2D([0], [0], color=WORKLOADS[n]["color"], lw=2.5,
                      ls=":", marker="s", markersize=12,
                      markerfacecolor=WORKLOADS[n]["color"], markeredgecolor="none",
                      alpha=0.9, label=n) for n in WORKLOADS]
    # Place the workload legend outside, centered directly above the plotting area.
    ax.legend(handles=handles, fontsize=15, loc="lower center",
              bbox_to_anchor=(0.5, 1.015), frameon=False,
              ncol=3, borderaxespad=0.0,
            columnspacing=0.8,  # Default: 2.0
            handletextpad=0.4,  # Default: 0.8
            handlelength=1.2)  # Shorter legend symbols)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    
    base = outdir / model["filename"]
    png_path = Path(str(base) + ".png")
    pdf_path = Path(str(base) + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    
    plt.show()
    plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("./paper_plots"))
    parser.add_argument("--n", type=int, default=5000, help="Max sampled requests per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-low", type=float, default=5.0)
    parser.add_argument("--p-high", type=float, default=95.0)
    args = parser.parse_args()
    if args.n < 1 or not (0 <= args.p_low < 50 < args.p_high <= 100):
        parser.error("Require n>=1 and 0<=p-low<50<p-high<=100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sampled = {
        name: load_requests(args.data_dir / spec["filename"], args.n, args.seed)
        for name, spec in WORKLOADS.items()
    }
    for model_name, params in MODELS.items():
        plot_roofline(model_name, params, sampled, args.out_dir,
                      args.p_low, args.p_high)


if __name__ == "__main__":
    main()
