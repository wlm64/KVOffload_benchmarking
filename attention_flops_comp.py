# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:40:29 2026

@author: Willie
"""


#!/usr/bin/env python3
"""
Compare 2N FLOPs/token against context-aware FLOPs for:
- LLaMA-3.1-70B
- LLaMA-3.1-405B
- Qwen3-30B-A3B
- Qwen3-235B-A22B
- DeepSeek-V3

Requirements:
    pip install numpy matplotlib

Run:
    python plot_context_scaling_flops.py
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    n_active_b: float
    q_heads: int
    qk_dim_per_head: int
    value_dim_per_head: int
    kv_heads: int

    @property
    def d_attn(self):
        # FLOP-equivalent attention width for conventional attention.
        # Query and value widths can differ for MLA.
        return self.q_heads * (
            self.qk_dim_per_head + self.value_dim_per_head
        ) / 2.0

    @property
    def two_n(self):
        return 2.0 * self.n_active_b * 1e9


MODELS = (
    Model("LLaMA-3.1-70B", 80, 70.0, 64, 128, 128, 8),
    Model("LLaMA-3.1-405B", 126, 405.0, 128, 128, 128, 8),
    Model("Qwen3-30B-A3B", 48, 3.3, 32, 128, 128, 4),
    Model("Qwen3-235B-A22B", 94, 22.0, 64, 128, 128, 4),
    # DeepSeek MLA, projected attention approximation
    Model("DeepSeek-V3", 61, 37.0, 128, 192, 128, 128),
)


def configure_models(mla_mode):
    if mla_mode == "projected":
        return MODELS

    # Exploratory absorbed-MLA configuration.
    # Not an exact FLOP model for optimized MLA kernels.
    return MODELS[:-1] + (
        Model("DeepSeek-V3 (absorbed)",
              61, 37.0, 128, 576, 512, 1),
    )


def flops_kaplan(model, total_context):
    """
    Full causal prefill, averaged across all context tokens.

    F(L) = 2N + 2 * layers * L * d_attn
    """
    return (
        model.two_n
        + 2.0 * model.layers * total_context * model.d_attn
    )


def flops_reused_prefix(model, total_context, new_tokens):
    """
    Prefill with a reused KV prefix.

    L = K + T
    K = cached tokens
    T = newly computed tokens

    F(K,T) = 2N
           + 4 * layers * d_attn * (K + T/2)

    Here T = min(new_tokens, L).
    """
    t = np.minimum(float(new_tokens), total_context)
    k = total_context - t

    return (
        model.two_n
        + 4.0 * model.layers * model.d_attn
        * (k + t / 2.0)
    )


def kfmt(x, _pos):
    if x == 0:
        return "0"
    return f"{x / 1024:.1f}K".replace(".0K", "K")


def setup_ax(ax, max_context):
    ax.set_xlim(0, max_context)
    ax.set_xticks(np.linspace(0, max_context, 5))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.grid(True, linestyle=":", alpha=0.35)


def save_fig(fig, outdir, stem):
    for suffix in ["png"]:
        fig.savefig(
            outdir / f"{stem}.{suffix}",
            dpi=220,
            bbox_inches="tight"
        )
    plt.close(fig)


def plot_absolute(models, x, outdir):
    """Plot absolute GFLOPs/token for each model."""

    fig, axs = plt.subplots(
        2, 3, figsize=(13.8, 7.4),
        constrained_layout=True
    )

    for ax, model in zip(axs.flat, models):

        ax.plot(
            x,
            flops_kaplan(model, x) / 1e9,
            linewidth=2.2,
            label=r"$2N+2n_{layer}L d_{attn}$"
        )

        ax.axhline(
            model.two_n / 1e9,
            linestyle="--",
            linewidth=1.8,
            label=r"Original: $2N$"
        )

        ax.set_title(model.name, fontsize=11, weight="bold")
        ax.set_xlabel("Total sequence length")
        ax.set_ylabel("GFLOPs / token")
        ax.set_ylim(bottom=0)

        setup_ax(ax, int(x[-1]))
        ax.legend(
            loc="upper left",
            fontsize=8,
            frameon=False
        )

    axs.flat[-1].axis("off")

    fig.suptitle(
        "Context-dependent attention FLOPs vs. 2N "
        "(full causal prefill)",
        fontsize=15
    )

    save_fig(fig, outdir, "flops_vs_context")


def plot_ratios(models, x, outdir):
    """Plot FLOP increase relative to the original 2N model."""

    fig, ax = plt.subplots(
        figsize=(10.2, 5.7),
        constrained_layout=True
    )

    for model in models:
        ratio = flops_kaplan(model, x) / model.two_n

        ax.plot(
            x,
            ratio,
            linewidth=2.1,
            label=model.name
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Original 2N"
    )

    ax.set_ylabel("Extended FLOPs / original 2N")
    ax.set_xlabel("Total sequence length")
    ax.set_ylim(bottom=0.9)

    setup_ax(ax, int(x[-1]))
    ax.legend(loc="upper left", frameon=False, ncol=2)

    ax.set_title(
        "How much does 2N underestimate FLOPs per token?"
    )

    save_fig(fig, outdir, "flops_relative_to_2N")


def plot_reused(models, x, new_tokens, outdir):
    """Compare full prefill against reused-prefix prefill."""

    fig, axs = plt.subplots(
        2, 3, figsize=(13.8, 7.4),
        constrained_layout=True
    )

    for ax, model in zip(axs.flat, models):

        full = flops_kaplan(model, x) / model.two_n

        reused = (
            flops_reused_prefix(model, x, new_tokens)
            / model.two_n
        )

        ax.plot(
            x,
            full,
            linewidth=2,
            label="Full prefill (Kaplan)"
        )

        ax.plot(
            x,
            reused,
            linestyle="-.",
            linewidth=2,
            label=f"Reused prefix (T <= {new_tokens:,})"
        )

        ax.axhline(
            1,
            linestyle="--",
            linewidth=1.3,
            label="Original 2N"
        )

        ax.set_title(model.name, fontsize=11, weight="bold")
        ax.set_xlabel("Total sequence length (K + T)")
        ax.set_ylabel("FLOPs per NEW token / 2N")
        ax.set_ylim(bottom=0.9)

        setup_ax(ax, int(x[-1]))
        ax.legend(
            loc="upper left",
            fontsize=8,
            frameon=False
        )

    axs.flat[-1].axis("off")

    fig.suptitle(
        "Full prefill vs. reused-prefix prefill: "
        "context attention cost",
        fontsize=15
    )

    save_fig(fig, outdir, f"flops_reused_prefix_new_tokens_{new_tokens}")


def plot_reused_2n(models, x, new_tokens, outdir,
                   show_legend=True, ymax=None):
    """Paper-ready FLOP comparison, 5x4 inches, 600 DPI."""

    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

    # Consistent model appearance across figures
    styles = {
        "LLaMA-3.1-70B": ("LLama 70B", "tab:blue", "-"),
        "LLaMA-3.1-405B": ("LLama 405B", "tab:orange", "--"),
        "Qwen3-30B-A3B": ("Qwen 30B", "tab:green", "-."),
        "Qwen3-235B-A22B": ("Qwen 235B", "tab:red", ":"),
        "DeepSeek-V3": ("DeepSeek V3", "tab:purple", "-"),
        "DeepSeek-V3 (absorbed)": (
            "DeepSeek V3", "tab:purple", "-"
        ),
    }

    for model in models:
        ratio = (
            flops_reused_prefix(model, x, new_tokens)
            / model.two_n
        )

        label, color, linestyle = styles[model.name]

        ax.plot(
            x,
            ratio,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            label=label,
        )

    # 2N reference line
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=0,
    )

    max_context = int(x[-1])
    
    # Calculate tick spacing, rounded to nearest 1,000
    tick_step = max(1000, round(max_context / 4 / 1000) * 1000)
    
    ticks = np.arange(0, max_context + 1, tick_step)
    
    ax.set_xticks(ticks)
    ax.set_xlim(0, max_context)
    
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, pos: "0" if val == 0 else f"{val/1000:.0f}K"
        )
    )

    # Y-axis
    ax.set_ylim(
        0.8,
        ymax if ymax is not None else None
    )

    ax.set_xlabel("Prompt Length", fontsize=20)
    ax.set_ylabel(r"Normalized FLOPs", fontsize=20)

    ax.tick_params(
        axis="both",
        labelsize=20,
        width=1.2,
        length=5,
    )

    # Light horizontal grid
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    # Compact legend inside plot
    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=14,
            ncol=2,
            frameon=False,
            columnspacing=0.8,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.3,
        )

    fig.tight_layout()

    # Save at the actual requested DPI and canvas size
    out_path = (
        outdir
        / f"reused_flops_relative_to_2N_new_tokens_{new_tokens}.png"
    )
    
    plt.ylim(None,5)

    fig.savefig(out_path, dpi=600)
    plt.show()
    plt.close(fig)


def print_summary(models):
    print(
        f"{'Model':25} {'layers':>7} "
        f"{'d_attn':>9} {'2N (GF)':>10} "
        f"{'at 32K':>10} {'at 128K':>10} "
        f"{'equal at':>12}"
    )

    print("-" * 91)

    for m in models:

        ratio32 = float(
            flops_kaplan(m, np.array([32768]))[0]
            / m.two_n
        )

        ratio128 = float(
            flops_kaplan(m, np.array([131072]))[0]
            / m.two_n
        )

        equality_context = (
            m.two_n / (2.0 * m.layers * m.d_attn)
        )

        print(
            f"{m.name:25} {m.layers:7d} "
            f"{m.d_attn:9,.0f} "
            f"{m.two_n/1e9:10.1f} "
            f"{ratio32:9.2f}x "
            f"{ratio128:9.2f}x "
            f"{equality_context:11,.0f}"
        )

def print_normalized_flops(models, prompt_length, new_tokens):
    """
    Print normalized FLOPs (F / 2N) for each model
    at a specified total prompt length.
    """

    print(f"\nPrompt Length: {prompt_length:,}")
    print(f"New Tokens: {min(new_tokens, prompt_length):,}")
    print(f"Reused KV Tokens: {max(0, prompt_length - new_tokens):,}\n")

    print(
        f"{'Model':25} "
        f"{'Full Prefill / 2N':>20} "
        f"{'Reused Prefix / 2N':>22}"
    )
    print("-" * 70)

    for model in models:
        full_ratio = (
            flops_kaplan(model, prompt_length)
            / model.two_n
        )

        reused_ratio = (
            flops_reused_prefix(model, prompt_length, new_tokens)
            / model.two_n
        )

        print(
            f"{model.name:25} "
            f"{full_ratio:20.4f} "
            f"{reused_ratio:22.4f}"
        )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-context",
        type=int,
        default=32000
    )

    parser.add_argument(
        "--new-tokens",
        type=int,
        default=64
    )

    parser.add_argument(
        "--mla-mode",
        choices=["projected", "absorbed"],
        default="projected"
    )

    parser.add_argument(
        "--outdir",
        default="context_flops_plots"
    )

    args = parser.parse_args()

    # if args.max_context <= 0 or args.new_tokens <= 0:
    #     parser.error(
    #         "--max-context and --new-tokens must be positive"
    #     )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = configure_models(args.mla_mode)

    x = np.linspace(1, args.max_context, 600)

    print_summary(models)
    
    print_normalized_flops(
        models,
        215000,
        15
    )

    plot_absolute(models, x, outdir)
    plot_ratios(models, x, outdir)
    plot_reused(models, x, args.new_tokens, outdir)
    plot_reused_2n(models, x, args.new_tokens, outdir)

    print(f"\nSaved plots in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
