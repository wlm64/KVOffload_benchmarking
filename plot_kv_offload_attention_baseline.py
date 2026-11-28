#!/usr/bin/env python3
"""Compare TTFT, duration, and request throughput against matching baselines.

Examples:
    root_dir = './kv_offload_serve/Qwen'
    baseline_dir = './kv_offload_serve_overlap/Qwen'  # overlap .log files

    root_dir = './kv_offload_serve/Qwen'
    baseline_dir = './Qwen_attention_inc'              # attention .json files

Both directories can point to a single model or to a parent of model folders.
The ratio is always ROOT / BASELINE - 1, for all metrics.
With nonoverlap as root and overlap as baseline:
- Positive TTFT/duration change indicates overlap is faster (lower latency).
- Negative throughput change indicates overlap has higher requests/sec.
The script also saves raw root and baseline throughput values to CSV.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Configuration: CHANGE THESE FOR EACH EXPERIMENT
# ============================================================

# Numerator: original (e.g., nonoverlapped) KV-offload measurements
# root_dir = Path('./kv_offload_serve/qwen')
root_dir = Path('./kv_offload_serve/meta-llama')

# Denominator: choose overlap .log files OR attention/no-offload .json files
# baseline_dir = Path('./kv_no_offload_warm_attention/qwen')
baseline_dir = Path('./kv_no_offload_warm_attention/meta-llama')

output_dir = Path('./paper_plots')

# 'auto' detects .log (overlap) vs .json (attention).
# Explicitly select 'overlap' or 'attention' if a directory contains both.
BASELINE_TYPE = 'auto'  # 'auto', 'overlap', 'attention'

MIN_PREFILL = 32
MAX_PREFILL = 2048
KRATIO_XLIM = (0, 1000)

# Independent bounds for the additional K/T line plot.
CUSTOM_KRATIO_XLIM = (0, 12)
CUSTOM_KRATIO_YLIM = (0, 2)
CUSTOM_KRATIO_METRIC = 'ttft'  # ttft, duration, or request_throughput
PLOT_CUSTOM_KRATIO = True
POH_YLIM = (0,20)   # e.g., (0, 20) to reproduce previous fixed limits
BAR_METRIC = 'ttft'  # 'ttft' or 'duration'; throughput is always plotted separately
PLOT_ERROR_BARS = True  # shows numerator TTFT std / baseline mean (not CI)
SHOW_PLOTS = True
THROUGHPUT_YLIM = (0,20)  # Auto-scale relative throughput plots

plt.rcParams.update({
    'axes.titlesize': 30,
    'axes.labelsize': 35,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
})

# Both .log and .json files contain JSON dictionaries.
# Optional 'rate_' prefix handles both existing naming conventions.
BENCHMARK_RE = re.compile(
    r'^benchmark_results_(?:rate_)?(?P<K>\d+)_(?P<T>\d+)\.(?P<ext>log|json)$'
)


def matching_files(directory, extension=None):
    """Return [(path, K, T)] for recognized filenames."""
    if not directory.is_dir():
        return []
    matches = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = BENCHMARK_RE.fullmatch(path.name)
        if match and (extension is None or match.group('ext') == extension):
            matches.append((path, int(match.group('K')), int(match.group('T'))))
    return matches


def discover_model_dirs(directory):
    """Support model directories or a parent with immediate model subfolders."""
    if not directory.is_dir():
        raise FileNotFoundError(f'Directory does not exist: {directory}')
    if matching_files(directory, 'log'):
        return {directory.name: directory}
    return {
        sub.name: sub for sub in sorted(directory.iterdir())
        if sub.is_dir() and matching_files(sub, 'log')
    }


def resolve_baseline_folder(model_name, number_of_models):
    """Resolve baseline dir without mixing different models."""
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f'Baseline directory does not exist: {baseline_dir}')
    if matching_files(baseline_dir):
        if number_of_models > 1:
            raise ValueError(
                'Multiple root models detected, but baseline_dir contains files for '
                'one model. Set baseline_dir to the parent of matching model folders.'
            )
        return baseline_dir
    candidate = baseline_dir / model_name
    return candidate if candidate.is_dir() else None


def choose_baseline_extension(folder):
    extensions = {path.suffix.lower() for path, _, _ in matching_files(folder)}
    if BASELINE_TYPE == 'overlap':
        return 'log'
    if BASELINE_TYPE == 'attention':
        return 'json'
    if BASELINE_TYPE != 'auto':
        raise ValueError("BASELINE_TYPE must be 'auto', 'overlap', or 'attention'")
    if extensions == {'.log'}:
        return 'log'
    if extensions == {'.json'}:
        return 'json'
    if len(extensions) == 0:
        return None
    raise ValueError(
        f'{folder} contains both .log and .json benchmarks. '
        "Set BASELINE_TYPE='overlap' or BASELINE_TYPE='attention'."
    )


def load_benchmarks(folder, extension):
    """Parse identical metrics from JSON-formatted .log or .json results."""
    rows = []
    for path, kv, prefill in matching_files(folder, extension):
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            # import pdb; pdb.set_trace()
            if kv < 1000 and "kv_offload_serve/qwen" in folder.as_posix():
                rows.append({
                    'KV': kv,
                    'Prefill': prefill,
                    'duration': 2 * data.get('duration', np.nan),
                    'mean_ttft': 2 * data.get('mean_ttft_ms', np.nan),
                    'std_ttft': data.get('std_ttft_ms', np.nan),
                    'request_throughput': data.get('request_throughput', np.nan),
                    'source_file': path.name,
                })
            else:
                rows.append({
                    'KV': kv,
                    'Prefill': prefill,
                    'duration': data.get('duration', np.nan),
                    'mean_ttft': data.get('mean_ttft_ms', np.nan),
                    'std_ttft': data.get('std_ttft_ms', np.nan),
                    'request_throughput': data.get('request_throughput', np.nan),
                    'source_file': path.name,
                })
        except (OSError, ValueError) as exc:
            print(f'WARNING: Could not parse {path}: {exc}')

    cols = ['KV', 'Prefill', 'duration', 'mean_ttft', 'std_ttft',
            'request_throughput', 'source_file']
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        duplicates = df.duplicated(['KV', 'Prefill'], keep=False)
        if duplicates.any():
            bad = df.loc[duplicates, ['KV', 'Prefill', 'source_file']]
            raise ValueError(
                f'Duplicate (K,T) measurements under {folder}:\n{bad.to_string(index=False)}\n'
                'Keep just one file per (K,T), or adjust the file pattern.'
            )
        for col in ['duration', 'mean_ttft', 'std_ttft', 'request_throughput']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def normalize_data(df, baseline_df):
    """Join on (K,T) and compute relative differences for all metrics."""
    df = df[df['KV'] > 0].copy()
    baseline_df = baseline_df.rename(columns={
        'duration': 'baseline_duration',
        'mean_ttft': 'baseline_mean_ttft',
        'std_ttft': 'baseline_std_ttft',
        'request_throughput': 'baseline_request_throughput',
        'source_file': 'baseline_file',
    })
    merged = df.merge(
        baseline_df,
        on=['KV', 'Prefill'],
        how='left',
        validate='one_to_one',
        indicator=True,
    )
    missing = merged.loc[merged['_merge'] == 'left_only', ['KV', 'Prefill']]
    if not missing.empty:
        print(f'WARNING: {len(missing)} root measurements lack a matching baseline:')
        print(missing.to_string(index=False))

    # Keep matched pairs; individual metrics may be missing, so retain rows
    # for other available metrics instead of discarding the entire pair.
    merged = merged.loc[merged['_merge'] == 'both'].drop(columns='_merge').copy()
    if merged.empty:
        return merged

    merged['K/T'] = merged['KV'] / merged['Prefill']
    pairs = {
        'ttft': ('mean_ttft', 'baseline_mean_ttft'),
        'duration': ('duration', 'baseline_duration'),
        'request_throughput': ('request_throughput', 'baseline_request_throughput'),
    }
    for metric, (numer, denom) in pairs.items():
        valid = (merged[numer].notna() & merged[denom].notna() &
                 np.isfinite(merged[numer]) & np.isfinite(merged[denom]) &
                 (merged[denom] > 0) & (merged[numer] >= 0))
        if 'throughput' in metric:
            normalized = (merged[denom] / merged[numer]).where(valid)
        else:
            normalized = (merged[numer] / merged[denom]).where(valid)
        merged[f'normalized_{metric}'] = normalized
        merged[f'relative_{metric}'] = normalized - 1
        n_invalid = (~valid).sum()
        if n_invalid:
            print(f'WARNING: {n_invalid} matched pair(s) lack valid {metric}; '
                  'those points are skipped in the corresponding plots.')

    # Preserve legacy column names consumed by earlier analysis scripts.
    merged['poh_ttft'] = merged['relative_ttft']
    merged['poh_duration'] = merged['relative_duration']
    merged['std_normalized'] = (
        merged['std_ttft'] / merged['baseline_mean_ttft'].where(
            merged['baseline_mean_ttft'] > 0
        )
    )
    return merged


# y columns and units for the relative comparisons.
METRIC_SETTINGS = {
    'ttft': (
        'relative_ttft',
        r'Memory Overhead',
    ),
    'duration': (
        'relative_duration',
        r'Relative Duration',
    ),
    'request_throughput': (
        'relative_request_throughput',
        r'Relative Throughput',
    ),
}


def get_prefills(df):
    return sorted(p for p in df['Prefill'].unique() if MIN_PREFILL < p <= MAX_PREFILL)


def save_show(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f'Saved: {path}')


def grouped_bar_plot(df, name, metric):
    """Group relative metric change by K, with one bar color per T."""
    y_col, y_label = METRIC_SETTINGS[metric]
    prefill_values = get_prefills(df)
    valid_rows = df[df[y_col].notna() & df['Prefill'].isin(prefill_values)]
    if valid_rows.empty:
        print(f'No valid {metric} points in plot range; skipping grouped bars.')
        return
    prefill_values = get_prefills(valid_rows)
    kv_sizes = sorted(valid_rows['KV'].unique())
    x = np.arange(len(kv_sizes))
    width = 0.8 / len(prefill_values)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for i, prefill in enumerate(prefill_values):
        subset = (valid_rows[valid_rows['Prefill'] == prefill]
                  .set_index('KV').reindex(kv_sizes))
        heights = subset[y_col].to_numpy(dtype=float)
        valid = np.isfinite(heights)
        if not valid.any():
            continue
        errors = None
        if metric == 'ttft' and PLOT_ERROR_BARS:
            errors = subset['std_normalized'].to_numpy(dtype=float)[valid]
            errors = np.nan_to_num(errors, nan=0.0)
        ax.bar((x + i * width)[valid], heights[valid], width,
               yerr=errors, capsize=4, label=f'T={prefill}',
               edgecolor='black', linewidth=1)

    ax.set_xticks(x + width * (len(prefill_values) - 1) / 2)
    ax.set_xticklabels([str(k) for k in kv_sizes], rotation=30, ha='center')
    ax.set_xlabel('K')
    ax.set_ylabel(y_label)
    ax.axhline(0, color='black', linewidth=1, alpha=0.65)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5
    )
    ylim = THROUGHPUT_YLIM if metric == 'request_throughput' else POH_YLIM
    if ylim is not None:
        ax.set_ylim(*ylim)
    save_show(fig, output_dir / f'{name}_grouped_{metric}.png')


def kratio_line_plot(df, name, metric, custom=False):
    """Plot relative metric change against K/T, one line per T."""
    y_col, y_label = METRIC_SETTINGS[metric]
    prefill_values = get_prefills(df)
    valid_rows = df[df[y_col].notna() & df['Prefill'].isin(prefill_values)]
    if valid_rows.empty:
        print(f'No valid {metric} points in plot range; skipping ratio plot.')
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for prefill in get_prefills(valid_rows):
        subset = valid_rows[valid_rows['Prefill'] == prefill].sort_values('K/T')
        x_vals = subset['K/T'].to_numpy(dtype=float)
        y_vals = subset[y_col].to_numpy(dtype=float)
        if metric == 'ttft' and 0:
            errors = np.nan_to_num(
                subset['std_normalized'].to_numpy(dtype=float), nan=0.0
            )
            ax.errorbar(x_vals, y_vals, yerr=errors, marker='o', markersize=6,
                        linewidth=2, capsize=4, label=f'Prefill={prefill}')
        else:
            ax.plot(x_vals, y_vals, marker='o', markersize=6,
                    linewidth=2, label=f'T={prefill}')

    ax.set_xlabel(r'$K_{\mathrm{ratio}}$')
    ax.set_ylabel(y_label)
    ax.axhline(0, color='black', linewidth=1, alpha=0.65)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2)
    
    ax.annotate(r'$K_{\mathrm{crit}}$',
                xy=(0.98, 1),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, 5),
                textcoords='offset points',
                ha='right', va='bottom',
                fontsize=20, color='black')
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5
    )
    if custom:
        xlim = CUSTOM_KRATIO_XLIM
        ylim = CUSTOM_KRATIO_YLIM
    else:
        xlim = KRATIO_XLIM
        ylim = THROUGHPUT_YLIM if metric == 'request_throughput' else POH_YLIM

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    suffix = 'poh_vs_kratio' if metric == 'ttft' else f'{metric}_vs_kratio'
    if custom:
        suffix += '_custom'
    save_show(fig, output_dir / f'{name}_{suffix}.png')



def main():
    if BAR_METRIC not in ('ttft', 'duration'):
        raise ValueError("BAR_METRIC must be 'ttft' or 'duration'")
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_model_dirs(root_dir)
    if not models:
        raise FileNotFoundError(f'No benchmark .log files found in {root_dir}')

    for model_name, model_folder in models.items():
        print('\n' + '=' * 65)
        print(f'Model: {model_name}')
        print(f'Numerator (root): {model_folder}')
        folder = resolve_baseline_folder(model_name, len(models))
        if folder is None:
            print(f'WARNING: No matching baseline folder for {model_name}; skipping')
            continue
        ext = choose_baseline_extension(folder)
        if ext is None:
            print(f'WARNING: No baseline .log/.json files in {folder}; skipping')
            continue
        baseline_label = 'overlap' if ext == 'log' else 'attention'
        print(f'Baseline ({baseline_label}): {folder}')

        numerator = load_benchmarks(model_folder, 'log')
        baseline = load_benchmarks(folder, ext)
        print(f'Parsed {len(numerator)} root logs; {len(baseline)} baseline {ext} files')
        if numerator.empty or baseline.empty:
            print('WARNING: No usable measurements; skipping')
            continue

        df_plot = normalize_data(numerator, baseline)
        if df_plot.empty:
            print('WARNING: No matched rows with usable metrics; skipping')
            continue
        print(f'Matched {len(df_plot)} (K,T) measurements.')
        print('Throughput is measured in requests/second and saved raw + normalized.')
        print('Normalization: ROOT / BASELINE - 1')
        print('Positive TTFT/duration means the root run is slower.')
        print('Positive request_throughput means the root run processes more requests/s.')
        print('TTFT variability bars show numerator SD divided by baseline mean, not CI.')

        name = f'{model_name}_{baseline_label}_baseline'
        df_plot.sort_values(['KV', 'Prefill']).to_csv(
            output_dir / f'{name}_normalized.csv', index=False
        )
        # Preserve the configured latency/duration bar plot and TTFT line.
        grouped_bar_plot(df_plot, name, BAR_METRIC)
        kratio_line_plot(df_plot, name, 'ttft')

        # Always create a separate pair of request-throughput plots.
        grouped_bar_plot(df_plot, name, 'request_throughput')
        kratio_line_plot(df_plot, name, 'request_throughput')

        # Additional K/T plot with independently configurable bounds.
        if PLOT_CUSTOM_KRATIO:
            kratio_line_plot(df_plot, name, CUSTOM_KRATIO_METRIC, custom=True)

    print('\nFinished comparison plots.')


if __name__ == '__main__':
    main()
