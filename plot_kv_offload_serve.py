import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
root_dir = "./kv_offload_serve"
gpu = 'h100'
output_dir = f"./paper_plots"
os.makedirs(output_dir, exist_ok=True)

# Font settings
plt.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
})

# Iterate over each creator folder
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    # Find subfolder with model name
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        continue
    model_folder = os.path.join(folder_path, subfolders[0])

    # Find all benchmark log files
    log_files = [f for f in os.listdir(model_folder) if f.startswith("benchmark_results") and f.endswith(".log")]
    if not log_files:
        print(f"Skipping {folder}, no benchmark logs found")
        continue

    # Parse all logs into a DataFrame
    records = []
    for log_file in log_files:
        kv_match = [int(s) for s in log_file.replace(".log","").split("_") if s.isdigit()]
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
                "std_ttft": data.get("std_ttft_ms", np.nan)
            })

    df = pd.DataFrame(records)
    if df.empty:
        continue

    # Normalize by KV=0 baseline for each prefill
    baseline = df[df["KV"] == 0].set_index("Prefill")["mean_ttft"]
    duration_baseline = df[df["KV"] == 0].set_index("Prefill")["duration"]
    df_plot = df.copy()
    df_plot['K/T'] = df_plot['KV'] / df_plot['Prefill']
    df_plot = df_plot.set_index("Prefill")
    df_plot["normalized_ttft"] = df_plot.apply(lambda row: row["mean_ttft"] / baseline.get(row.name, np.nan), axis=1)
    df_plot["normalized_duration"] = df_plot.apply(lambda row: row["duration"] / duration_baseline.get(row.name, np.nan), axis=1)
    df_plot["std_normalized"] = df_plot["std_ttft"] / baseline.reindex(df_plot.index).values
    kv0 = df_plot[df_plot['KV'] == 0]
    df_plot = df_plot[df_plot['KV'] != 0]

    # Prepare axes
    kv_sizes = sorted(df_plot["KV"].unique())
    prefill_values = sorted(df_plot.index.unique())
    prefill_values = [x for x in prefill_values if x <= 2048 and x >32]
    
    x = np.arange(len(kv_sizes))
    width = 0.8 / len(prefill_values)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for i, prefill in enumerate(prefill_values):
        if prefill <= 2048:
            df_prefill = df_plot[df_plot.index == prefill].set_index("KV")
            heights = df_prefill["normalized_ttft"].reindex(kv_sizes).values - 1
            errors = df_prefill["std_normalized"].reindex(kv_sizes).values
            ax.bar(
                x + i*width,
                heights,
                width,
                label=f"Prefill={prefill}",
                edgecolor="black",
                yerr=errors,
                capsize=4,
                linewidth=1
            )


    # Labels and formatting
    ax.set_xticks(x + width*(len(prefill_values)-1)/2)
    ax.set_xticklabels([str(k) for k in kv_sizes], rotation=30, ha="center")
    ax.set_xlabel("Offloaded KV")
    ax.set_ylabel("$P_{OH}$")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    #plt.title(folder)
    # Tighten layout but preserve fixed size
    fig.tight_layout(rect=[0, 0, 1, 1])  # same rect across all plots

    out_path = os.path.join(output_dir, f"{folder}_normalized_ttft.png")
    
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()
    print(f"✅ Saved plot: {out_path}")
    df_plot = pd.concat([df_plot, kv0])

print("✅ All grouped bar charts saved.")
