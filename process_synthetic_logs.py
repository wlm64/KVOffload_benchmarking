import re
import os
import glob
import sys
import pandas as pd
import json

# --- Get folder path from command-line argument ---
folder = "logs"

if not os.path.isdir(folder):
    print(f"❌ Folder not found: {folder}")
    sys.exit(1)

# --- Find all scheduler logs ---
log_files = glob.glob(os.path.join(folder, "*.log"))
if not log_files:
    print(f"No scheduler logs found in {folder}")
    sys.exit(0)

results_sched = []

for file in log_files:
    if "benchmark_results" not in file: 
        tag_match = re.search(r"_(\w+)\.log$", os.path.basename(file))  
        tag = tag_match.group(1) if tag_match else os.path.basename(file)
        if "il" not in tag:
            tag += '_il_0'
        
        rate = tag.split('_')[2]
        il = tag.split('_')[-1]
        
        total_tokens = []
        total_tokens_kv = []
    
        with open(file, "r") as f:
            for line in f:
                match_tokens = re.search(r"Total tokens scheduled this iteration:\s*(\d+)", line)
                match_tokens_kv = re.search(r"Total tokens \+ KV scheduled this iteration:\s*(\d+)", line)
                if match_tokens:
                    total_tokens.append(int(match_tokens.group(1)))
                if match_tokens_kv:
                    total_tokens_kv.append(int(match_tokens_kv.group(1)))
    
        avg_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0
        avg_tokens_kv = sum(total_tokens_kv) / len(total_tokens_kv) if total_tokens_kv else 0
        sum_tokens = sum(total_tokens)
        sum_tokens_kv = sum(total_tokens_kv)
    
        results_sched.append({
            "Tag": tag,
            "RPS": rate,
            "IL": il,
            #"TPS": int(rate) * int(il),
            "Average Tokens Scheduled": round(avg_tokens, 2),
            "Average Tokens+KV Scheduled": round(avg_tokens_kv, 2),
            "Total Tokens Scheduled": sum_tokens,
            "Total Tokens+KV Scheduled": sum_tokens_kv,
        })

# --- Save scheduler summary ---
df_sched = pd.DataFrame(results_sched)
df_sched.to_csv(os.path.join(folder, "scheduler_summary.csv"), index=False)
print(f"✅ Saved scheduler_summary.csv")

# --- Power parsing ---
pattern = os.path.join(folder, "power_util_rate_*.csv")
files = glob.glob(pattern)
results_power = []

for file in files:
    tag_match = re.search(r"power_util_rate_(.*)\.csv", os.path.basename(file))
    tag = tag_match.group(1) if tag_match else "unknown"
    if "il" not in tag:
        tag += '_il_0'
    rate = tag.split('_')[0]
    il = tag.split('_')[-1]

    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    df["utilization.gpu [%]"] = (
        df["utilization.gpu [%]"].astype(str).str.replace("%", "").astype(float)
    )
    df["power.draw [W]"] = (
        df["power.draw [W]"].astype(str).str.replace("W", "").astype(float)
    )

    df = df[df["utilization.gpu [%]"] > 0]

    if len(df) == 0:
        avg_power = 0
        max_power = 0
        pct_above_270 = 0
    else:
        avg_power = df["power.draw [W]"].mean()
        max_power = df["power.draw [W]"].max()
        pct_above_270 = (df["power.draw [W]"] > 270).mean() * 100

    results_power.append({
        "Tag": tag,
        "RPS": rate,
        "IL": il,
        "Average Power (W)": round(avg_power, 2),
        "Max Power (W)": round(max_power, 2),
        #"Pct Above 270W (%)": round(pct_above_270, 2),
    })

df_power = pd.DataFrame(results_power)
df_power.to_csv(os.path.join(folder, "power_summary.csv"), index=False)
print(f"✅ Saved power_summary.csv")
