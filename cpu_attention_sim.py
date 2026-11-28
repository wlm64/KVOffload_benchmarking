import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams.update({
#     "axes.titlesize": 25,
#     "axes.labelsize": 30,
#     "xtick.labelsize": 25,
#     "ytick.labelsize": 25,
#     "legend.fontsize": 25,
# })

plt.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 20,
})

# ==========================================================
# Model parameters
# ==========================================================

nlayer = 94
dattn = 8192
dmodel = 4096

proj_flops = 2 * nlayer * dattn * dmodel

models = {
    "Llama-70B": {"size": 70, "kv": 0.0003},      # TB/token
    "DeepSeek-V3": {"size": 37, "kv": 0.00007},
    "Qwen3-235B": {"size": 22, "kv": 0.0002},
}

# ==========================================================
# Hardware
# ==========================================================

# # TB/s
PCIe_BW = 64
VRAM_BW = 8000
DRAM_BW = 300

# TFLOP/s
B200_FLOPS = 795
CPU_FLOPS = 18

# TB/s
# PCIe_BW = 32
# VRAM_BW = 300
# DRAM_BW = 100

# # TFLOP/s
# B200_FLOPS = 242
# CPU_FLOPS = 1.3

# ==========================================================
# Timing models
# ==========================================================

def gpu_time(P, K, kv_tb):

    pcie = K * kv_tb / PCIe_BW
    hbm = (P + K) * kv_tb / VRAM_BW

    attention = 2 * nlayer * (P + K) * dattn
    total_flops = (attention + proj_flops) * P

    compute = total_flops / (B200_FLOPS * 1e12)

    return pcie + hbm + compute


def cpu_time(P, K, kv_tb):

    pcie = P * kv_tb / PCIe_BW
    dram = (P + K) * kv_tb / DRAM_BW

    attention = 2 * nlayer * (P + K) * dattn
    total_flops = (attention + proj_flops) * P

    compute = total_flops / (CPU_FLOPS * 1e12)

    return pcie + dram + compute


def gpu_time_breakdown(P, K, kv_tb):

    pcie = K * kv_tb / PCIe_BW
    hbm = (P + K) * kv_tb / VRAM_BW

    attention = 2 * nlayer * (P + K) * dattn
    total_flops = (attention + proj_flops) * P

    compute = total_flops / (B200_FLOPS * 1e12)

    total = pcie + hbm + compute

    return pcie, hbm, compute, total


def cpu_time_breakdown(P, K, kv_tb):

    pcie = P * kv_tb / PCIe_BW
    dram = (P + K) * kv_tb / DRAM_BW

    attention = 2 * nlayer * (P + K) * dattn
    total_flops = (attention + proj_flops) * P

    compute = total_flops / (CPU_FLOPS * 1e12)

    total = pcie + dram + compute

    return pcie, dram, compute, total


# ==========================================================
# Plot 1
# ==========================================================

prefills = [1, 16, 64, 256, 1024]
reuse_tokens = np.logspace(0, 6, 200)

model = models["Qwen3-235B"]

plt.figure(figsize=(10,6))

curve_data = {}

for P in prefills:

    gpu = np.array([gpu_time(P, K, model["kv"]) for K in reuse_tokens])
    cpu = np.array([cpu_time(P, K, model["kv"]) for K in reuse_tokens])

    relative = cpu / gpu
    curve_data[P] = relative

    plt.plot(reuse_tokens, relative, label=f"T={P}")

# ----------------------------------------------------------
# Reference lines
# ----------------------------------------------------------

# GPU baseline
plt.axhline(1, color="black", linestyle="--", linewidth=1)

plt.text(
    reuse_tokens[-1],
    1.05,
    "GPU Baseline",
    ha="right",
    va="bottom",
    fontsize=20,
)

# Compute ratio
compute_ratio = B200_FLOPS / CPU_FLOPS

plt.axhline(
    compute_ratio,
    color="red",
    linestyle=":",
    linewidth=1.5,
)

plt.text(
    reuse_tokens[-1],
    compute_ratio * .6,
    "Compute Ratio",
    ha="right",
    va="bottom",
    fontsize=20,
    color="red",
)

# Memory ratio
memory_ratio = PCIe_BW / DRAM_BW

plt.axhline(
    memory_ratio,
    color="red",
    linestyle=":",
    linewidth=1.5,
)

plt.text(
    reuse_tokens[-1],
    memory_ratio * .6,
    "Memory Ratio",
    ha="right",
    va="bottom",
    fontsize=20,
    color="red",
)

# ----------------------------------------------------------
# ShareGPT annotation
# ----------------------------------------------------------

share_k = 11000

idx = np.argmin(np.abs(reuse_tokens - share_k))
share_y = curve_data[64][idx]

plt.plot(share_k, share_y, "kx", markersize=10, mew=2)
print(share_y)
plt.annotate(
    "ShareGPT",
    xy=(share_k + 3000, share_y + .3),
    xytext=(120000, share_y * 2),
    ha="center",
    va="center",
    arrowprops=dict(
        arrowstyle="->",
        relpos=(0.5, 0.5),   # arrow points to center of text
    ),
    fontsize=20,
)

# ----------------------------------------------------------
# NarrativeQA annotation
# ----------------------------------------------------------

narr_k = 100000

idx = np.argmin(np.abs(reuse_tokens - narr_k))
narr_y = curve_data[16][idx]
print(narr_y)
plt.plot(narr_k, narr_y, "kx", markersize=10, mew=2)

plt.annotate(
    "DocQnA",
    xy=(narr_k - 20000, narr_y - .1),
    xytext=(12000, narr_y * 0.5),
    ha="center",
    va="center",
    arrowprops=dict(
        arrowstyle="->",
        relpos=(0.5, 0.5),   # arrow points to center of text
    ),
    fontsize=20,
)

# ----------------------------------------------------------

plt.xscale("log")
plt.yscale("log")
plt.ylim(.1)

plt.xlabel("Reused KV")
plt.ylabel("$Attn_{ratio}$")

plt.grid(True, which="both", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("./CPU_attention_comparison_235B.png")
plt.show()


# ==========================================================
# Plot 2 (kappa ratio)
# ==========================================================

prefills = [1, 16, 64, 256, 1024]
reuse_tokens = np.logspace(0, 10, 200)

plt.figure(figsize=(8,6))

for P in prefills:

    gpu = np.array([gpu_time(P, K, model["kv"]) for K in reuse_tokens])
    cpu = np.array([cpu_time(P, K, model["kv"]) for K in reuse_tokens])

    relative = cpu / gpu

    plt.plot(reuse_tokens / P, relative, label=f"P={P}")

# GPU baseline
plt.axhline(1, color="black", linestyle="--", linewidth=1)

plt.text(
    reuse_tokens[-1] / prefills[0],
    1.05,
    "GPU Baseline",
    ha="right",
    va="bottom",
    fontsize=11,
)

# Compute ratio
plt.axhline(
    compute_ratio,
    color="red",
    linestyle=":",
    linewidth=1.5,
)

plt.text(
    reuse_tokens[-1] / prefills[0],
    compute_ratio * 1.03,
    "Compute Ratio",
    ha="right",
    va="bottom",
    fontsize=11,
    color="red",
)

# Memory ratio
plt.axhline(
    memory_ratio,
    color="red",
    linestyle=":",
    linewidth=1.5,
)

plt.text(
    reuse_tokens[-1] / prefills[0],
    memory_ratio * 1.03,
    "Memory Ratio",
    ha="right",
    va="bottom",
    fontsize=11,
    color="red",
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$\kappa_{\mathrm{ratio}}$")
plt.ylabel("Relative Attention Time (CPU / GPU)")

plt.grid(True, which="both", alpha=0.3)
plt.legend(title="Prefill Tokens")

plt.tight_layout()
plt.show()


# ==========================================================
# Generate CSV
# ==========================================================

rows = []

prefills = [1, 16, 64, 256, 1024]
reuse_tokens = np.logspace(0, 10, 200)

for P in prefills:
    for K in reuse_tokens:

        gpu_pcie, gpu_hbm, gpu_compute, gpu_total = gpu_time_breakdown(
            P, K, model["kv"]
        )

        cpu_pcie, cpu_dram, cpu_compute, cpu_total = cpu_time_breakdown(
            P, K, model["kv"]
        )

        rows.append({
            "P_prefill_tokens": P,
            "K_reused_tokens": K,
            "kappa_K_over_P": K / P,

            "GPU_PCIe_ms": gpu_pcie * 1000,
            "GPU_HBM_ms": gpu_hbm * 1000,
            "GPU_compute_ms": gpu_compute * 1000,
            "GPU_total_ms": gpu_total * 1000,

            "CPU_PCIe_ms": cpu_pcie * 1000,
            "CPU_DRAM_ms": cpu_dram * 1000,
            "CPU_compute_ms": cpu_compute * 1000,
            "CPU_total_ms": cpu_total * 1000,

            "CPU_GPU_ratio": cpu_total / gpu_total,
        })

df = pd.DataFrame(rows)

df.to_csv(
    "qwen3_235B_attention_breakdown.csv",
    index=False,
)

print(df.head())
print(f"Saved {len(df)} rows")