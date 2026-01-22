mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Measured H100 peaks
PEAK_TFLOPS = 465.30
PEAK_BW = 1888.41  # GB/s

# Load roofline-ready CSV
df = pd.read_csv("h100_roofline_ready.csv")

# Arithmetic intensity range
ai = np.logspace(-3, 4, 1000)

# Roofs
bw_roof = PEAK_BW * ai / 1000  # Convert GB/s * FLOP/Byte → TFLOPs
compute_roof = np.full_like(ai, PEAK_TFLOPS)

# Plot
plt.figure(figsize=(12, 8))

# Rooflines
plt.loglog(ai, bw_roof, '--', label="Memory Roof", color="darkorange")
plt.loglog(ai, compute_roof, '-', label="Compute Roof", color="royalblue")

# Kernels
plt.scatter(
    df["AI"],
    df["achieved_flops_tflops"],
    c=df["SoL_Compute"],
    cmap="viridis",
    edgecolors="black",
    s=60
)

# Labels
plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
plt.ylabel("Achieved Performance (TFLOPs)")
plt.title("H100 Speed-of-Light Roofline (Measured)")
plt.colorbar(label="Compute SoL")

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save
plt.savefig("h100_roofline.png", dpi=300)
plt.show()


