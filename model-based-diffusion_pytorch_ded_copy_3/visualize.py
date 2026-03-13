import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# Font size control
# ===============================
a = 14

plt.rcParams.update({
    "font.size": a,
    "axes.labelsize": a,
    "legend.fontsize": a-2,
    "xtick.labelsize": a-2,
    "ytick.labelsize": a-2
})

# ===============================
# CSV load
# ===============================
csv_path = "/home/ftk3187/github/IEMS490/model-based-diffusion_pytorch_ded_copy_3/results/plots/laser_5/Ns1024_H50_Nd50_T0.1_tr10.0_sm20.0_con1000_u00.1_noise0.3___Nd50_bad(noisy)/trajectory.csv"

df = pd.read_csv(csv_path)

temperature = df["temperature"].values
depth = df["depth"].values
laser_power = df["laser_power"].values
reference = df["reference_temp"].values

N = len(temperature)

# ===============================
# Constraint bounds
# ===============================
upper_bound = 0.225
lower_bound = 0.075

# ===============================
# Figure
# ===============================
fig, axes = plt.subplots(3,1, figsize=(10,8))

# =================================
# 1. Melt Pool Temperature
# =================================
axes[0].plot(reference[:N], label="Reference", color="orange")
axes[0].plot(temperature[:N], label="GAMMA simulation", color="blue")

axes[0].set_ylabel("Melt Pool Temperature (K)")
axes[0].legend()
axes[0].grid(alpha=0.3)


# =================================
# 2. Melt Pool Depth
# =================================
axes[1].plot(depth[:N], label="GAMMA simulation", color="green")

axes[1].axhline(
    y=upper_bound,
    linestyle="--",
    color="red",
    label="Upper Bound"
)

axes[1].axhline(
    y=lower_bound,
    linestyle="--",
    color="blue",
    label="Lower Bound"
)

axes[1].set_ylabel("Melt Pool Depth (mm)")
axes[1].legend()
axes[1].grid(alpha=0.3)


# =================================
# 3. Laser Power
# =================================
axes[2].plot(laser_power[:N], color="purple", label="Applied Laser Power")

axes[2].set_ylabel("Laser Power (W)")
axes[2].set_xlabel("MPC Time Step")

axes[2].legend()
axes[2].grid(alpha=0.3)

axes[0].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

axes[1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

axes[2].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

# ===============================
# Layout
# ===============================
plt.tight_layout()

save_path = csv_path.replace("trajectory.csv","replot.png")
plt.savefig(save_path, dpi=300)

print("saved:", save_path)

plt.show()