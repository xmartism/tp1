"""
Usage:
    python Pipeline/plot_windows.py Pipeline/outputs/financial/test/deepAR_windows.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "Pipeline/outputs/financial/test/deepAR_windows.csv"

df = pd.read_csv(path, parse_dates=["timestamp"])

fig, ax = plt.subplots(figsize=(16, 5))

# Plot actual values (one line — deduplicated by timestamp)
actuals = df.drop_duplicates("timestamp").sort_values("timestamp")
ax.plot(actuals["timestamp"], actuals["actual"], color="black", linewidth=1.2, label="Actual", zorder=3)

# Plot each window's predictions in a distinct colour
windows = sorted(df["window_index"].unique())
colors = cm.tab20(np.linspace(0, 1, len(windows)))

for win, color in zip(windows, colors):
    w = df[df["window_index"] == win].sort_values("timestamp")
    ax.plot(w["timestamp"], w["prediction"], color=color, linewidth=0.9, alpha=0.7, label=f"Window {win}")

ax.set_title(f"Predictions vs Actual — {path}")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend(loc="upper left", fontsize=7, ncol=2)
plt.tight_layout()

out_path = path.replace(".csv", ".png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()