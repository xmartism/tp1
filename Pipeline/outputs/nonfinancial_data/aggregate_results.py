"""
Agreguje výsledky experimentov z viacerých priečinkov (results.csv)
do jedného CSV so stĺpcami mean a std pre každú metriku.

Priečinky sú zoskupené po troch (rovnaký dataset/konfigurácia, rôzny seed):
    skupina 1: [1, 9, 17]
    skupina 2: [2, 10, 18]
    ...
    skupina 8: [8, 16, 24]

Použitie (z aktualneho priecinku):
    python3 aggregate_results.py

Výstup:
    aggregated_results.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

METRICS = ["Train Time (s)", "Epochs Trained", "MSE", "MAE", "MAPE (%)", "MDA (%)"]

# Groups of 3 directories (1-indexed)
# Group 1: [1,9,17], Group 2: [2,10,18], ..., Group 8: [8,16,24]
GROUPS = [(i, i + 8, i + 16) for i in range(1, 9)]

base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

all_group_results = []

for group in GROUPS:
    dfs = []
    for dir_idx in group:
        csv_path = base_dir / str(dir_idx) / "results.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        print(f"Group {group}: no data found")
        continue

    combined = pd.concat(dfs, ignore_index=True)

    # Group by Model (and optionally Horizon/Lookback) then compute mean+std
    group_stats = (
        combined.groupby("Model")[METRICS]
        .agg(["mean", "std"])
    )
    # Flatten MultiIndex columns: ("MSE", "mean") -> "MSE_mean"
    group_stats.columns = [f"{col}_{stat}" for col, stat in group_stats.columns]
    group_stats = group_stats.reset_index()
    group_stats.insert(0, "dirs", str(group))

    all_group_results.append(group_stats)
    print(f"\n=== Group {group} ===")
    for _, row in group_stats.iterrows():
        print(f"  {row['Model']}:")
        for m in METRICS:
            print(f"    {m}: mean={row[f'{m}_mean']:.4f}  std={row[f'{m}_std']:.4f}")

if all_group_results:
    final = pd.concat(all_group_results, ignore_index=True)
    out_path = base_dir / "aggregated_results.csv"
    final.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved aggregated results to: {out_path}")