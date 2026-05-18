"""
Agreguje výsledky experimentov z viacerých priečinkov (results.csv)
do jedného CSV so stĺpcami mean a std pre každú metriku.

Priečinky sú zoskupené po troch (rovnaký dataset/konfigurácia, rôzny seed):
    skupina 1: [1, 9, 17]
    skupina 2: [2, 10, 18]
    ...
    skupina 8: [8, 16, 24]

Použitie (z aktualneho priecinku):
    python3 aggregate_results.py <nazov_priecinku> <num_datasets> <seeds_per_dataset>

Výstup:
    <nazov_priecinku>/aggregated_results.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

METRICS = ["Train Time (s)", "Epochs Trained", "MSE", "MAE", "MAPE (%)", "MDA (%)"]

if len(sys.argv) < 4:
    print("Použitie: python3 aggregate_results.py <nazov_priecinku> <num_datasets> <seeds_per_dataset>")
    sys.exit(1)
base_dir          = Path(sys.argv[1])
NUM_DATASETS      = int(sys.argv[2])
SEEDS_PER_DATASET = int(sys.argv[3])

# Groups of SEEDS_PER_DATASET directories (1-indexed)
# Dataset 1: [1, 6, 11], Dataset 2: [2, 7, 12], ..., Dataset 5: [5, 10, 15]
GROUPS = [
    tuple(d + s * NUM_DATASETS for s in range(SEEDS_PER_DATASET))
    for d in range(1, NUM_DATASETS + 1)
]

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