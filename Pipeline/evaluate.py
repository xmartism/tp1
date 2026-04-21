"""
evaluate.py — Sliding-window evaluator

Reads the `--windows-file` (CSV) produced by the pipeline for each model.
windows-file CSV columns:
  window_index, timestamp, prediction, actual

Metrics are computed:
  - per-window (each window separately)
  - aggregate across all predictions at once — this goes into results.csv

Outputs:
  - results.csv              (one aggregate row per model)
  - <model>_per_window.csv   (per-window detail)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_last_epoch(windows_file: str, model_name: str) -> int | None:
    """Reads the number of trained epochs from metadata.txt in the model directory."""
    metadata_path = Path(windows_file).parent / f"{model_name}_model" / "metadata.txt"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path) as f:
            for line in f:
                if line.startswith("epoch:"):
                    return int(line.split(":")[1].strip())
    except Exception as e:
        print(f"[WARN] Could not read epoch from {metadata_path}: {e}")
    return None


def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """Computes MSE, MAE, MAPE and DA for the given predictions and actuals."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    )

    if len(y_true) > 1:
        actual_diff = y_true[1:] - y_true[:-1]
        pred_diff   = y_pred[1:] - y_pred[:-1]
        da = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100)
    else:
        da = float("nan")

    return {"mse": mse, "mae": mae, "mape": mape, "da": da}


def load_windows(windows_file: str) -> pd.DataFrame:
    """Loads the windows CSV and verifies it is not empty."""
    try:
        df = pd.read_csv(windows_file)
    except Exception as e:
        print(f"[ERROR] Cannot load windows file: {e}")
        sys.exit(1)

    if df.empty:
        print("[ERROR] Windows file is empty.")
        sys.exit(1)

    return df


def compute_per_window_metrics(df_windows: pd.DataFrame, args) -> pd.DataFrame:
    """
    Computes metrics for each window separately.
    Returns a DataFrame where each row corresponds to one window.
    """
    rows = []

    for win_idx, group in df_windows.groupby("window_index", sort=True):
        preds      = group["prediction"].tolist()
        actuals    = group["actual"].tolist()
        timestamps = group["timestamp"].tolist()

        if len(preds) != len(actuals):
            print(
                f"[ERROR] Window {win_idx}: len(predictions)={len(preds)} != "
                f"len(actuals)={len(actuals)}"
            )
            sys.exit(1)

        if len(preds) == 0:
            print(f"[WARN] Window {win_idx} is empty, skipping.")
            continue

        m = calculate_metrics(actuals, preds)

        rows.append({
            "Model":           args.model_name,
            "Dataset":         args.dataset_name,
            "Target":          args.target,
            "Horizon":         args.horizon,
            "Lookback":        args.lookback_window,
            "Stride":          args.stride,
            "Seed":            args.seed,
            "Window":          win_idx,
            "Window_Start_TS": timestamps[0]  if timestamps else "",
            "Window_End_TS":   timestamps[-1] if timestamps else "",
            "MSE":             round(m["mse"],  4),
            "MAE":             round(m["mae"],  4),
            "MAPE (%)":        round(m["mape"], 4),
            "MDA (%)":         round(m["da"],   2) if not np.isnan(m["da"]) else "N/A",
        })

    if not rows:
        print("[ERROR] No valid windows to evaluate.")
        sys.exit(1)

    return pd.DataFrame(rows)


def compute_aggregate_metrics(df_windows: pd.DataFrame) -> tuple[dict, str]:
    """
    Computes aggregate metrics over averaged predictions per timestamp.
    For overlapping windows (stride < horizon) the same timestamp may appear
    multiple times — predictions are averaged so each point contributes equally.

    MSE, MAE, MAPE are computed over the full averaged series.
    DA is computed per-window and then averaged, because it measures direction
    within a window — computing it across window boundaries would be meaningless.

    Returns (metrics_dict, agg_mda_str).
    """
    # Average predictions per timestamp for point-wise metrics
    df_agg = df_windows.groupby("timestamp", sort=True).agg(
        prediction=("prediction", "mean"),
        actual=("actual", "first"),  # actual is identical across windows
    ).reset_index()

    # All metrics computed over the averaged series
    agg     = calculate_metrics(df_agg["actual"].tolist(), df_agg["prediction"].tolist())
    agg_mda = round(agg["da"], 2) if not np.isnan(agg["da"]) else "N/A"
    return agg, agg_mda


def save_results(df_per_window: pd.DataFrame, agg: dict, agg_mda, args) -> None:
    """
    Saves the per-window CSV and appends an aggregate row to results.csv.
    results.csv is created if it does not exist, otherwise the row is appended.
    """
    # Per-window CSV
    per_window_path = Path(args.results_file).parent / f"{args.model_name}_per_window.csv"
    df_per_window.to_csv(per_window_path, index=False)
    print(f"[INFO] Per-window metrics -> {per_window_path}")

    # Aggregate row
    epochs_trained = read_last_epoch(args.windows_file, args.model_name)
    num_windows    = len(df_per_window)

    df_agg = pd.DataFrame([{
        "Dataset":         args.dataset_name,
        "Target":          args.target,
        "Model":           args.model_name,
        "Horizon":         args.horizon,
        "Lookback window": args.lookback_window,
        "Stride":          args.stride,
        "Num Windows":     num_windows,
        "Seed":            args.seed,
        "Train Time (s)":  round(args.train_time, 2),
        "Epochs Trained":  epochs_trained if epochs_trained is not None else "N/A",
        "MSE":             round(agg["mse"],  4),
        "MAE":             round(agg["mae"],  4),
        "MAPE (%)":        round(agg["mape"], 4),
        "MDA (%)":         agg_mda,
    }])

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        df_agg.to_csv(results_path, mode="a", header=False, index=False)
    else:
        df_agg.to_csv(results_path, mode="w", header=True,  index=False)

    print(f"\n  [EVAL] {args.model_name} | {args.dataset_name} | Windows: {num_windows}")
    print(
        f"         Time: {args.train_time:.2f}s | "
        f"MSE: {agg['mse']:.4f} | MAE: {agg['mae']:.4f} | "
        f"MAPE: {agg['mape']:.2f}% | MDA: {agg_mda}%"
    )
    print(f"  [EVAL] Results -> {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sliding-window model evaluator")

    parser.add_argument("--model-name",      required=True)
    parser.add_argument("--windows-file",    required=True,
                        help="CSV file with all windows (pipeline output)")
    parser.add_argument("--results-file",    required=True,
                        help="Output results.csv (append mode)")
    parser.add_argument("--horizon",         type=int,   required=True)
    parser.add_argument("--lookback-window", type=int,   required=True)
    parser.add_argument("--stride",          type=int,   required=True)
    parser.add_argument("--seed",            type=int,   required=True)
    parser.add_argument("--dataset-name",    default="Unknown")
    parser.add_argument("--train-time",      type=float, default=0.0)
    parser.add_argument("--target",          required=True)

    args = parser.parse_args()

    df_windows    = load_windows(args.windows_file)
    df_per_window = compute_per_window_metrics(df_windows, args)
    agg, agg_mda  = compute_aggregate_metrics(df_windows)
    save_results(df_per_window, agg, agg_mda, args)


if __name__ == "__main__":
    main()