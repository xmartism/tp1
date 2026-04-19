"""
Forecasting Pipeline with shared normalization and sliding window evaluation

Steps:
  1. Dataset split (70/15/15) and normalization (StandardScaler fitted on train only)
  2. Each model is trained once on train+val data (--mode train)
  3. Pipeline iterates sliding windows over the test set and calls
     each model in --mode predict for every window
  4. Predictions are inverse-transformed and saved with timestamps
  5. evaluate.py aggregates results across all windows

Sliding window parameters:
  --stride   (default: horizon)  number of steps to advance the window
             Example: horizon=24, stride=24 → non-overlapping windows
             Example: horizon=24, stride=12 → 50% overlap

Parameters:
  --dataset        (required)  path to CSV
  --target         (required)  name of the target column
  --date           (default: "date")
  --horizon        (required)  number of forecast steps
  --lookback-window (default: 4 * horizon)  context window length for predict
  --stride         (default: horizon)       sliding window step size
  --seed           (default: random)
  --output-dir     (default: Pipeline/outputs)

Example:
  python3 Pipeline/pipeline.py \
      --dataset Data/weatherHistory.csv \
      --target "Temperature (C)" \
      --date "Formatted Date" \
      --horizon 24 \
      --lookback-window 96 \
      --stride 24 \
      --seed 42 \
      --output-dir Pipeline/outputs/exp1
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Always run relative to the project root (one level above this script)
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name": "deepAR",
        "script": "Models/DeepAR/deepAR.py",
    },
    # {
    #     "name": "nbeats",
    #     "script": "Models/NBeats/NBeats.py",
    # },
    # {
    #     "name": "tsmixer",
    #     "script": "Models/Tsmixer/tsmixer.py",
    # },
    # {
    #     "name": "tft",
    #     "script": "Models/TFT/tft.py",
    # },
    {
        "name": "dlinear",
        "script": "Models/LTSF-Linear/run_longExp.py",
    },
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)


# ---------------------------------------------------------------------------
# Dataset splitting & normalization
# ---------------------------------------------------------------------------

def split_and_scale_dataset(
    dataset_path: str,
    date_col: str,
    target_col: str,
) -> tuple[Path, Path, Path, Path, StandardScaler, list, pd.DataFrame]:
    """
    Loads the dataset, splits it chronologically (70/15/15) and scales all numeric columns.
    StandardScaler is fitted on training data only.

    Parameters:
        dataset_path — cesta k CSV datasetu
        date_col     — name of the date column
        target_col   — name of the target column

    Vracia:
        train_path, val_path, test_scaled_path, test_orig_path,
        scaler, numeric_cols, test_df_original
    """
    path = Path(dataset_path)
    sep  = "\t" if path.suffix.lower() == ".txt" else ","
    df   = pd.read_csv(path, sep=sep)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.sort_values(date_col).reset_index(drop=True)

    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    test_df_original = test_df.copy()

    test_ratio = 1 - TRAIN_RATIO - VAL_RATIO
    print(
        f"[INFO] Dataset split: "
        f"{len(train_df)} train ({TRAIN_RATIO:.0%}) / "
        f"{len(val_df)} val ({VAL_RATIO:.0%}) / "
        f"{len(test_df)} test ({test_ratio:.0%}) "
        f"- {n} rows total"
    )

    # Scale all numeric columns except the date column
    numeric_cols = [
        c for c in train_df.columns
        if c != date_col and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols]   = scaler.transform(val_df[numeric_cols])
    test_df[numeric_cols]  = scaler.transform(test_df[numeric_cols])

    target_idx = numeric_cols.index(target_col)
    print(
        f"[INFO] Normalization applied to {len(numeric_cols)} columns: {numeric_cols}\n"
        f"       Target '{target_col}': Mean={scaler.mean_[target_idx]:.4f}, Scale={scaler.scale_[target_idx]:.4f}"
    )

    tmp_dir        = Path(tempfile.mkdtemp(prefix="pipeline_"))
    train_path     = tmp_dir / "train.csv"
    val_path       = tmp_dir / "val.csv"
    test_path      = tmp_dir / "test.csv"
    test_orig_path = tmp_dir / "test_original.csv"

    train_df.to_csv(train_path,     index=False)
    val_df.to_csv(val_path,         index=False)
    test_df.to_csv(test_path,       index=False)
    test_df_original.to_csv(test_orig_path, index=False)

    print(f"[INFO] Train (scaled) -> {train_path}")
    print(f"[INFO] Val   (scaled) -> {val_path}")
    print(f"[INFO] Test  (scaled) -> {test_path}")
    print(f"[INFO] Test  (orig)   -> {test_orig_path}")

    return train_path, val_path, test_path, test_orig_path, scaler, numeric_cols, test_df_original


# ---------------------------------------------------------------------------
# Sliding window helpers
# ---------------------------------------------------------------------------

def build_sliding_windows(
    test_df: pd.DataFrame,
    date_col: str,
    horizon: int,
    stride: int,
) -> list[dict]:
    """
    Vygeneruje zoznam sliding windows cez test set.

    Parameters:
        test_df   — scaled test set; defines window boundaries and timestamps
        date_col  — name of the date column (for generating horizon_timestamps)
        horizon   — number of forecast steps per window
        stride    — number of steps to advance the window (stride=horizon → non-overlapping)

    Each window dict contains:
        window_index       — sequential window number
        test_start_idx     — window start as index in test_df (where actuals begin)
        horizon_timestamps — ISO timestamps for the forecasted horizon steps

    Returns list of window dicts.
    """
    test_len = len(test_df)
    if test_len < horizon:
        raise ValueError(
            f"Test set ({test_len} rows) is shorter than horizon ({horizon}). "
            "Cannot evaluate with sliding windows."
        )

    windows = []
    window_start = 0  # index within test_df

    while window_start + horizon <= test_len:
        if date_col in test_df.columns:
            horizon_timestamps = [
                pd.Timestamp(ts).isoformat()
                for ts in test_df[date_col].iloc[window_start:window_start + horizon].tolist()
            ]
        else:
            horizon_timestamps = [f"step_{i}" for i in range(horizon)]

        windows.append({
            "window_index":       len(windows),
            "test_start_idx":     window_start,
            "horizon_timestamps": horizon_timestamps,
        })

        window_start += stride

    print(
        f"[INFO] Sliding windows: {len(windows)} windows "
        f"(horizon={horizon}, stride={stride})"
    )
    return windows


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], step_label: str) -> bool:
    """Runs a subprocess command and returns True if it succeeded."""
    print(f"\n{'='*60}")
    print(f"  {step_label}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(
            f"\n[ERROR] '{step_label}' failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def prepare_data(
    args,
) -> tuple[Path, Path, Path, StandardScaler, int, pd.DataFrame, pd.DataFrame, int]:
    """
    Splits and scales the dataset, builds full_df_scaled and computes test_start_abs.

    Returns:
        train_path, val_path, full_df_scaled, scaler, target_idx,
        test_df_scaled, test_df_original, test_start_abs
    """
    (
        train_path,
        val_path,
        test_scaled_path,
        _test_orig_path,
        scaler,
        numeric_cols,
        test_df_original,
    ) = split_and_scale_dataset(args.dataset, args.date, args.target)

    target_idx = numeric_cols.index(args.target)

    train_df_scaled = pd.read_csv(train_path)
    val_df_scaled   = pd.read_csv(val_path)
    test_df_scaled  = pd.read_csv(test_scaled_path)

    full_df_scaled = pd.concat(
        [train_df_scaled, val_df_scaled, test_df_scaled], ignore_index=True
    )
    test_start_abs = len(train_df_scaled) + len(val_df_scaled)

    return (
        train_path, val_path,
        full_df_scaled, scaler, target_idx,
        test_df_scaled, test_df_original, test_start_abs,
    )


def train_model(
    script: str, name: str, train_path: Path, val_path: Path,
    model_dir: Path, args, lookback_window: int, seed: int,
) -> float | None:
    """
    Trains the model. Returns training time in seconds, or None if training failed.
    """
    train_cmd = [
        sys.executable, script,
        "--mode",            "train",
        "--train-dataset",   str(train_path),
        "--val-dataset",     str(val_path),
        "--target",          args.target,
        "--date",            args.date,
        "--horizon",         str(args.horizon),
        "--lookback-window", str(lookback_window),
        "--model-dir",       str(model_dir),
        "--seed",            str(seed),
    ]
    start_time = time.time()
    success    = run_command(train_cmd, f"[{name}] Training")
    train_time = time.time() - start_time

    if not success:
        print(f"[WARN] Skipping '{name}' — training failed.")
        return None
    return train_time


def predict_windows(
    script: str, name: str, model_dir: Path,
    windows: list[dict], full_df_scaled: pd.DataFrame,
    test_start_abs: int, test_df_original: pd.DataFrame,
    scaler, target_idx: int, tmp_dir: Path, args,
    lookback_window: int, seed: int,
) -> list[dict] | None:
    """
    Runs prediction for each sliding window.
    Returns a list of window results, or None if any window failed.

    Each result contains: window_index, timestamps, predictions, actuals.
    """
    all_window_outputs = []

    for win in windows:
        win_idx        = win["window_index"]
        test_start_idx = win["test_start_idx"]

        # Full history up to the window start — the model trims to lookback_window steps itself
        context_end_abs = test_start_abs + test_start_idx
        context_df      = full_df_scaled.iloc[:context_end_abs]

        context_csv     = tmp_dir / f"{name}_context_w{win_idx}.csv"
        context_df.to_csv(context_csv, index=False)

        win_output_file = tmp_dir / f"{name}_pred_w{win_idx}.csv"

        predict_cmd = [
            sys.executable, script,
            "--mode",            "predict",
            "--context-dataset", str(context_csv),
            "--target",          args.target,
            "--date",            args.date,
            "--horizon",         str(args.horizon),
            "--lookback-window", str(lookback_window),
            "--model-dir",       str(model_dir),
            "--output",          str(win_output_file),
            "--seed",            str(seed),
        ]

        if not run_command(predict_cmd, f"[{name}] Predict window {win_idx}"):
            print(f"[WARN] Predict failed for '{name}' window {win_idx}.")
            return None

        win_df       = pd.read_csv(win_output_file)
        scaled_preds = win_df["prediction"].tolist()
        timestamps   = win_df["timestamp"].tolist()

        preds_arr     = np.array(scaled_preds)
        preds_inverse = (preds_arr * scaler.scale_[target_idx] + scaler.mean_[target_idx]).tolist()

        actuals = (
            test_df_original[args.target]
            .iloc[test_start_idx: test_start_idx + args.horizon]
            .tolist()
        )

        all_window_outputs.append({
            "window_index": win_idx,
            "timestamps":   timestamps,
            "predictions":  preds_inverse,
            "actuals":      actuals,
        })

    return all_window_outputs


def save_windows_csv(all_window_outputs: list[dict], output_path: Path) -> None:
    """Saves predictions and actuals for all windows into a single CSV file."""
    rows = []
    for w in all_window_outputs:
        for ts, pred, actual in zip(w["timestamps"], w["predictions"], w["actuals"]):
            rows.append({
                "window_index": w["window_index"],
                "timestamp":    ts,
                "prediction":   pred,
                "actual":       actual,
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def evaluate_model(
    name: str, agg_output_file: Path, results_file: Path,
    args, lookback_window: int, stride: int, seed: int, train_time: float,
) -> bool:
    """Runs evaluate.py for the given model and returns True if it succeeded."""
    eval_cmd = [
        sys.executable, "Pipeline/evaluate.py",
        "--model-name",      name,
        "--windows-file",    str(agg_output_file),
        "--results-file",    str(results_file),
        "--horizon",         str(args.horizon),
        "--lookback-window", str(lookback_window),
        "--stride",          str(stride),
        "--seed",            str(seed),
        "--dataset-name",    Path(args.dataset).name,
        "--train-time",      str(train_time),
        "--target",          args.target,
    ]
    return run_command(eval_cmd, f"[{name}] Evaluation")


def run_pipeline(args) -> bool:
    """
    Main pipeline logic: scaling, model training,
    sliding window prediction, inverse-transform and evaluation.
    Returns True if all models succeeded.
    """
    seed            = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    lookback_window = args.lookback_window if args.lookback_window is not None else 4 * args.horizon
    stride          = args.stride if args.stride is not None else args.horizon

    print(f"[INFO] Seed={seed} | Horizon={args.horizon} | Lookback={lookback_window} | Stride={stride}")

    (
        train_path, val_path,
        full_df_scaled, scaler, target_idx,
        test_df_scaled, test_df_original, test_start_abs,
    ) = prepare_data(args)

    windows = build_sliding_windows(
        test_df=test_df_scaled,
        date_col=args.date,
        horizon=args.horizon,
        stride=stride,
    )

    if not windows:
        print("[ERROR] No sliding windows were generated. Check the test set length and horizon.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.csv"
    tmp_dir      = Path(tempfile.mkdtemp(prefix="pipeline_windows_"))

    failed_models = []

    for model in MODELS:
        name      = model["name"]
        script    = model["script"]
        model_dir = output_dir / f"{name}_model"

        print(f"\n\n{'#'*60}")
        print(f"#  MODEL: {name.upper()}")
        print(f"{'#'*60}")

        train_time = train_model(
            script, name, train_path, val_path,
            model_dir, args, lookback_window, seed,
        )
        if train_time is None:
            failed_models.append(name)
            continue

        all_window_outputs = predict_windows(
            script, name, model_dir,
            windows, full_df_scaled,
            test_start_abs, test_df_original,
            scaler, target_idx, tmp_dir, args,
            lookback_window, seed,
        )
        if all_window_outputs is None:
            failed_models.append(name)
            continue

        agg_output_file = output_dir / f"{name}_windows.csv"
        save_windows_csv(all_window_outputs, agg_output_file)

        if not evaluate_model(name, agg_output_file, results_file, args, lookback_window, stride, seed, train_time):
            print(f"[WARN] Evaluation failed for '{name}'.")
            failed_models.append(name)

    total  = len(MODELS)
    passed = total - len(failed_models)

    print(f"\n\n{'='*60}")
    print(f"  PIPELINE COMPLETE  —  {passed}/{total} models succeeded")
    if failed_models:
        print(f"  Failed models: {', '.join(failed_models)}")
    print(f"  Results saved to: {results_file}")
    print(f"{'='*60}\n")

    return len(failed_models) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parses arguments and runs the pipeline."""
    parser = argparse.ArgumentParser(description="Sliding-window forecasting pipeline")

    parser.add_argument("--dataset",          required=True, help="Path to the CSV dataset")
    parser.add_argument("--target",           required=True, help="Target column name")
    parser.add_argument("--date",             default="date", help="Date column name (default: date)")
    parser.add_argument("--horizon",          type=int, required=True, help="Forecast horizon")
    parser.add_argument("--lookback-window",  type=int, default=None,
                        help="Context window length (default: 4 * horizon)")
    parser.add_argument("--stride",           type=int, default=None,
                        help="Sliding window step size (default: horizon → non-overlapping)")
    parser.add_argument("--seed",             type=int, default=None)
    parser.add_argument("--output-dir",       default="Pipeline/outputs")

    args = parser.parse_args()
    ok   = run_pipeline(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()