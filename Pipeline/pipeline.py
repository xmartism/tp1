"""
Forecasting Pipeline

Rozdeli dataset (70/15/15) a postupne spusti vsetky modely.
Vystupy: Pipeline/outputs/<experiment_id>/<model>_output.json, Pipeline/outputs/<experiment_id>/results.csv, Pipeline/outputs/experiments.csv

Parametre:
--dataset       (povinny)  cesta k CSV datasetu
--target        (povinny)  nazov cieloveho stlpca
--date          (default: "date")  nazov stlpca s datumom
--horizon       (povinny)  pocet krokov predikcie
--lookback-window (volitelny, default: 4 * horizon)  dlzka vstupneho okna
--seed          (volitelny, default: nahodne vygenerovany)  seed pre reprodukovatelnost
--output-dir    (default: Pipeline/outputs)  priecinok pre vystupy tohto behu

Priklad (priamo):
python3 Pipeline/pipeline.py \
    --dataset data/weatherHistory.csv \
    --target "Temperature (C)" \
    --date "Formatted Date" \
    --horizon 24 \
    --lookback-window 96 \
    --seed 42 \
    --output-dir Pipeline/outputs/1

Priklad (cez run_experiments.py - odporucane):
python3 Pipeline/run_experiments.py
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import random
import pandas as pd
import time


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    # {
    #     "name": "deepAR",
    #     "script": "Models/DeepAR/deepAR.py",
    # },
    # {
    #     "name": "nbeats",
    #      "script": "Models/NBeats/NBeats.py",
    # },
    {
        "name": "tsmixer",
        "script": "Models/tsmixer/tsmixer.py",
    },
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
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(dataset_path: str, date_col: str) -> tuple[Path, Path, Path]:
    """
    Load the dataset and split chronologically:
        70% train | 15% validation | 15% test

    Returns paths to (train.csv, val.csv, test.csv) written in a temp directory.
    """
    path = Path(dataset_path)
    sep = "\t" if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(path, sep=sep)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    test_ratio = 1 - TRAIN_RATIO - VAL_RATIO
    print(
        f"[INFO] Dataset split: "
        f"{len(train_df)} train ({TRAIN_RATIO:.0%}) / "
        f"{len(val_df)} val ({VAL_RATIO:.0%}) / "
        f"{len(test_df)} test ({test_ratio:.0%}) "
        f"- {n} rows total"
    )

    tmp_dir    = Path(tempfile.mkdtemp(prefix="pipeline_"))
    train_path = tmp_dir / "train.csv"
    val_path   = tmp_dir / "val.csv"
    test_path  = tmp_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"[INFO] Train -> {train_path}")
    print(f"[INFO] Val   -> {val_path}")
    print(f"[INFO] Test  -> {test_path}")

    return train_path, val_path, test_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], step_label: str) -> bool:
    """Run a subprocess command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {step_label}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] '{step_label}' failed with exit code {result.returncode}",
              file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):

    # Split once - all models share the same splits
    train_path, val_path, test_path = split_dataset(args.dataset, args.date)
    lookback_window = args.lookback_window if args.lookback_window is not None else 4 * args.horizon
    seed = args.seed if args.seed is not None else random.randint(0, 2 ** 31 - 1)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.csv"

    failed_models = []

    for model in MODELS:
        name        = model["name"]
        script      = model["script"]
        output_file = str(output_dir / f"{name}_output.txt")

        print(f"\n\n{'#'*60}")
        print(f"#  RUNNING MODEL: {name.upper()}")
        print(f"{'#'*60}")

        # ------------------------------------------------------------------
        # Step 1: Train & predict
        # ------------------------------------------------------------------
        model_cmd = [
            sys.executable, script,
            "--train-dataset", str(train_path),
            "--val-dataset",   str(val_path),
            "--test-dataset",  str(test_path),
            "--target",        args.target,
            "--date",          args.date,
            "--horizon",       str(args.horizon),
            "--lookback-window", str(lookback_window),
            "--seed",          str(seed),
            "--output",        output_file,
        ]

        start_time = time.time()
        success = run_command(model_cmd, f"[{name}] Training & Prediction")
        train_time = time.time() - start_time

        if not success:
            print(f"[WARN] Skipping evaluation for '{name}' due to model failure.")
            failed_models.append(name)
            continue

        # ------------------------------------------------------------------
        # Step 2: Evaluate -> results.csv
        # ------------------------------------------------------------------
        eval_cmd = [
            sys.executable, "Pipeline/evaluate.py",
            "--model-name", name,
            "--output-file", output_file,
            "--results-file", str(results_file),
            "--test-dataset", str(test_path),
            "--target", args.target,
            "--horizon", str(args.horizon),
            "--lookback-window", str(lookback_window),
            "--seed", str(seed),
            "--dataset-name", Path(args.dataset).name,
            "--train-time", str(train_time)
        ]

        success = run_command(eval_cmd, f"[{name}] Evaluation")

        if not success:
            print(f"[WARN] Evaluation failed for '{name}'.")
            failed_models.append(name)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total  = len(MODELS)
    passed = total - len(failed_models)

    print(f"\n\n{'='*60}")
    print(f"  PIPELINE COMPLETE  -  {passed}/{total} models succeeded")
    if failed_models:
        print(f"  Failed models: {', '.join(failed_models)}")
    print(f"  Results saved to: {results_file}")
    print(f"{'='*60}\n")

    return len(failed_models) == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sequential model forecasting pipeline")

    parser.add_argument("--dataset",      required=True, help="Path to the full dataset CSV")
    parser.add_argument("--target",       required=True, help="Target column name")
    parser.add_argument("--date",         default="date", help="Date column name (default: date)")
    parser.add_argument("--horizon",      type=int, required=True, help="Forecast horizon")
    parser.add_argument("--lookback-window", type=int, default=None, help="Dlzka vstupneho okna (default: 4 * horizon)")
    parser.add_argument("--seed",         type=int, default=None, help="Seed pre reprodukovatelnost")
    parser.add_argument("--output-dir",   default="Pipeline/outputs",
                        help="Directory for model outputs and results (default: Pipeline/outputs)")

    args = parser.parse_args()

    ok = run_pipeline(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
