"""
Experiment configuration and pipeline runner.

Each combination (dataset × horizon) is run as a separate experiment.
Results are saved to:
  Pipeline/outputs/experiments.csv     — registry of all experiments
  Pipeline/outputs/<id>/               — outputs and metrics for each experiment

Example:
  python3 Pipeline/run_experiments.py
"""

import os
import subprocess
import sys
import csv
from pathlib import Path
import time

# Always run relative to the project root (one level above this script)
os.chdir(Path(__file__).resolve().parent.parent)

from experiments_config import EXPERIMENTS_NONFINANCIAL, EXPERIMENTS_FINANCIAL

USE_FINANCIAL = True

EXPERIMENTS = EXPERIMENTS_FINANCIAL if USE_FINANCIAL else EXPERIMENTS_NONFINANCIAL

OUTPUTS_ROOT     = Path("Pipeline/outputs/financial") if USE_FINANCIAL else Path("Pipeline/outputs/nonfinancial")
EXPERIMENTS_FILE = OUTPUTS_ROOT / "experiments.csv"
EXPERIMENTS_COLS = ["id", "dataset", "target", "date", "horizon", "results_file", "status"]


def next_experiment_id() -> int:
    """Returns the next experiment ID (max existing + 1, or 1)."""
    if not EXPERIMENTS_FILE.exists():
        return 1
    with open(EXPERIMENTS_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 1
    return max(int(r["id"]) for r in rows) + 1


def register_experiment(experiment_id: int, dataset: str, target: str, date_col: str,
                         horizon: int, results_file: Path, status: str) -> None:
    """Appends a row to experiments.csv."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not EXPERIMENTS_FILE.exists()
    with open(EXPERIMENTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "id":           experiment_id,
            "dataset":      dataset,
            "target":       target,
            "date":         date_col,
            "horizon":      horizon,
            "results_file": str(results_file),
            "status":       status,
        })


def update_experiment_status(experiment_id: int, status: str) -> None:
    """Updates the 'status' column for the given experiment in experiments.csv."""
    if not EXPERIMENTS_FILE.exists():
        return
    with open(EXPERIMENTS_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["id"]) == experiment_id:
            row["status"] = status
    with open(EXPERIMENTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_COLS)
        writer.writeheader()
        writer.writerows(rows)


def run_single_experiment(experiment_id: int, dataset: str, target: str,
                           date_col: str, horizon: int,
                           lookback_window: int | None = None,
                           stride: int | None = None,
                           seed: int | None = None) -> bool:
    """Runs the pipeline for a single experiment and updates its status in experiments.csv."""
    out_dir      = OUTPUTS_ROOT / str(experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.csv"

    print(f"\n{'#'*60}")
    print(f"#  EXPERIMENT {experiment_id}")
    print(f"#  Dataset : {dataset}")
    print(f"#  Target  : {target}")
    print(f"#  Horizon : {horizon}")
    if lookback_window is not None:
        print(f"#  Lookback: {lookback_window}")
    if stride is not None:
        print(f"#  Stride  : {stride}")
    if seed is not None:
        print(f"#  Seed    : {seed}")
    print(f"#  Out dir : {out_dir}")
    print(f"{'#'*60}")

    register_experiment(experiment_id, dataset, target, date_col, horizon, results_file, "running")

    cmd = [
        sys.executable, "Pipeline/pipeline.py",
        "--dataset",    dataset,
        "--target",     target,
        "--date",       date_col,
        "--horizon",    str(horizon),
        "--output-dir", str(out_dir),
        *(["--lookback-window", str(lookback_window)] if lookback_window is not None else []),
        *(["--stride",          str(stride)]          if stride          is not None else []),
        *(["--seed",            str(seed)]             if seed            is not None else []),
    ]

    print(f"  CMD: {' '.join(cmd)}\n")

    result  = subprocess.run(cmd, text=True)
    success = result.returncode == 0
    update_experiment_status(experiment_id, "success" if success else "failed")

    if not success:
        print(f"\n[ERROR] Experiment {experiment_id} failed (exit code {result.returncode}).",
              file=sys.stderr)
    return success


def main():
    """Runs all experiments defined in EXPERIMENTS and prints a summary."""
    total  = sum(len(e["horizons"]) for e in EXPERIMENTS)
    passed = 0
    failed = []

    experiment_id = next_experiment_id()

    for exp_cfg in EXPERIMENTS:
        dataset         = exp_cfg["dataset"]
        target          = exp_cfg["target"]
        date_col        = exp_cfg.get("date", "date")
        lookback_window = exp_cfg.get("lookback-window", None)
        stride          = exp_cfg.get("stride", None)
        seed            = exp_cfg.get("seed", None)

        for horizon in exp_cfg["horizons"]:
            ok = run_single_experiment(
                experiment_id=experiment_id,
                dataset=dataset,
                target=target,
                date_col=date_col,
                horizon=horizon,
                lookback_window=lookback_window,
                stride=stride,
                seed=seed,
            )
            if ok:
                passed += 1
            else:
                failed.append(f"exp_{experiment_id} ({Path(dataset).stem}, h={horizon})")
            experiment_id += 1

    print(f"\n\n{'='*60}")
    print(f"  ALL EXPERIMENTS DONE  —  {passed}/{total} succeeded")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Experiment registry: {EXPERIMENTS_FILE}")
    print(f"{'='*60}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()