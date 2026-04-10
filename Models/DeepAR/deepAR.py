"""
DeepAR Forecasting Tool (GluonTS/MXNet)

Pouzitie:

python3 deepAR.py \
    --train-dataset train.csv \
    --val-dataset val.csv \
    --test-dataset test.csv \
    --target spotreba \
    --horizon 24 \
    --output prediction.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mxnet as mx

from gluonts.dataset.common import ListDataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import Callback
from gluonts.mx.distribution import GaussianOutput
from gluonts.evaluation import make_evaluation_predictions


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping(Callback):

    def __init__(self, patience=5, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.last_epoch = 0
        self.best_network_params = None

    def on_epoch_end(self, epoch_no, epoch_loss, training_network, trainer, best_epoch_info, ctx):
        self.last_epoch = epoch_no

        if epoch_loss < self.best_loss - self.min_delta:
            self.best_loss = epoch_loss
            self.best_epoch = epoch_no
            self.epochs_without_improvement = 0

            if self.restore_best:
                self.best_network_params = {
                    k: v.data().copy()
                    for k, v in training_network.collect_params().items()
                }

        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:

            if self.restore_best and self.best_network_params is not None:

                for k, v in training_network.collect_params().items():
                    if k in self.best_network_params:
                        v.set_data(self.best_network_params[k])

            return False

        return True


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_csv_dataset(filepath, date_col, target_col):

    path = Path(filepath)

    if not path.exists():
        print(f"File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    sep = '\t' if path.suffix.lower() == ".txt" else ","

    df = pd.read_csv(filepath, sep=sep)

    if date_col not in df.columns:
        raise ValueError(f"Missing column '{date_col}'")

    if target_col not in df.columns:
        raise ValueError(f"Missing column '{target_col}'")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    return df


def infer_freq(df):
    freq = pd.infer_freq(df.index)
    if freq is None:
        return "h"
    # Normalize deprecated aliases
    freq_map = {
        "H": "h", "60T": "h", "60min": "h",
        "T": "min",
        "M": "ME", "Q": "QE-DEC", "Y": "YE-DEC",
    }
    return freq_map.get(freq, freq)


def build_dataset_entry(df, target_col, covariate_cols):
    """Build a single GluonTS entry dict from a dataframe."""
    entry = {
        "start":  df.index[0],
        "target": df[target_col].values.astype(float),
    }
    if covariate_cols:
        entry["feat_dynamic_real"] = df[covariate_cols].values.T.astype(float)
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--val-dataset",   required=True)
    parser.add_argument("--test-dataset",  required=True)

    parser.add_argument("--target", required=True)
    parser.add_argument("--date",   default="date")

    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output",  required=True)

    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)

    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-cells",  type=int, default=40)

    parser.add_argument("--context-length", type=int, default=None)

    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--patience",    type=int, default=10)

    parser.add_argument("--lookback-window", type=int, default=None)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Random seed
    # ------------------------------------------------------------------

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------

    train_df = load_csv_dataset(args.train_dataset, args.date, args.target)
    val_df   = load_csv_dataset(args.val_dataset,   args.date, args.target)
    test_df  = load_csv_dataset(args.test_dataset,  args.date, args.target)

    freq = infer_freq(train_df)

    print(f"[INFO] Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

    # ------------------------------------------------------------------
    # Feature selection (all numeric columns except target)
    # ------------------------------------------------------------------

    covariate_cols = [
        c for c in train_df.columns
        if c != args.target and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # ------------------------------------------------------------------
    # Build GluonTS datasets
    #
    # train_dataset  — pure training data (70%)
    # val_dataset    — train + val concatenated so the model sees a longer
    #                  history and we evaluate on the val window
    # test_dataset   — train + val + test concatenated for final prediction
    # ------------------------------------------------------------------

    # Concatenate splits progressively so context is always available
    train_val_df      = pd.concat([train_df, val_df])
    train_val_test_df = pd.concat([train_df, val_df, test_df])

    train_dataset = ListDataset(
        [build_dataset_entry(train_df,          args.target, covariate_cols)], freq=freq
    )
    val_dataset = ListDataset(
        [build_dataset_entry(train_val_df,      args.target, covariate_cols)], freq=freq
    )
    test_dataset = ListDataset(
        [build_dataset_entry(train_val_test_df, args.target, covariate_cols)], freq=freq
    )

    # ------------------------------------------------------------------
    # Context length
    # ------------------------------------------------------------------

    lookback_window = args.lookback_window if args.lookback_window else args.horizon * 4

    if args.context_length:
        context_length = args.context_length
    else:
        context_length = lookback_window

    # ------------------------------------------------------------------
    # MXNet context
    # ------------------------------------------------------------------

    ctx = mx.cpu()
    if args.device == "gpu":
        try:
            ctx = mx.gpu()
        except Exception:
            ctx = mx.cpu()

    # ------------------------------------------------------------------
    # Model — train on train split, monitor loss on val split
    # ------------------------------------------------------------------

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=1e-4,
        restore_best=True,
    )

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=args.horizon,
        context_length=context_length,
        num_layers=args.num_layers,
        num_cells=args.num_cells,
        distr_output=GaussianOutput(),
        scaling=False,
        batch_size=args.batch_size,
        trainer=Trainer(
            epochs=args.epochs,
            learning_rate=args.lr,
            hybridize=True,
            ctx=ctx,
            callbacks=[early_stopping],
            add_default_callbacks=True,
        ),
    )

    print("[INFO] Training on train split...")
    predictor = estimator.train(
        training_data=train_dataset,
        validation_data=val_dataset,
    )

    # ------------------------------------------------------------------
    # Prediction — run on full history (train + val + test)
    # ------------------------------------------------------------------

    print("[INFO] Running prediction on test split...")
    forecast_it = predictor.predict(test_dataset)
    forecasts = list(forecast_it)
    predictions = forecasts[0].quantile(0.5).tolist()[:args.horizon]

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    metadata_path = output_path.parent / "metadata_deepAR.txt"
    with open(metadata_path, "w") as f:
        f.write(f"epochs:{early_stopping.last_epoch}\n")

    print(f"[INFO] Predictions saved to {output_path}")
    print(f"[INFO] Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()