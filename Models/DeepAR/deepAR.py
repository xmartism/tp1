"""
DeepAR Forecasting Tool (GluonTS/MXNet)

Pouzitie - train:
    python3 deepAR.py --mode train \
        --train-dataset train.csv \
        --val-dataset val.csv \
        --target spotreba \
        --horizon 24 \
        --model-dir models/deepAR \
        [--epochs 100] [--lr 1e-3] [--batch-size 64] \
        [--num-layers 3] [--num-cells 40] \
        [--lookback-window N] \
        [--device cpu|gpu] [--patience 10] \
        [--seed 0]

Pouzitie - predict:
    python3 deepAR.py --mode predict \
        --context-dataset history.csv \
        --target spotreba \
        --horizon 24 \
        --lookback-window 96 \
        --model-dir models/deepAR \
        --output output.csv \
        [--seed 0]

    context-dataset: CSV s celou históriou až po začiatok predikovaného okna.
                     Model si sám oreže posledných --lookback-window krokov.

Output output.csv format:
    timestamp,prediction
    2021-01-01T01:00:00,1.23
    2021-01-01T02:00:00,4.56
"""

import argparse
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
from gluonts.model.predictor import Predictor


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping(Callback):
    """Stops training when val loss stops improving and optionally restores the best weights."""

    def __init__(self, patience=5, min_delta=1e-4, restore_best=True):
        """Args:
            patience: epochs to wait before stopping.
            min_delta: minimum improvement to count as progress.
            restore_best: if True, restores weights from the best epoch on stop.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.last_epoch = 0
        self.best_network_params = None

    def on_epoch_end(self, epoch_no, epoch_loss, training_network, trainer, best_epoch_info, ctx):
        """Called by GluonTS after each epoch. Returns False to signal early stop."""
        self.last_epoch = epoch_no + 1

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
    """Read a CSV or TSV file, parse the date column and set it as index."""
    path = Path(filepath)

    if not path.exists():
        print(f"[ERROR] File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    sep = "\t" if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(filepath, sep=sep)

    if date_col not in df.columns:
        raise ValueError(f"Missing column '{date_col}'")
    if target_col not in df.columns:
        raise ValueError(f"Missing column '{target_col}'")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    return df


def infer_freq(df):
    """Infer pandas frequency string from the DatetimeIndex, normalising deprecated aliases."""
    freq = pd.infer_freq(df.index)
    if freq is None:
        return "h"
    freq_map = {
        "H": "h", "60T": "h", "60min": "h",
        "T": "min",
        "M": "ME", "Q": "QE-DEC", "Y": "YE-DEC",
    }
    return freq_map.get(freq, freq)


def build_dataset_entry(df, target_col, covariate_cols):
    """Build a GluonTS-compatible dict entry from a DataFrame."""
    entry = {
        "start": df.index[0],
        "target": df[target_col].values.astype(float),
    }
    if covariate_cols:
        entry["feat_dynamic_real"] = df[covariate_cols].values.T.astype(float)
    return entry


def get_covariate_cols(df, target_col):
    """Return all numeric columns except the target (used as dynamic covariates)."""
    return [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def mode_train(args):
    """Train DeepAR on train split, monitor val loss, serialize model to --model-dir."""
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_df = load_csv_dataset(args.train_dataset, args.date, args.target)
    val_df   = load_csv_dataset(args.val_dataset,   args.date, args.target)

    freq = infer_freq(train_df)
    print(f"[INFO] Train: {len(train_df)} rows | Val: {len(val_df)} rows | Freq: {freq}")

    covariate_cols = get_covariate_cols(train_df, args.target)

    # validation_data in GluonTS is used only for monitoring val loss during training.
    # We concatenate train+val so the model sees the full history up to the val window end,
    # avoiding a cold-start at the beginning of the validation period.
    train_val_df = pd.concat([train_df, val_df])

    train_dataset = ListDataset(
        [build_dataset_entry(train_df,     args.target, covariate_cols)], freq=freq
    )
    val_dataset = ListDataset(
        [build_dataset_entry(train_val_df, args.target, covariate_cols)], freq=freq
    )

    lookback_window = args.lookback_window if args.lookback_window else args.horizon * 4

    ctx = mx.cpu()
    if args.device == "gpu":
        try:
            ctx = mx.gpu()
        except Exception:
            ctx = mx.cpu()

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=1e-4,
        restore_best=True,
    )

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=args.horizon,
        context_length=lookback_window,
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

    print("[INFO] Training...")
    predictor = estimator.train(
        training_data=train_dataset,
        validation_data=val_dataset,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.serialize(model_dir)

    metadata_path = model_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"epoch: {early_stopping.last_epoch}\n")

    print(f"[INFO] Model saved to {model_dir}")
    print(f"[INFO] Metadata saved to {metadata_path}")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def mode_predict(args):
    """Load serialized model, fill missing timestamps and values in context window, trim to lookback, predict."""
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"[ERROR] Model dir '{model_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # Load serialised predictor
    ctx = mx.cpu()
    predictor = Predictor.deserialize(model_dir, ctx=ctx)

    # Load the full history up to the window start
    input_df = load_csv_dataset(args.context_dataset, args.date, args.target)

    # Reindex to a regular frequency and fill all missing data:
    #   - duplicate timestamps are collapsed by averaging their values
    #   - missing timestamps are inserted via reindex (NaN for all columns)
    #   - missing numeric values (both from gaps and originally NaN) are filled
    #     by time-based interpolation, with forward/backward fill as fallback for edge rows
    input_df  = input_df.groupby(input_df.index).mean(numeric_only=True)
    freq_str  = infer_freq(input_df)
    full_idx  = pd.date_range(start=input_df.index[0], end=input_df.index[-1], freq=freq_str)
    input_df  = input_df.reindex(full_idx)
    input_df  = input_df.interpolate(method="time").ffill().bfill()
    input_df.index.name = args.date

    # Trim to the last lookback_window steps
    lookback_window = args.lookback_window if args.lookback_window else args.horizon * 4
    if len(input_df) > lookback_window:
        input_df = input_df.iloc[-lookback_window:]

    covariate_cols = get_covariate_cols(input_df, args.target)
    freq = infer_freq(input_df)

    dataset = ListDataset(
        [build_dataset_entry(input_df, args.target, covariate_cols)],
        freq=freq,
    )

    forecast_it = predictor.predict(dataset)
    forecasts   = list(forecast_it)
    preds       = forecasts[0].quantile(0.5).tolist()[:args.horizon]

    # Generate timestamps for the predicted horizon
    last_ts   = input_df.index[-1]
    inferred  = pd.infer_freq(input_df.index)
    offset    = pd.tseries.frequencies.to_offset(inferred or "h")
    timestamps = [
        (last_ts + offset * (i + 1)).isoformat()
        for i in range(len(preds))
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "timestamp":  timestamps,
        "prediction": preds,
    }).to_csv(output_path, index=False)

    print(f"[INFO] Predictions saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and dispatch to mode_train or mode_predict."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "predict"], required=True)

    parser.add_argument("--train-dataset")
    parser.add_argument("--val-dataset")
    parser.add_argument("--context-dataset", help="predict: full history CSV up to the window start")
    parser.add_argument("--target",         required=True)
    parser.add_argument("--date",           default="date")
    parser.add_argument("--horizon",        type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--num-layers",     type=int,   default=3)
    parser.add_argument("--num-cells",      type=int,   default=40)
    parser.add_argument("--lookback-window", type=int,  default=None,
                        help="number of history steps the model uses as context (default: 4 * horizon)")
    parser.add_argument("--device",         choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--num-samples",    type=int,   default=200)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--seed",           type=int,   default=0)

    parser.add_argument("--model-dir",      help="directory to save (train) or load (predict) the model")

    args = parser.parse_args()

    if args.mode == "train":
        for required in ["train_dataset", "val_dataset", "model_dir"]:
            if getattr(args, required) is None:
                parser.error(f"--mode train requires --{required.replace('_', '-')}")
        mode_train(args)

    elif args.mode == "predict":
        for required in ["context_dataset", "output", "model_dir"]:
            if getattr(args, required) is None:
                parser.error(f"--mode predict requires --{required.replace('_', '-')}")
        mode_predict(args)


if __name__ == "__main__":
    main()