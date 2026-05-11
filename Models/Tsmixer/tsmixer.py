import argparse
import logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
import time
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import sys
from pathlib import Path
import models


DATASET_HPARAMS = {
    "ETTm2":          dict(lr=0.001,  n_block=2, dropout=0.9, ff_dim=64),
    "weather":        dict(lr=0.0001, n_block=4, dropout=0.3, ff_dim=32),
    "electricity":    dict(lr=0.0001, n_block=4, dropout=0.7, ff_dim=64),
    "traffic":        dict(lr=0.0001, n_block=8, dropout=0.7, ff_dim=64),
    "SinTwentyWaves": dict(lr=0.001,  n_block=2, dropout=0.3, ff_dim=64),
}

DEFAULT_HPARAMS = dict(lr=0.0001, n_block=2, dropout=0.1, ff_dim=64)

class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch = epoch + 1  # epoch je 0-indexed


def load_csv(filepath: str, date_col: str) -> pd.DataFrame:
    """Read a CSV/TSV file, parse the date column and set it as the index."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    sep = "\t" if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(filepath, sep=sep)

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in {filepath}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regularise the DatetimeIndex and fill missing numeric values.

    Steps:
      1. Collapse duplicate timestamps by averaging numeric columns.
      2. Reindex to a uniform frequency (inferred from the index).
      3. Interpolate numerics with 'time', then ffill/bfill for edge rows.
    """
    df = df.groupby(df.index).mean(numeric_only=True)

    freq_str = pd.infer_freq(df.index)
    if freq_str is None:
        freq_str = "h"

    full_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq_str)
    df = df.reindex(full_idx)
    df = df.interpolate(method="time").ffill().bfill()
    df.index.name = None  # keep index anonymous; date_col info is in calling code
    return df


def infer_freq(df: pd.DataFrame, freq_override: str = None) -> str:
    """Return a pandas offset string for the DataFrame's DatetimeIndex."""
    if freq_override:
        return freq_override
    freq = pd.infer_freq(df.index)
    if freq is None:
        return "h"
    alias_map = {
        "H": "h", "60T": "h", "60min": "h",
        "T": "min",
        "M": "ME", "Q": "QE-DEC", "Y": "YE-DEC",
    }
    return alias_map.get(freq, freq)


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric columns (used as model features)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def df_to_tf_dataset(
        df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        target_col: str,
        batch_size: int,
        shuffle: bool,
) -> tuple[tf.data.Dataset, int, int]:
    """
    Convert a (scaled) DataFrame to a tf.data.Dataset of (inputs, labels) pairs.

    Returns:
        dataset      — tf.data.Dataset
        n_feature    — total number of numeric columns (model input width)
        target_idx   — column index of the target within the numeric columns
    """
    numeric_cols = get_numeric_cols(df)
    if target_col not in numeric_cols:
        raise ValueError(
            f"Target column '{target_col}' not found or not numeric. "
            f"Available numeric columns: {numeric_cols}"
        )

    target_idx = numeric_cols.index(target_col)
    n_feature = len(numeric_cols)
    data = df[numeric_cols].values.astype(np.float32)

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=seq_len + pred_len,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    ds = ds.map(lambda w: (w[:, :seq_len, :], w[:, seq_len:, target_idx:target_idx + 1]))

    return ds, n_feature, target_idx

def parse_args():
    parser = argparse.ArgumentParser(
        description="TSMixer – pipeline-compatible wrapper"
    )

    parser.add_argument("--mode", choices=["train", "predict"], required=True)

    # Shared
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--date", default="date", help="Date column name")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--lookback-window", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Directory to save (train) or load (predict) the model")

    # Train-only
    parser.add_argument("--train-dataset", help="Path to train CSV (train mode)")
    parser.add_argument("--val-dataset", help="Path to val CSV (train mode)")

    # Predict-only
    parser.add_argument("--context-dataset", help="Full history CSV up to window start (predict mode)")
    parser.add_argument("--output", help="Output CSV path (predict mode)")

    # Optional hyperparameter overrides
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset name for hyperparameter preset (e.g. 'weather')")
    parser.add_argument("--freq", default=None,
                        help="Pandas frequency string (e.g. 'B', 'h', 'D'). If omitted, inferred from data.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--norm-type", default="B", choices=["L", "B"])
    parser.add_argument("--activation", default="relu", choices=["relu", "gelu"])
    parser.add_argument("--model", default="tsmixer_rev_in")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "train":
        for required in ["train_dataset", "val_dataset"]:
            if getattr(args, required) is None:
                print(
                    f"[ERROR] --mode train requires --{required.replace('_', '-')}",
                    file=sys.stderr,
                )
                sys.exit(1)
        mode_train(args)

    elif args.mode == "predict":
        for required in ["context_dataset", "output"]:
            if getattr(args, required) is None:
                print(
                    f"[ERROR] --mode predict requires --{required.replace('_', '-')}",
                    file=sys.stderr,
                )
                sys.exit(1)
        mode_predict(args)

def mode_train(args):
    """Train TSMixer on train/val splits and serialize the weights to --model-dir."""
    tf.keras.utils.set_random_seed(args.seed)

    hparams = DATASET_HPARAMS.get(args.dataset_name, DEFAULT_HPARAMS)
    lr       = hparams["lr"]
    n_block  = hparams["n_block"]
    dropout  = hparams["dropout"]
    ff_dim   = hparams["ff_dim"]

    print(f"[TSMixer] mode=train | dataset_name={args.dataset_name or 'unknown'} "
        f"| lr={lr} n_block={n_block} dropout={dropout} ff_dim={ff_dim} "
        f"| lookback={args.lookback_window} horizon={args.horizon} seed={args.seed}"
    )

    train_df = load_csv(args.train_dataset, args.date)
    val_df   = load_csv(args.val_dataset,   args.date)

    train_ds, n_feature, _ = df_to_tf_dataset(
        train_df, args.lookback_window, args.horizon,
        args.target, args.batch_size, shuffle=True,
    )
    val_ds, _, _ = df_to_tf_dataset(
        val_df, args.lookback_window, args.horizon,
        args.target, args.batch_size, shuffle=False,
    )

    numeric_cols = get_numeric_cols(train_df)
    target_slice = slice(
        numeric_cols.index(args.target),
        numeric_cols.index(args.target) + 1,
    )

    build_model = getattr(models, args.model).build_model
    model = build_model(
        input_shape=(args.lookback_window, n_feature),
        pred_len=args.horizon,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=dropout,
        n_block=n_block,
        ff_dim=ff_dim,
        target_slice=target_slice,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(model_dir / "tsmixer_best")

    epoch_tracker = EpochTracker()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=1,
        ),
        epoch_tracker,
    ]

    t0 = time.time()
    model.fit(
        train_ds,
        epochs=args.train_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"[TSMixer] Training finished in {elapsed:.1f}s")

    config_path = model_dir / "model_config.npz"
    np.savez(
        config_path,
        n_feature=np.array(n_feature),
        target_idx=np.array(numeric_cols.index(args.target)),
        n_block=np.array(n_block),
        dropout=np.array(dropout),
        ff_dim=np.array(ff_dim),
        feature_cols=np.array(numeric_cols),
    )

    # metadata.txt — consistent location expected by evaluate.py
    metadata_path = model_dir / "metadata.txt"
    metadata_path.write_text(f"epoch: {epoch_tracker.last_epoch}\n")

    print(f"[TSMixer] Model weights  -> {ckpt_path}")
    print(f"[TSMixer] Model config   -> {config_path}")
    print(f"[TSMixer] Metadata       -> {metadata_path}")

def mode_predict(args):
    """Load serialized model, trim context to lookback window, predict horizon steps."""
    tf.keras.utils.set_random_seed(args.seed)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"[ERROR] Model dir '{model_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # --- Restore training config ---
    config = np.load(model_dir / "model_config.npz")
    n_feature = int(config["n_feature"])
    target_idx = int(config["target_idx"])
    n_block = int(config["n_block"])
    dropout = float(config["dropout"])
    ff_dim = int(config["ff_dim"])

    lookback = args.lookback_window
    horizon = args.horizon

    target_slice = slice(target_idx, target_idx + 1)

    hparams = DATASET_HPARAMS.get(args.dataset_name, DEFAULT_HPARAMS)
    lr = hparams["lr"]

    build_model = getattr(models, args.model).build_model
    model = build_model(
        input_shape=(lookback, n_feature),
        pred_len=horizon,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=dropout,
        n_block=n_block,
        ff_dim=ff_dim,
        target_slice=target_slice,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

    ckpt_path = str(model_dir / "tsmixer_best")
    model.load_weights(ckpt_path)

    input_df = load_csv(args.context_dataset, args.date)
    input_df = fill_missing(input_df)

    numeric_cols = config["feature_cols"].tolist()
    input_df = input_df.reindex(columns=numeric_cols)

    # Trim to last lookback_window steps
    if len(input_df) > lookback:
        input_df = input_df.iloc[-lookback:]

    if len(input_df) < lookback:
        print(
            f"[WARN] Context length ({len(input_df)}) < lookback window ({lookback}). "
            "Padding with zeros at the front.",
            file=sys.stderr,
        )
        freq_str = infer_freq(input_df, args.freq)
        pad_end = input_df.index[0] - pd.tseries.frequencies.to_offset(freq_str)
        pad_idx = pd.date_range(end=pad_end, periods=lookback - len(input_df), freq=freq_str)
        pad = pd.DataFrame(
            np.zeros((len(pad_idx), len(numeric_cols))),
            index=pad_idx,
            columns=numeric_cols,
        )
        input_df = pd.concat([pad, input_df])

    # --- Run inference ---
    x = input_df.values.astype(np.float32)  # (lookback, n_feature)
    x = x[np.newaxis, :, :]  # (1, lookback, n_feature)
    preds = model.predict(x, verbose=0)  # (1, horizon, 1)
    preds_flat = preds[0, :, 0].tolist()[:horizon]

    # --- Generate future timestamps ---
    last_ts = input_df.index[-1] if isinstance(input_df.index, pd.DatetimeIndex) else None
    if last_ts is not None:
        freq_str = infer_freq(input_df, args.freq) if isinstance(input_df.index, pd.DatetimeIndex) else "h"
        offset = pd.tseries.frequencies.to_offset(freq_str)
        timestamps = [
            (last_ts + offset * (i + 1)).isoformat()
            for i in range(len(preds_flat))
        ]
    else:
        timestamps = [f"step_{i}" for i in range(len(preds_flat))]

    # --- Save output CSV ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": timestamps, "prediction": preds_flat}).to_csv(
        output_path, index=False
    )
    print(f"[TSMixer] Predictions saved to {output_path}")


if __name__ == "__main__":
    main()