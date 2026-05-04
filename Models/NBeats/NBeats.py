import argparse
import sys
import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.missing_values import fill_missing_values

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

torch.set_float32_matmul_precision('medium')


# ---------------------------------------------------------------------------
# Dátové funkcie
# ---------------------------------------------------------------------------

def load_data_to_timeseries(filepath, date_col, target_col):
    """Načíta CSV a vytvorí TimeSeries pre target aj kovariáty."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=[date_col], keep='first')

    # Automatické zistenie frekvencie
    temp_dates = pd.DatetimeIndex(pd.to_datetime(df[date_col], utc=True))
    zistena_frekvencia = pd.infer_freq(temp_dates[:10]) or 'D'

    # Kovariáty
    covariate_cols = [
        col for col in df.columns
        if col not in [date_col, target_col] and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Target series
    target_series = TimeSeries.from_dataframe(
        df, time_col=date_col, value_cols=target_col,
        fill_missing_dates=True, freq=zistena_frekvencia
    )
    target_series = fill_missing_values(target_series, fill='auto')

    # Covariate series
    cov_series = None
    if covariate_cols:
        cov_series = TimeSeries.from_dataframe(
            df, time_col=date_col, value_cols=covariate_cols,
            fill_missing_dates=True, freq=zistena_frekvencia
        )
        cov_series = fill_missing_values(cov_series, fill='auto')

    return target_series, cov_series


# ---------------------------------------------------------------------------
# Mód: TRAIN
# ---------------------------------------------------------------------------

def mode_train(args):
    pl.seed_everything(args.seed, workers=True)

    print("[N-BEATS] Načítavam dáta na trénovanie...")
    train_target, train_cov = load_data_to_timeseries(args.train_dataset, args.date, args.target)
    val_target, val_cov = load_data_to_timeseries(args.val_dataset, args.date, args.target)

    lookback = args.lookback_window if args.lookback_window else 4 * args.horizon

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-4,
        mode="min",
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(save_dir=str(model_dir), name="logs")

    model = NBEATSModel(
        input_chunk_length=lookback,
        output_chunk_length=args.horizon,
        generic_architecture=True,
        num_stacks=30,
        layer_widths=512,
        n_epochs=50,
        batch_size=64,
        random_state=args.seed,
        pl_trainer_kwargs={
            "accelerator": "auto",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": False,
            "logger": csv_logger,
            "callbacks": [early_stopper],
            "gradient_clip_val": 1.0
        }
    )

    fit_kwargs = {"series": train_target, "val_series": val_target, "verbose": False}
    if train_cov is not None:
        fit_kwargs["past_covariates"] = train_cov
        fit_kwargs["val_past_covariates"] = val_cov

    print("[N-BEATS] Spúšťam trénovanie...")
    model.fit(**fit_kwargs)

    # Uloženie modelu na disk
    model_path = model_dir / "nbeats_model.pt"
    model.save(str(model_path))
    print(f"[N-BEATS] Model uložený do {model_path}")

    # Uloženie metadát
    stopped_epoch = model.trainer.current_epoch
    with open(model_dir / "metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"epoch: {stopped_epoch}\n")


# ---------------------------------------------------------------------------
# Mód: PREDICT
# ---------------------------------------------------------------------------

def mode_predict(args):
    pl.seed_everything(args.seed, workers=True)

    model_dir = Path(args.model_dir)
    model_path = model_dir / "nbeats_model.pt"

    if not model_path.exists():
        print(f"[ERROR] Model nenasiel: {model_path}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # FIX PRE PYTORCH 2.6+
    # PyTorch 2.6 zmenil defaultné správanie načítavania na weights_only=True.
    # Urobíme dočasný monkeypatch, aby sme povolili načítanie nášho modelu.
    # -----------------------------------------------------------------------
    import torch
    original_load = torch.load

    def patched_load(*a, **kw):
        kw['weights_only'] = False
        return original_load(*a, **kw)

    torch.load = patched_load

    try:
        # Načítanie uloženého modelu
        model = NBEATSModel.load(str(model_path))
    finally:
        # Vrátenie pôvodnej funkcie torch.load hneď po načítaní
        torch.load = original_load
    # -----------------------------------------------------------------------

    # Načítanie histórie (context)
    context_target, context_cov = load_data_to_timeseries(args.context_dataset, args.date, args.target)

    predict_kwargs = {"n": args.horizon, "series": context_target}
    if context_cov is not None:
        predict_kwargs["past_covariates"] = context_cov

    # Predikcia
    prediction = model.predict(**predict_kwargs)

    # Formátovanie na rovnaký výstup ako má DeepAR (CSV: timestamp, prediction)
    timestamps = [ts.isoformat() for ts in prediction.time_index]
    preds = prediction.values().flatten().tolist()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "timestamp": timestamps,
        "prediction": preds
    }).to_csv(output_path, index=False)

    print(f"[N-BEATS] Predikcia uložená do {output_path}")


# ---------------------------------------------------------------------------
# Hlavný parser (zhodný s pipeline)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="N-BEATS model kompatibilný s pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)

    parser.add_argument("--train-dataset", help="Cesta k train.csv")
    parser.add_argument("--val-dataset", help="Cesta k val.csv")
    parser.add_argument("--context-dataset", help="História pre predikciu")
    parser.add_argument("--target", required=True, help="Názov cieľového stĺpca")
    parser.add_argument("--date", default="date", help="Názov stĺpca s dátumom")
    parser.add_argument("--horizon", type=int, required=True, help="Počet krokov predikcie")
    parser.add_argument("--lookback-window", type=int, default=None)
    parser.add_argument("--output", help="Cesta k CSV výstupu (pre predict)")
    parser.add_argument("--model-dir", required=True, help="Priečinok pre uloženie/načítanie modelu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "train":
        if not args.train_dataset or not args.val_dataset:
            parser.error("--mode train vyžaduje --train-dataset a --val-dataset")
        mode_train(args)
    elif args.mode == "predict":
        if not args.context_dataset or not args.output:
            parser.error("--mode predict vyžaduje --context-dataset a --output")
        mode_predict(args)


if __name__ == "__main__":
    main()