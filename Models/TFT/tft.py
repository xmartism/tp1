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
from pytorch_lightning.callbacks import Callback

from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.missing_values import fill_missing_values

import warnings
import logging

# 1. Ignorovanie varovaní a nastavenie logovania
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

torch.set_float32_matmul_precision('medium')


# ---------------------------------------------------------------------------
# Vlastný callback na výpis progresu pre pipeline logy
# ---------------------------------------------------------------------------
class PipelineProgressCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs
            total_batches = trainer.num_training_batches

            if total_batches and total_batches > 0:
                percent = (batch_idx / total_batches) * 100
                print(
                    f"[TFT] Tréning - Epocha {epoch}/{max_epochs} | Dávka {batch_idx}/{total_batches} ({percent:.1f}%)")


# ---------------------------------------------------------------------------
# Dátové funkcie (zhodné s N-BEATS pre férové porovnanie)
# ---------------------------------------------------------------------------
def load_data_to_timeseries(filepath, date_col, target_col, freq=None):
    """Načíta CSV a vytvorí TimeSeries pre target aj kovariáty."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=[date_col], keep='first')

    # Automatické zistenie frekvencie
    temp_dates = pd.DatetimeIndex(pd.to_datetime(df[date_col], utc=True))
    zistena_frekvencia = freq or pd.infer_freq(temp_dates[:10]) or 'D'

    # Kovariáty (všetky numerické okrem date a target)
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

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tft_model.pt"

    # --- ZMENA 1: Kontrola, či už je model kompletne dotrénovaný ---
    if model_path.exists():
        print(f"[TFT] Hotový model už existuje v '{model_path}'. Preskakujem trénovanie.")
        return
    # ---------------------------------------------------------------

    print("[TFT] Načítavam dáta na trénovanie...")
    train_target, train_cov = load_data_to_timeseries(args.train_dataset, args.date, args.target, args.freq)
    val_target, val_cov = load_data_to_timeseries(args.val_dataset, args.date, args.target, args.freq)

    lookback = args.lookback_window if args.lookback_window else 4 * args.horizon

    # Pridanie Early Stoppingu
    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.00,
        mode='min',
    )

    my_progress = PipelineProgressCallback()
    csv_logger = CSVLogger(save_dir=str(model_dir), name="logs")

    max_epochs = 100

    # Model TFT
    model = TFTModel(
        input_chunk_length=lookback,
        output_chunk_length=args.horizon,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=2,
        dropout=0.1,
        batch_size=32,
        n_epochs=max_epochs,
        add_relative_index=True,
        random_state=args.seed,

        # --- ZMENA 2: Parametre pre automatické načítanie checkpointov ---
        model_name="tft_checkpoints",
        work_dir=str(model_dir),
        save_checkpoints=True,
        force_reset=False,  # Zabezpečí, že ak existuje checkpoint, model z neho bude pokračovať
        # -----------------------------------------------------------------

        pl_trainer_kwargs={
            "accelerator": "auto",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": False,
            "logger": csv_logger,
            "callbacks": [my_stopper, my_progress]
        }
    )

    fit_kwargs = {"series": train_target, "val_series": val_target, "verbose": False}
    if train_cov is not None:
        fit_kwargs["past_covariates"] = train_cov
        fit_kwargs["val_past_covariates"] = val_cov

    print("[TFT] Spúšťam trénovanie...")
    model.fit(**fit_kwargs)

    # Uloženie finálneho modelu na disk
    model.save(str(model_path))
    print(f"[TFT] Model uložený do {model_path}")

    # Uloženie metadát
    stopped_epoch = max_epochs
    if hasattr(model, 'trainer') and model.trainer is not None:
        stopped_epoch = model.trainer.current_epoch
    if my_stopper.stopped_epoch > 0:
        stopped_epoch = my_stopper.stopped_epoch

    with open(model_dir / "metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"epoch: {stopped_epoch}\n")


# ---------------------------------------------------------------------------
# Mód: PREDICT
# ---------------------------------------------------------------------------
def mode_predict(args):
    pl.seed_everything(args.seed, workers=True)

    model_dir = Path(args.model_dir)
    model_path = model_dir / "tft_model.pt"

    if not model_path.exists():
        print(f"[ERROR] Model nenajdeny: {model_path}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # FIX PRE PYTORCH 2.6+
    # -----------------------------------------------------------------------
    import torch
    original_load = torch.load

    def patched_load(*a, **kw):
        kw['weights_only'] = False
        return original_load(*a, **kw)

    torch.load = patched_load

    try:
        model = TFTModel.load(str(model_path))
    finally:
        torch.load = original_load
    # -----------------------------------------------------------------------

    # Načítanie histórie (context)
    context_target, context_cov = load_data_to_timeseries(args.context_dataset, args.date, args.target, args.freq)

    predict_kwargs = {"n": args.horizon, "series": context_target}
    if context_cov is not None:
        predict_kwargs["past_covariates"] = context_cov

    # Predikcia
    print("[TFT] Vytváram predikciu...")
    prediction = model.predict(**predict_kwargs)

    # Formátovanie na rovnaký výstup ako má N-BEATS (CSV: timestamp, prediction)
    timestamps = [ts.isoformat() for ts in prediction.time_index]
    preds = prediction.values().flatten().tolist()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "timestamp": timestamps,
        "prediction": preds
    }).to_csv(output_path, index=False)

    print(f"[TFT] Predikcia uložená do {output_path}")


# ---------------------------------------------------------------------------
# Hlavný parser (zhodný s pipeline)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TFT model kompatibilný s pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)

    parser.add_argument("--train-dataset", help="Cesta k train.csv")
    parser.add_argument("--val-dataset", help="Cesta k val.csv")
    parser.add_argument("--context-dataset", help="História pre predikciu")
    parser.add_argument("--target", required=True, help="Nazov cieloveho stlpca")
    parser.add_argument("--date", default="date", help="Nazov stlpca s datumom")
    parser.add_argument("--horizon", type=int, required=True, help="Pocet krokov predikcie")
    parser.add_argument("--lookback-window", type=int, default=None)
    parser.add_argument("--freq", default=None,
                        help="Frekvencia dát (napr. 'B', 'h', 'D'). Ak je vynechané, odvodí sa z dát.")
    parser.add_argument("--output", help="Cesta k CSV vystupu (pre predict)")
    parser.add_argument("--model-dir", required=True, help="Priecinok pre ulozenie/nacitanie modelu")
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