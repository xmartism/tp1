import argparse
import sys
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.missing_values import fill_missing_values

import warnings
import logging

# 1. Ignoruje všetky bežné Python varovania (napr. tie o časových zónach a starom kóde)
warnings.filterwarnings("ignore")

# 2. Nastaví úroveň logovania pre PyTorch Lightning a Darts iba na skutočné CHYBY (ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

torch.set_float32_matmul_precision('medium')


def main():
    parser = argparse.ArgumentParser(description="N-BEATS model pre pipeline")
    parser.add_argument("--train-dataset", required=True, help="Cesta k train.csv")
    parser.add_argument("--val-dataset", required=True, help="Cesta k val.csv")
    parser.add_argument("--test-dataset", required=True, help="Cesta k test.csv")
    parser.add_argument("--target", required=True, help="Nazov cieloveho stlpca")
    parser.add_argument("--date", required=True, help="Nazov stlpca s datumom")
    parser.add_argument("--horizon", type=int, required=True, help="Pocet krokov predikcie")
    parser.add_argument("--output", required=True, help="Cesta k vystupnemu textovemu suboru")

    # NOVÉ PARAMETRE
    parser.add_argument("--lookback-window", type=int, default=None, help="Dlzka vstupneho okna (default: 4 * horizon)")
    parser.add_argument("--seed", type=int, default=42, help="Seed pre reprodukovatelnost")
    args = parser.parse_args()

    # Nastavenie seedu pre PyTorch a Numpy (reprodukovateľnosť)
    pl.seed_everything(args.seed, workers=True)

    print(f"[N-BEATS] Načítavam dáta...")
    try:
        df_train = pd.read_csv(args.train_dataset)
        df_val = pd.read_csv(args.val_dataset)
        df_test = pd.read_csv(args.test_dataset)

        # --- Odstránenie duplicitných časov ---
        df_train = df_train.drop_duplicates(subset=[args.date], keep='first')
        df_val = df_val.drop_duplicates(subset=[args.date], keep='first')
        df_test = df_test.drop_duplicates(subset=[args.date], keep='first')

    except Exception as e:
        print(f"[ERROR] Zlyhalo načítanie CSV súborov: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # 1. Automatické zistenie frekvencie z prvých 10 riadkov
        temp_dates = pd.DatetimeIndex(pd.to_datetime(df_train[args.date], utc=True))
        zistena_frekvencia = pd.infer_freq(temp_dates[:10])

        if zistena_frekvencia is None:
            print("[WARN] Nepodarilo sa zistiť frekvenciu automaticky. Použijem default 'D' (Dni).")
            zistena_frekvencia = 'D'
        else:
            print(f"[N-BEATS] Automaticky zistená frekvencia dát: {zistena_frekvencia}")

        # 2. Vytvorenie TimeSeries s dynamickou frekvenciou
        series_train = TimeSeries.from_dataframe(df_train, time_col=args.date, value_cols=args.target,
                                                 fill_missing_dates=True, freq=zistena_frekvencia)
        series_val = TimeSeries.from_dataframe(df_val, time_col=args.date, value_cols=args.target,
                                               fill_missing_dates=True, freq=zistena_frekvencia)
        series_test = TimeSeries.from_dataframe(df_test, time_col=args.date, value_cols=args.target,
                                                fill_missing_dates=True, freq=zistena_frekvencia)

        series_train = fill_missing_values(series_train, fill='auto')
        series_val = fill_missing_values(series_val, fill='auto')
        series_test = fill_missing_values(series_test, fill='auto')
    except Exception as e:
        print(f"[ERROR] Problém s vytváraním TimeSeries: {e}", file=sys.stderr)
        sys.exit(1)

    # Spojenie train a val pre potreby histórie pri predikcii
    train_val_series = series_train.append(series_val)

    # NOVÉ: Lookback window logika
    lookback = args.lookback_window if args.lookback_window is not None else 4 * args.horizon

    if len(series_train) <= lookback + args.horizon:
        print(f"[ERROR] Trénovací set je príliš krátky pre lookback {lookback} a horizon {args.horizon}.",
              file=sys.stderr)
        sys.exit(1)

    # NOVÉ: Nastavenie Early Stopping
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Kolko epôch bez zlepšenia sa má čakať
        min_delta=0.001,  # Minimálne zlepšenie, aby sa počítalo ako "zlepšenie"
        mode="min",
    )

    print(f"[N-BEATS] Trénujem model (Lookback: {lookback}, Horizon: {args.horizon}, Seed: {args.seed})...")
    model = NBEATSModel(
        input_chunk_length=lookback,
        output_chunk_length=args.horizon,
        generic_architecture=True,
        num_stacks=30,
        layer_widths=512,
        n_epochs=50,  # Zvýšené na 50, aby mal early stopping šancu zafungovať (pôvodne bolo 3)
        batch_size=1024,
        random_state=args.seed,
        pl_trainer_kwargs={
            "accelerator": "auto",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": False,
            "logger": False,
            "callbacks": [early_stopper]  # Pridaný callback pre Early Stopping
        }
    )

    # NOVÉ: Do fit pridávame aj val_series, inak nebude fungovať val_loss pri Early Stopping
    model.fit(series=series_train, val_series=series_val, verbose=False)

    # Zistenie, na ktorej epoche sa model zastavil
    stopped_epoch = model.trainer.current_epoch
    print(f"[N-BEATS] Trénovanie skončilo na epoche: {stopped_epoch}")

    print(f"[N-BEATS] Vytváram predikciu...")
    # Predikujeme z train_val histórie
    prediction = model.predict(n=args.horizon, series=train_val_series)

    # Extrahujeme predikcie do obyčajného Python listu
    pred_list = prediction.values().flatten().tolist()

    print(f"[N-BEATS] Zapisujem výsledky do {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pred_list, f)

    # NOVÉ: Uloženie metadát o ukončenej epoche
    out_path = Path(args.output)
    metadata_file = out_path.parent / "nbeats_metadata.txt"

    with open(metadata_file, "w", encoding="utf-8") as meta_f:
        meta_f.write(f"epoch: {stopped_epoch}\n")
    print(f"[N-BEATS] Metadáta uložené do {metadata_file}")

    print(f"[N-BEATS] Hotovo!")


if __name__ == "__main__":
    main()