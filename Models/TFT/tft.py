import argparse
import sys
import torch
import pandas as pd
import numpy as np
import json
import logging
import warnings

from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.missing_values import fill_missing_values

# 1. Ignorovanie varovaní a nastavenie logovania (zhodné s N-BEATS)
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

torch.set_float32_matmul_precision('medium')


def main():
    parser = argparse.ArgumentParser(description="TFT model pre pipeline")
    parser.add_argument("--train-dataset", required=True, help="Cesta k train.csv")
    parser.add_argument("--val-dataset", required=True, help="Cesta k val.csv")
    parser.add_argument("--test-dataset", required=True, help="Cesta k test.csv")
    parser.add_argument("--target", required=True, help="Nazov cieloveho stlpca")
    parser.add_argument("--date", required=True, help="Nazov stlpca s datumom")
    parser.add_argument("--horizon", type=int, required=True, help="Pocet krokov predikcie")
    parser.add_argument("--output", required=True, help="Cesta k vystupnemu textovemu suboru")
    args = parser.parse_args()

    print(f"[TFT] Načítavam dáta...")
    print(f"[TFT] Načítavam dáta...")
    try:
        df_train = pd.read_csv(args.train_dataset)
        df_val = pd.read_csv(args.val_dataset)
        df_test = pd.read_csv(args.test_dataset)

        # Odstránenie duplicitných časov
        df_train = df_train.drop_duplicates(subset=[args.date], keep='first')
        df_val = df_val.drop_duplicates(subset=[args.date], keep='first')
        df_test = df_test.drop_duplicates(subset=[args.date], keep='first')

    except Exception as e:
        print(f"[ERROR] Zlyhalo načítanie CSV súborov: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Zistenie frekvencie
        temp_dates = pd.DatetimeIndex(pd.to_datetime(df_train[args.date], utc=True))
        zistena_frekvencia = pd.infer_freq(temp_dates[:10])

        if zistena_frekvencia is None:
            print("[WARN] Nepodarilo sa zistiť frekvenciu automaticky. Použijem default 'D'.")
            zistena_frekvencia = 'D'
        else:
            print(f"[TFT] Automaticky zistená frekvencia dát: {zistena_frekvencia}")

        # Vytvorenie TimeSeries
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

    # Spojenie histórie pre predikciu
    train_val_series = series_train.append(series_val)

    # Nastavenie Lookbacku (rovnako ako N-BEATS: horizon * 3)
    lookback = args.horizon * 3

    if len(series_train) <= lookback + args.horizon:
        print(f"[ERROR] Trénovací set je príliš krátky pre lookback {lookback}.", file=sys.stderr)
        sys.exit(1)

    print(f"[TFT] Trénujem model (Lookback: {lookback}, Horizon: {args.horizon})...")

    # Model TFT s minimálnym preprocessingom (žiadne externé scalery, len interné nastavenia)
    model = TFTModel(
        input_chunk_length=lookback,
        output_chunk_length=args.horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=32,
        n_epochs=2,  # Nastavené na 3 pre rýchlosť pipeline, podobne ako N-BEATS
        add_relative_index=True,  # Dôležité pre TFT, ak nepoužívame iné statické kovariáty
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "cuda",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": False,
            "logger": False
        }
    )

    # Trénujeme (predpokladáme oškálované dáta z pipeline, ak sú potrebné)
    model.fit(series_train, verbose=False)

    print(f"[TFT] Vytváram predikciu...")
    # Predikujeme z train_val histórie
    prediction = model.predict(n=args.horizon, series=train_val_series)

    # Extrahujeme predikcie do listu pre JSON
    pred_list = prediction.values().flatten().tolist()

    print(f"[TFT] Zapisujem výsledky do {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pred_list, f)

    print(f"[TFT] Hotovo!")


if __name__ == "__main__":
    main()