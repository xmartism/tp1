import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mae, mse, mape

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def check_device(requested_device):
    if requested_device == "gpu" and torch.cuda.is_available():
        return "gpu", [0]
    return "cpu", "auto"


def run_cli():
    parser = argparse.ArgumentParser(description="N-BEATS Forecaster CLI")

    # Povinné parametre
    parser.add_argument("--data", type=str, required=True, help="Cesta k CSV")
    parser.add_argument("--target", type=str, required=True, help="Cieľový stĺpec")
    parser.add_argument("--time_col", type=str, required=True, help="Stĺpec s časom")
    parser.add_argument("--freq", type=str, required=True, help="Frekvencia (napr. 'H', 'D')")
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)

    # Voliteľné
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    args = parser.parse_args()

    # 1. Načítanie dát
    try:
        df = pd.read_csv(args.data)

        # OŠETRENIE: Ak stĺpec time_col neexistuje (prípad tvojho sínusu)
        if args.time_col not in df.columns:
            print(f"--- INFO: Stĺpec '{args.time_col}' nenájdený. Vytváram číselný index. ---")
            df[args.time_col] = range(len(df))
            freq_val = None  # Darts pri číselnom indexe nepotrebuje freq string
        else:
            # Ak je to dátum, skúsime ho sformátovať
            if not args.freq.isdigit():
                df[args.time_col] = pd.to_datetime(df[args.time_col])
            freq_val = args.freq

        series = TimeSeries.from_dataframe(df, args.time_col, args.target, freq=freq_val)
        series = fill_missing_values(series, fill='auto')
    except Exception as e:
        print(f"CHYBA PRI NAČÍTANÍ: {e}")
        sys.exit(1)

    accel, dev = check_device(args.device)

    # 2. Príprava dát (Tvoj pôvodný Scaler!)
    train, val = series.split_before(len(series) - args.pred_len)
    scaler = Scaler()  # TOTO JE KĽÚČ K SPRÁVNYM VÝSLEDKOM
    train_sc = scaler.fit_transform(train)

    # 3. Model N-BEATS (Tvoje overené parametre)
    model = NBEATSModel(
        input_chunk_length=args.input_len,
        output_chunk_length=args.pred_len if args.pred_len <= args.input_len else args.input_len,
        generic_architecture=True,
        num_stacks=30,
        layer_widths=512,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        pl_trainer_kwargs={"accelerator": accel, "devices": dev, "enable_progress_bar": True}
    )

    print(f"Trénujem N-BEATS...")
    model.fit(train_sc)

    # 4. Predikcia
    pred_sc = model.predict(len(val))
    prediction = scaler.inverse_transform(pred_sc)

    print(f"\n--- VÝSLEDOK ---")
    print(f"MAE: {mae(val, prediction):.4f}")
    print(f"MAPE: {mape(val, prediction):.2f}%")

    # 5. Graf
    plt.figure(figsize=(12, 6))
    train[-args.input_len * 2:].plot(label='Train')
    val.plot(label='Realita', color='black')
    prediction.plot(label='N-BEATS', color='red')
    plt.title(f"Predikcia: {args.target}")
    plt.show()


if __name__ == "__main__":
    run_cli()