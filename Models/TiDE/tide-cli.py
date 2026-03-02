import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Importy z Darts
from darts import TimeSeries
from darts.models import TiDEModel
# ... (ostatné importy ostávajú rovnaké)
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mae, mse
from sklearn.preprocessing import StandardScaler


def check_device(requested_device):
    if requested_device == "gpu":
        if torch.cuda.is_available():
            print("--- INFO: GPU je dostupné. ---")
            return "gpu", [0]
        else:
            print("--- VAROVANIE: GPU nedostupné, prepínam na CPU. ---")
    return "cpu", "auto"


def run_cli():
    parser = argparse.ArgumentParser(
        description="TiDE Forecaster CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- POVINNÉ ---
    mand = parser.add_argument_group('Povinné parametre')
    mand.add_argument("--data", type=str, required=True, help="Cesta k CSV")
    mand.add_argument("--target", type=str, required=True, help="Cieľový stĺpec")
    mand.add_argument("--time_col", type=str, required=True, help="Stĺpec s časom/indexom")
    mand.add_argument("--freq", type=str, required=True, help="Frekvencia (napr. 'H', 'D' alebo číslo '1')")
    mand.add_argument("--input_len", type=int, required=True, help="Lookback window")
    mand.add_argument("--pred_len", type=int, required=True, help="Predpoveď")
    mand.add_argument("--epochs", type=int, required=True, help="Počet epoch")

    # --- VOLITEĽNÉ ---
    opt = parser.add_argument_group('Voliteľné parametre')
    opt.add_argument("--date_format", type=str, default=None, help="Formát dátumu (napr. '%%d.%%m.%%Y %%H:%%M:%%S')")
    opt.add_argument("--batch_size", type=int, default=32, help="Batch size")
    opt.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    opt.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Zariadenie")

    args = parser.parse_args()

    # 1. Načítanie a spracovanie frekvencie
    try:
        df = pd.read_csv(args.data)

        # Ak je freq číslo, interpretujeme index ako celočíselný
        if args.freq.isdigit():
            freq_val = int(args.freq)
            # Uistíme sa, že time_col sú čísla
            df[args.time_col] = pd.to_numeric(df[args.time_col])
        else:
            freq_val = args.freq
            df[args.time_col] = pd.to_datetime(df[args.time_col], format=args.date_format)

        series = TimeSeries.from_dataframe(df, args.time_col, args.target, freq=freq_val, fill_missing_dates=True)
        series = fill_missing_values(series, fill='auto')
    except Exception as e:
        print(f"CHYBA: {e}")
        sys.exit(1)

    accel, dev = check_device(args.device)

    # 2. Model a Tréning
    train, val = series.split_before(len(series) - args.pred_len)
    scaler = Scaler(scaler=StandardScaler())
    train_sc = scaler.fit_transform(train)

    model = TiDEModel(
        input_chunk_length=args.input_len,
        output_chunk_length=args.pred_len,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        pl_trainer_kwargs={"accelerator": accel, "devices": dev}
    )

    model.fit(train_sc)

    # 3. Vyhodnotenie
    pred_sc = model.predict(len(val))
    prediction = scaler.inverse_transform(pred_sc)

    print(f"Výsledné MAE: {mae(val, prediction):.4f}")

    prediction.plot(label='TiDE', color='magenta')
    val.plot(label='Realita')
    plt.show()


if __name__ == "__main__":
    run_cli()