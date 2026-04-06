import argparse
import sys
import torch
import pandas as pd
import numpy as np
import json
import logging
import warnings
from pathlib import Path

from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.missing_values import fill_missing_values
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback

# 1. Ignorovanie varovaní a nastavenie logovania
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

torch.set_float32_matmul_precision('medium')


# 2. Vlastný callback na výpis progresu pre pipeline logy
class PipelineProgressCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Vypíše progres každých 10 dávok (môžeš zmeniť podľa veľkosti datasetu)
        if batch_idx % 10 == 0:
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs
            total_batches = trainer.num_training_batches

            # Ochrana proti deleniu nulou
            if total_batches and total_batches > 0:
                percent = (batch_idx / total_batches) * 100
                print(
                    f"[TFT] Tréning - Epocha {epoch}/{max_epochs} | Dávka {batch_idx}/{total_batches} ({percent:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="TFT model pre pipeline")
    parser.add_argument("--train-dataset", required=True, help="Cesta k train.csv")
    parser.add_argument("--val-dataset", required=True, help="Cesta k val.csv")
    parser.add_argument("--test-dataset", required=True, help="Cesta k test.csv")
    parser.add_argument("--target", required=True, help="Nazov cieloveho stlpca")
    parser.add_argument("--date", required=True, help="Nazov stlpca s datumom")
    parser.add_argument("--horizon", type=int, required=True, help="Pocet krokov predikcie")
    parser.add_argument("--lookback-window", type=int, required=True, help="Dlzka vstupneho okna")
    parser.add_argument("--seed", type=int, required=True, help="Seed pre reprodukovatelnost")
    parser.add_argument("--output", required=True, help="Cesta k vystupnemu textovemu/json suboru")
    args = parser.parse_args()

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

    lookback = args.lookback_window

    if len(series_train) <= lookback + args.horizon:
        print(f"[ERROR] Trénovací set je príliš krátky pre lookback {lookback}.", file=sys.stderr)
        sys.exit(1)

    print(f"[TFT] Trénujem model (Lookback: {lookback}, Horizon: {args.horizon}, Seed: {args.seed})...")

    # Pridanie Early Stoppingu
    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.00,
        mode='min',
    )

    # Inštancia nášho vlastného progress callbacku
    my_progress = PipelineProgressCallback()

    max_epochs = 100

    # Model TFT s minimálnym preprocessingom
    model = TFTModel(
        input_chunk_length=lookback,
        output_chunk_length=args.horizon,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=2,
        dropout=0.1,
        batch_size=1024,
        n_epochs=max_epochs,
        add_relative_index=True,
        random_state=args.seed,
        pl_trainer_kwargs={
            "accelerator": "cuda",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": False,  # Štandardný progress bar necháme vypnutý
            "logger": False,
            "callbacks": [my_stopper, my_progress]  # Pridali sme my_progress
        }
    )

    # Trénujeme (podáme aj val_series pre sledovanie val_loss early stopperom)
    model.fit(series_train, val_series=series_val, verbose=False)

    # Zistenie počtu epoch (kde sa to zastavilo)
    ukoncena_epocha = max_epochs
    if hasattr(model, 'trainer') and model.trainer is not None:
        ukoncena_epocha = model.trainer.current_epoch

    # Ak zafungoval early stopping (zastavilo sa skôr)
    if my_stopper.stopped_epoch > 0:
        ukoncena_epocha = my_stopper.stopped_epoch

    print(f"[TFT] Vytváram predikciu...")
    # Predikujeme z train_val histórie
    prediction = model.predict(n=args.horizon, series=train_val_series)

    # Extrahujeme predikcie do listu pre JSON
    pred_list = prediction.values().flatten().tolist()

    # Zápis predikcií
    print(f"[TFT] Zapisujem výsledky do {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pred_list, f)

    # Zápis metadát (rovnaký priečinok ako args.output)
    output_path = Path(args.output)
    metadata_path = output_path.parent / "tft_metadata.txt"
    print(f"[TFT] Zapisujem metadáta do {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"epoch:{ukoncena_epocha}\n")

    print(f"[TFT] Hotovo!")


if __name__ == "__main__":
    main()