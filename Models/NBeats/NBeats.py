import argparse
import sys
import torch
import pandas as pd
import numpy as np

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, mse
from darts.utils.missing_values import fill_missing_values

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
    args = parser.parse_args()


    print(f"[N-BEATS] Načítavam dáta...")
    try:
        df_train = pd.read_csv(args.train_dataset)
        df_val = pd.read_csv(args.val_dataset)
        df_test = pd.read_csv(args.test_dataset)
    except Exception as e:
        print(f"[ERROR] Zlyhalo načítanie CSV súborov: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        series_train = TimeSeries.from_dataframe(df_train, time_col=args.date, value_cols=args.target,
                                                 fill_missing_dates=True, freq='D')
        series_val = TimeSeries.from_dataframe(df_val, time_col=args.date, value_cols=args.target,
                                               fill_missing_dates=True, freq='D')
        series_test = TimeSeries.from_dataframe(df_test, time_col=args.date, value_cols=args.target,
                                                fill_missing_dates=True, freq='D')

        series_train = fill_missing_values(series_train, fill='auto')
        series_val = fill_missing_values(series_val, fill='auto')
        series_test = fill_missing_values(series_test, fill='auto')
    except Exception as e:
        print(f"[ERROR] Problém s vytváraním TimeSeries (skontroluj názvy stĺpcov): {e}", file=sys.stderr)
        sys.exit(1)

    scaler = Scaler()
    train_scaled = scaler.fit_transform(series_train)
    val_scaled = scaler.transform(series_val)
    lookback = args.horizon * 3

    if len(series_train) <= lookback + args.horizon:
        print(f"[ERROR] Trénovací set je príliš krátky pre lookback {lookback} a horizon {args.horizon}.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[N-BEATS] Trénujem model (Lookback: {lookback}, Horizon: {args.horizon})...")
    model = NBEATSModel(
        input_chunk_length=lookback,
        output_chunk_length=args.horizon,
        generic_architecture=True,
        num_stacks=30,
        layer_widths=512,
        n_epochs=10,
        batch_size=1024,
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "auto",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": False,
            "logger": False
        }
    )

    model.fit(train_scaled, verbose=False)


    print(f"[N-BEATS] Vytváram predikciu...")
    pred_scaled = model.predict(n=args.horizon, series=train_scaled)
    prediction = scaler.inverse_transform(pred_scaled)
    actual = series_val[:args.horizon]
    try:
        chyba_mape = mape(actual, prediction)
        chyba_mae = mae(actual, prediction)
        chyba_mse = mse(actual, prediction)
    except Exception as e:
        print(f"[WARN] Nepodarilo sa vypočítať metriky: {e}")
        chyba_mape, chyba_mae, chyba_mse = -1, -1, -1

    print(f"[N-BEATS] Zapisujem výsledky do {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Model: N-BEATS\n")
        f.write(f"MAPE: {chyba_mape}\n")
        f.write(f"MAE: {chyba_mae}\n")
        f.write(f"MSE: {chyba_mse}\n")

        f.write("\nPredikcie:\n")
        # Zápis hodnôt
        pred_df = prediction.to_dataframe()
        for idx, row in pred_df.iterrows():
            # Zaokrúhlenie na 4 desatinné miesta
            val = float(row.iloc[0])
            f.write(f"{idx},{val:.4f}\n")

    print(f"[N-BEATS] Hotovo!")


if __name__ == "__main__":
    main()