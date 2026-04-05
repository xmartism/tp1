import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # Mean Absolute Percentage Error (s ochranou proti deleniu nulou)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100

    # Directional Accuracy (DA)
    # Porovnávame, či predikcia odhadla správny smer zmeny (rast/pokles) voči predchádzajúcej skutočnej hodnote
    if len(y_true) > 1:
        actual_diff = y_true[1:] - y_true[:-1]
        pred_diff = y_pred[1:] - y_true[:-1]
        da = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
    else:
        da = np.nan  # Ak máme horizont len 1, smer sa nedá určiť

    return mse, mae, mape, da


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--lookback-window", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    # NOVÉ (nastavené ako nepovinné, aby skript nepadal, kým kolega neupraví pipeline)
    parser.add_argument("--dataset-name", type=str, default="Unknown", help="Názov zdrojového datasetu")
    parser.add_argument("--train-time", type=float, default=0.0, help="Čas trénovania modelu v sekundách")

    args = parser.parse_args()

    # 1. Načítanie predikcií
    try:
        with open(args.output_file, 'r') as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"[ERROR] Nepodarilo sa načítať predikcie z {args.output_file}: {e}")
        sys.exit(1)

    # 2. Načítanie skutočných hodnôt z testovacieho datasetu
    try:
        df_test = pd.read_csv(args.test_dataset)
        actuals = df_test[args.target].iloc[:args.horizon].tolist()
    except Exception as e:
        print(f"[ERROR] Nepodarilo sa načítať testovacie dáta: {e}")
        sys.exit(1)

    # Bezpečnostná kontrola dĺžky
    if len(predictions) != len(actuals):
        print(
            f"[ERROR] Dĺžka predikcií ({len(predictions)}) sa nezhoduje s horizontom ({len(actuals)}) pre {args.model_name}!")
        sys.exit(1)

    # 3. Výpočet metrík
    mse, mae, mape, da = calculate_metrics(actuals, predictions)

    # 4. Uloženie do results.csv
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Vytvoríme DataFrame s novými stĺpcami (Dataset, DA, Train Time)
    df_new = pd.DataFrame([{
        "Dataset": args.dataset_name,
        "Target": args.target,
        "Model": args.model_name,
        "Horizon": args.horizon,
        "Lookback window": args.lookback_window,
        "Seed": args.seed,
        "Train Time (s)": round(args.train_time, 2),
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape, 4),
        "DA (%)": round(da, 2) if not np.isnan(da) else "N/A"
    }])

    # Pripojenie alebo vytvorenie nového súboru
    if results_path.exists():
        df_new.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(results_path, mode='w', header=True, index=False)

    print(f"  [EVAL] {args.model_name} | Dataset: {args.dataset_name} | Target: {args.target}")
    print(
        f"         Time: {args.train_time:.2f}s | MSE: {mse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | DA: {da:.2f}%")


if __name__ == "__main__":
    main()