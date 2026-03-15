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
    # np.where nahradi nuly velmi malym cislom, aby sme predisli chybe Infinity
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100

    return mse, mae, mape

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--results-file", required=True)

    # Pridané argumenty pre načítanie skutočných dát
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--horizon", type=int, required=True)

    args = parser.parse_args()

    # 1. Načítanie predikcií
    # Tvoj výstup vyzerá ako Python/JSON zoznam, takže json.load to bez problémov prečíta
    try:
        with open(args.output_file, 'r') as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"[ERROR] Nepodarilo sa načítať predikcie z {args.output_file}: {e}")
        sys.exit(1)

    # 2. Načítanie skutočných hodnôt z testovacieho datasetu
    try:
        df_test = pd.read_csv(args.test_dataset)
        # Zoberieme len prvých X riadkov (podľa horizontu) z cieľového stĺpca
        actuals = df_test[args.target].iloc[:args.horizon].tolist()
    except Exception as e:
        print(f"[ERROR] Nepodarilo sa načítať testovacie dáta: {e}")
        sys.exit(1)

    # Bezpečnostná kontrola dĺžky
    if len(predictions) != len(actuals):
        print(f"[ERROR] Dĺžka predikcií ({len(predictions)}) sa nezhoduje s horizontom ({len(actuals)}) pre {args.model_name}!")
        sys.exit(1)

    # 3. Výpočet metrík
    mse, mae, mape = calculate_metrics(actuals, predictions)

    # 4. Uloženie do results.csv
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True) # Pre istotu vytvorí zložku

    # Vytvoríme DataFrame s jedným riadkom
    df_new = pd.DataFrame([{
        "Model": args.model_name,
        "Horizon": args.horizon,
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape, 4)
    }])

    # Ak už súbor existuje, pripojíme nový riadok, inak vytvoríme nový súbor s hlavičkou
    if results_path.exists():
        df_new.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(results_path, mode='w', header=True, index=False)

    print(f"  [EVAL] {args.model_name} -> MSE: {mse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()