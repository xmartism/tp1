"""
Real Evaluation Script

Číta výstupy z modelov (MAPE, MAE, MSE) a ukladá ich do spoločného CSV.
"""

import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name of the model being evaluated")
    parser.add_argument("--output-file", required=True, help="Path to the model output file")
    parser.add_argument("--results-file", required=True, help="Path to the results CSV file")
    args = parser.parse_args()

    # Predvolené hodnoty, ak by sa v texte nenašli
    mape_val = "N/A"
    mae_val = "N/A"
    mse_val = "N/A"

    # 1. Čítanie textového výstupu z modelu
    output_path = Path(args.output_file)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("MAPE:"):
                    mape_val = line.split(":", 1)[1].strip()
                elif line.startswith("MAE:"):
                    mae_val = line.split(":", 1)[1].strip()
                elif line.startswith("MSE:"):
                    mse_val = line.split(":", 1)[1].strip()
    else:
        print(f"[WARN] Súbor {args.output_file} neexistuje. Metriky nebudú načítané.")

    print(f"[{args.model_name.upper()}] Evaluácia: MAPE={mape_val}, MAE={mae_val}, MSE={mse_val}")

    # 2. Zápis do spoločného CSV súboru
    results_path = Path(args.results_file)

    # Skontrolujeme, či súbor existuje a či nie je prázdny
    file_exists = results_path.exists() and results_path.stat().st_size > 0

    with open(results_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Ak súbor neexistoval, zapíšeme najskôr hlavičku (názvy stĺpcov)
        if not file_exists:
            writer.writerow(["model_name", "MAPE", "MAE", "MSE", "output_file"])

        # Zapíšeme konkrétne výsledky modelu
        writer.writerow([args.model_name, mape_val, mae_val, mse_val, args.output_file])

    print(f"Výsledky úspešne zapísané do {args.results_file}")

if __name__ == "__main__":
    main()