import subprocess
import sys
import os
import pandas as pd
from pathlib import Path

# Zoznam experimentov presne podľa tvojej požiadavky
experiments = [
    {
        "name": "weather_history",
        "dataset": "data/weatherHistory.csv",
        "target": "Temperature (C)",
        "date_col": "Formatted Date",
        "type": "date",
        "original_format": None,  # Pandas si poradí s ISO formátom sám
        "horizon": 24
    },
    {
        "name": "sinusoida",
        "dataset": "data/sinus_1000_10waves.csv",
        "target": "value",
        "date_col": "time",
        "type": "int",
        "horizon": 50,
        "original_format": None
    },
    {
        "name": "jena_pocasie",
        "dataset": "data/jena_climate.csv",
        "target": "T (degC)",
        "date_col": "Date Time",
        "type": "date",
        "original_format": "%d.%m.%Y %H:%M:%S",  # Napr. 01.01.2009 00:10:00
        "horizon": 24
    },

]


def preprocess_dataset(exp):
    """Zmení formát dátumu na: 2000-01-01 00:00:00.000 +0200"""
    path = Path(exp["dataset"])
    print(f"  [Preprocess] Upravujem dátumy pre: {exp['name']}...")

    df = pd.read_csv(path)
    col = exp["date_col"]

    if exp["type"] == "int":
        # Ak je to číslo, začíname od roku 2000
        start_dt = pd.Timestamp("2000-01-01 00:00:00")
        df[col] = start_dt + pd.to_timedelta(df[col], unit='h')
    else:
        # Ak je to dátum, zachováme pôvodný čas, len zmeníme formát zápisu
        df[col] = pd.to_datetime(df[col], format=exp["original_format"], utc=True)

    # Prevod na cieľový formát: YYYY-MM-DD HH:MM:SS.000 +0200
    # .strftime vyrobí string, ku ktorému fixne pridáme +0200 (ak nie je definované inak)
    df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S.000") + " +0200"

    # Uložíme ako dočasný súbor pre pipeline
    processed_path = path.parent / f"formatted_{path.name}"
    df.to_csv(processed_path, index=False)

    return str(processed_path)


def main():
    python_exe = sys.executable
    os.makedirs("Pipeline/outputs", exist_ok=True)

    for i, exp in enumerate(experiments, 1):
        if not os.path.exists(exp["dataset"]):
            print(f"\n[!] Súbor {exp['dataset']} neexistuje. Preskakujem.")
            continue

        print(f"\n" + "=" * 90)
        print(f" [{i}/{len(experiments)}] SPUŠŤAM PIPELINE PRE: {exp['name'].upper()}")

        # 1. Krok: Preformátovanie dát pred volaním pipeline
        new_path = preprocess_dataset(exp)

        # 2. Krok: Spustenie samotnej pipeline (príkaz podobný tvojmu vzoru)
        cmd = [
            python_exe, "Pipeline/pipeline.py",
            "--dataset", new_path,
            "--target", exp["target"],
            "--date", exp["date_col"],
            "--horizon", str(exp["horizon"]),
            "--results-file", f"Pipeline/outputs/results_{exp['name']}.csv"
        ]

        print(f"  CMD: {' '.join(cmd)}")

        try:
            # Sekvenčné spustenie (jedno po druhom)
            subprocess.run(cmd, check=True, env=os.environ)
            print(f"\n[OK] Hotovo: {exp['name']}")
        except subprocess.CalledProcessError:
            print(f"\n[!] Chyba pri modeli v experimente {exp['name']}")

    print("\n" + "=" * 90)
    print(" Všetky tri experimenty boli úspešne spracované.")


if __name__ == "__main__":
    main()