# -*- coding: utf-8 -*-
"""
DeepAR Forecasting Tool

Pouzitie:
    python3 deepar_predict.py --dataset data.csv --horizon 24

Format vstupu:
    - Prvy stlpec: timestamp
    - Posledny stlpec: cielova premenna (target)
    - Stredne stlpce: kovariaty (volitelne)
    - Oddelovac: carka (CSV) alebo tabulator (TXT)

Priklad (jeden stlpec):
    timestamp,hodnota
    2020-01-01,100
    2020-01-02,110

Priklad (s kovariatmi):
    timestamp,teplota,den_tyzdna,spotreba
    2020-01-01,12.3,1,100
    2020-01-02,11.1,2,110
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import Callback
from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx.distribution import GaussianOutput


class EarlyStopping(Callback):
    def __init__(self, patience=10, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.best_network_params = None

    def on_epoch_end(self, epoch_no, epoch_loss, training_network, trainer, best_epoch_info, ctx) -> bool:
        if epoch_loss < self.best_loss - self.min_delta:
            self.best_loss = epoch_loss
            self.best_epoch = epoch_no
            self.epochs_without_improvement = 0
            if self.restore_best:
                self.best_network_params = {
                    k: v.data().copy()
                    for k, v in training_network.collect_params().items()
                }
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            if self.restore_best and self.best_network_params is not None:
                for k, v in training_network.collect_params().items():
                    if k in self.best_network_params:
                        v.set_data(self.best_network_params[k])
            return False
        return True


def load_dataset(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        print(f"CHYBA: Subor '{filepath}' neexistuje.", file=sys.stderr)
        sys.exit(1)

    sep = '\t' if path.suffix.lower() == '.txt' else ','

    try:
        df = pd.read_csv(filepath, sep=sep, parse_dates=[0], index_col=0)
    except Exception as e:
        print(f"CHYBA pri nacitani suboru: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("CHYBA: Subor je prazdny.", file=sys.stderr)
        sys.exit(1)

    return df


def infer_freq(df: pd.DataFrame) -> str:
    try:
        freq = pd.infer_freq(df.index)
        if freq is None:
            print("UPOZORNENIE: Nepodarilo sa odhadnut frekvenciu, pouzivam 'H'.", file=sys.stderr)
            return 'H'
        return freq
    except Exception:
        return 'H'


def main():
    parser = argparse.ArgumentParser(
        description="DeepAR — predikcia casoveho radu z CSV/TXT suboru."
    )
    parser.add_argument("--dataset", required=True,
                        help="Cesta k CSV alebo TXT suboru.")
    parser.add_argument("--horizon", type=int, required=True,
                        help="Dlzka predikcie (pocet krokov dopredu).")
    args = parser.parse_args()

    # 1. Nacitanie dat
    print(f"Nacitavam: {args.dataset}", file=sys.stderr)
    df = load_dataset(args.dataset)

    # Posledny stlpec = target, ostatne = kovariaty
    target_col = df.columns[-1]
    covariate_cols = list(df.columns[:-1])

    print(f"  Target:    '{target_col}'", file=sys.stderr)
    print(f"  Kovariaty: {covariate_cols if covariate_cols else 'ziadne'}", file=sys.stderr)
    print(f"  Dlzka:     {len(df)} bodov", file=sys.stderr)

    if len(df) <= args.horizon:
        print(f"CHYBA: Seria je prilis kratka ({len(df)} bodov) pre horizon {args.horizon}.",
              file=sys.stderr)
        sys.exit(1)

    # 2. Frekvencia
    freq = infer_freq(df)
    print(f"  Frekvencia: {freq}", file=sys.stderr)

    # 3. Context length podla frekvencie (z clanku Tabulka 3)
    freq_base = freq.upper().lstrip('0123456789')
    context_map = {
        'H': 168,   # hodinove — 1 tyzden (electricity, traffic v clanku)
        'D': 30,    # denne — 1 mesiac
        'W': 52,    # tyzdenne — 1 rok (ec v clanku)
        'M': 8,     # mesacne (parts v clanku)
    }
    context_length = context_map.get(freq_base, args.horizon * 2)
    print(f"  Context length: {context_length}", file=sys.stderr)

    # 4. Priprava GluonTS datasetu
    target = df[target_col].values.astype(float)

    entry = {"start": df.index[0], "target": target}

    # Pridaj kovariaty ak existuju
    if covariate_cols:
        feat_dynamic = df[covariate_cols].values.T.astype(float)  # shape: (num_covariates, T)
        entry["feat_dynamic_real"] = feat_dynamic

    gluonts_dataset = ListDataset([entry], freq=freq)

    # 5. Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, restore_best=True)

    # 6. Model — parametre z clanku (Tabulka 3)
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=args.horizon,
        context_length=context_length,
        num_layers=3,                # z clanku
        num_cells=40,                # z clanku
        distr_output=GaussianOutput(),
        scaling=True,
        batch_size=64,               # z clanku
        trainer=Trainer(
            epochs=200,
            learning_rate=1e-3,      # z clanku
            hybridize=True,
            callbacks=[early_stopping],
            add_default_callbacks=True,
        )
    )

    # 7. Trening
    print("Trenujem...", file=sys.stderr)
    predictor = estimator.train(gluonts_dataset)
    print(f"Hotovo (najlepsia epocha: {early_stopping.best_epoch + 1})", file=sys.stderr)

    # 8. Predikcia
    print("Generujem predikcie...", file=sys.stderr)
    forecast_it, _ = make_evaluation_predictions(
        dataset=gluonts_dataset,
        predictor=predictor,
        num_samples=200              # z clanku (Supplementary)
    )
    forecast = list(forecast_it)[0]

    # 9. Vystup — predikcie na stdout
    output = {
        "target": target_col,
        "start_date": str(forecast.start_date),
        "horizon": args.horizon,
        "predictions": {
            "p10": forecast.quantile(0.1).tolist(),
            "p50": forecast.quantile(0.5).tolist(),  # median
            "p90": forecast.quantile(0.9).tolist(),
            "mean": forecast.mean.tolist(),
        }
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()