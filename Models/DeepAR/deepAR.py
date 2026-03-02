"""
DeepAR Forecasting Tool (GluonTS/MXNet)

Pouzitie:
    python3 deepAR.py --dataset data.csv --horizon 24 --target spotreba
    python3 deepAR.py --dataset electricity --horizon 24
    python3 deepAR.py --dataset electricity --horizon 24 --metrics --compare compare.csv

Format vstupu (CSV/TXT):
    - Stlpec s datumom: --date (default: 'date')
    - Cielova premenna: --target
    - Pocet krokov predikcie: --horizon
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import Callback
from gluonts.mx.distribution import GaussianOutput
from gluonts.evaluation import make_evaluation_predictions, Evaluator


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping(Callback):
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = np.inf
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
            print(f"  + Zlepsenie: loss={epoch_loss:.4f} (epocha {epoch_no + 1})", file=sys.stderr)
        else:
            self.epochs_without_improvement += 1
            print(
                f"  - Bez zlepsenia: {self.epochs_without_improvement}/{self.patience} "
                f"(loss={epoch_loss:.4f}, best={self.best_loss:.4f})",
                file=sys.stderr
            )

        if self.epochs_without_improvement >= self.patience:
            print(
                f"\n>>> Early stopping po epoche {epoch_no + 1}. "
                f"Najlepsia epocha: {self.best_epoch + 1} (loss={self.best_loss:.4f})",
                file=sys.stderr
            )
            if self.restore_best and self.best_network_params is not None:
                for k, v in training_network.collect_params().items():
                    if k in self.best_network_params:
                        v.set_data(self.best_network_params[k])
            return False

        return True


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

BUILTIN_DATASETS = {"electricity", "traffic", "m4_hourly", "m4_daily", "m4_weekly", "m4_monthly"}

CONTEXT_MAP = {
    'H': 168,
    'D': 30,
    'W': 52,
    'M': 8,
}


def load_csv_dataset(filepath: str, date_col: str, target_col: str):
    path = Path(filepath)
    if not path.exists():
        print(f"CHYBA: Subor '{filepath}' neexistuje.", file=sys.stderr)
        sys.exit(1)

    sep = '\t' if path.suffix.lower() == '.txt' else ','

    try:
        df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        print(f"CHYBA pri nacitani suboru: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("CHYBA: Subor je prazdny.", file=sys.stderr)
        sys.exit(1)

    if date_col not in df.columns:
        print(f"CHYBA: Stlpec '{date_col}' neexistuje. Dostupne: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if target_col not in df.columns:
        print(f"CHYBA: Stlpec '{target_col}' neexistuje. Dostupne: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    except Exception as e:
        print(f"CHYBA pri parsovani datumov: {e}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepAR — universal forecasting tool.")

    # Obligatory
    parser.add_argument("--dataset", required=True,
                        help="Cesta k CSV/TXT suboru alebo nazov builtin datasetu (electricity, traffic, ...).")
    parser.add_argument("--horizon", type=int, required=True,
                        help="Dlzka predikcie (pocet krokov dopredu).")

    # CSV-specific
    parser.add_argument("--target", default=None,
                        help="Nazov stlpca s cielovou premennou (vyzadovane pre CSV/TXT).")
    parser.add_argument("--date", default="date",
                        help="Nazov stlpca s datumom (default: 'date').")

    # Optional training params
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="Zariadenie (default: cpu).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max pocet epoch (default: 200).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64).")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Pocet LSTM vrstiev (default: 3).")
    parser.add_argument("--num-cells", type=int, default=40,
                        help="Pocet buniek v LSTM vrstve (default: 40).")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Context length (default: automaticky podla frekvencie).")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Pocet vzoriek pri predikcii (default: 200).")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10).")

    # Optional output arguments
    parser.add_argument("--metrics", action="store_true",
                        help="Ak je zadane, vypise metriky (ND, NRMSE, RMSE, MASE).")
    parser.add_argument("--compare", default=None,
                        help="Cesta k vystupnemu CSV s dvoma stlpcami: actual, predicted.")
    parser.add_argument("--log-file", default=None,
                        help="Cesta k log suboru. Ak nie je zadane, vypis ide na stderr.")

    args = parser.parse_args()

    # Redirect stderr to log file if specified
    if args.log_file is not None:
        log_handle = open(args.log_file, "w", encoding="utf-8")
        sys.stderr = log_handle
    else:
        log_handle = None

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    is_builtin = args.dataset.lower() in BUILTIN_DATASETS

    if is_builtin:
        print(f"Nacitavam builtin dataset '{args.dataset}'...", file=sys.stderr)
        gluonts_data = get_dataset(args.dataset.lower(), regenerate=False)
        train_dataset = gluonts_data.train
        test_dataset = gluonts_data.test
        freq = gluonts_data.metadata.freq
        print(f"  Frekvencia: {freq}", file=sys.stderr)
    else:
        if args.target is None:
            print("CHYBA: --target je vyzadovane pre CSV/TXT subory.", file=sys.stderr)
            sys.exit(1)

        print(f"Nacitavam: {args.dataset}", file=sys.stderr)
        df = load_csv_dataset(args.dataset, date_col=args.date, target_col=args.target)

        covariate_cols = [
            c for c in df.columns
            if c != args.target and pd.api.types.is_numeric_dtype(df[c])
        ]

        print(f"  Target:    '{args.target}'", file=sys.stderr)
        print(f"  Kovariaty: {covariate_cols if covariate_cols else 'ziadne'}", file=sys.stderr)
        print(f"  Dlzka:     {len(df)} bodov", file=sys.stderr)

        if len(df) <= args.horizon * 2:
            print(f"CHYBA: Seria je prilis kratka ({len(df)}) pre horizon {args.horizon}.", file=sys.stderr)
            sys.exit(1)

        freq = infer_freq(df)
        print(f"  Frekvencia: {freq}", file=sys.stderr)

        full_target = df[args.target].values.astype(float)

        # Split: train = all except last horizon, test = full series
        train_target = full_target[:-args.horizon]

        print(f"  Trenovacich bodov: {len(train_target)}", file=sys.stderr)
        print(f"  Testovacich bodov (horizon): {args.horizon}", file=sys.stderr)

        train_entry = {"start": df.index[0], "target": train_target}
        test_entry  = {"start": df.index[0], "target": full_target}

        if covariate_cols:
            train_entry["feat_dynamic_real"] = df[covariate_cols].values[:-args.horizon].T.astype(float)
            test_entry["feat_dynamic_real"]  = df[covariate_cols].values.T.astype(float)

        train_dataset = ListDataset([train_entry], freq=freq)
        test_dataset  = ListDataset([test_entry],  freq=freq)

    # ------------------------------------------------------------------
    # 2. Context length
    # ------------------------------------------------------------------
    if args.context_length is not None:
        context_length = args.context_length
    else:
        freq_base = freq.upper().lstrip('0123456789')
        context_length = CONTEXT_MAP.get(freq_base, args.horizon * 2)

    print(f"  Context length: {context_length}", file=sys.stderr)
    print(f"  Zariadenie: {args.device.upper()}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 3. MXNet context
    # ------------------------------------------------------------------
    import mxnet as mx
    if args.device == 'gpu':
        try:
            test = mx.nd.array([1], ctx=mx.gpu(0))
            _ = test + 1
            ctx = mx.gpu(0)
            print("  GPU dostupne, pouzivam GPU.", file=sys.stderr)
        except Exception:
            print("  UPOZORNENIE: GPU nie je dostupne, pouzivam CPU.", file=sys.stderr)
            ctx = mx.cpu()
    else:
        ctx = mx.cpu()

    # ------------------------------------------------------------------
    # 4. Early stopping + model
    # ------------------------------------------------------------------
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, restore_best=True)

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=args.horizon,
        context_length=context_length,
        num_layers=args.num_layers,
        num_cells=args.num_cells,
        distr_output=GaussianOutput(),
        scaling=True,
        batch_size=args.batch_size,
        trainer=Trainer(
            epochs=args.epochs,
            learning_rate=args.lr,
            hybridize=True,
            ctx=ctx,
            callbacks=[early_stopping],
            add_default_callbacks=True,
        )
    )

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print("Trenujem...", file=sys.stderr)
    predictor = estimator.train(train_dataset)
    print(f"Hotovo. Najlepsia epocha: {early_stopping.best_epoch + 1}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 6. Prediction
    # ------------------------------------------------------------------
    print("Generujem predikcie...", file=sys.stderr)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=args.num_samples,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # ------------------------------------------------------------------
    # 7. Metrics (optional)
    # ------------------------------------------------------------------
    if args.metrics:
        print("\nPocitam metriky...", file=sys.stderr)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, _ = evaluator(tss, forecasts)
        print("\n" + "=" * 40, file=sys.stderr)
        print("METRIKY", file=sys.stderr)
        print("=" * 40, file=sys.stderr)
        print(f"ND:    {agg_metrics['ND']:.4f}", file=sys.stderr)
        print(f"NRMSE: {agg_metrics['NRMSE']:.4f}", file=sys.stderr)
        print(f"RMSE:  {agg_metrics['RMSE']:.4f}", file=sys.stderr)
        print(f"MASE:  {agg_metrics['MASE']:.4f}", file=sys.stderr)
        print("=" * 40, file=sys.stderr)

    # ------------------------------------------------------------------
    # 8. Compare CSV (optional)
    # ------------------------------------------------------------------
    if args.compare is not None:
        rows = []
        for ts, forecast in zip(tss, forecasts):
            actual_vals = ts[-args.horizon:].values.tolist()
            predicted_vals = forecast.quantile(0.5).tolist()
            for a, p in zip(actual_vals, predicted_vals):
                rows.append({"actual": a, "predicted": p})

        compare_df = pd.DataFrame(rows, columns=["actual", "predicted"])
        compare_df.to_csv(args.compare, index=False)
        print(f"Porovnanie ulozene: {args.compare}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 9. Main output — predictions only to stdout
    # ------------------------------------------------------------------
    if is_builtin:
        output = [f.quantile(0.5).tolist() for f in forecasts]
    else:
        output = forecasts[0].quantile(0.5).tolist()

    print(json.dumps(output))

    # Close log file if opened
    if log_handle is not None:
        sys.stderr = sys.__stderr__
        log_handle.close()


if __name__ == "__main__":
    main()