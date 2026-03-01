# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json
import numpy as np
import time
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import Callback
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.mx.distribution import GaussianOutput


class EarlyStopping(Callback):
    """
    Early stopping callback pre GluonTS Trainer.
    Sleduje val. loss a zastavi trening ak sa nezlepsi 'patience' epoch za sebou.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.best_network_params = None

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network,
        trainer,
        best_epoch_info,
        ctx,
    ) -> bool:
        """Vracia False = zastavi trening, True = pokracuje."""
        current_loss = epoch_loss

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = epoch_no
            self.epochs_without_improvement = 0
            if self.restore_best:
                self.best_network_params = {
                    k: v.data().copy()
                    for k, v in training_network.collect_params().items()
                }
            print(f"  + Zlepsenie: loss={current_loss:.4f} (epocha {epoch_no + 1})")
        else:
            self.epochs_without_improvement += 1
            print(
                f"  - Bez zlepsenia: {self.epochs_without_improvement}/{self.patience} "
                f"(loss={current_loss:.4f}, best={self.best_loss:.4f})"
            )

        if self.epochs_without_improvement >= self.patience:
            print(
                f"\n>>> Early stopping po epoche {epoch_no + 1}. "
                f"Najlepsia epocha: {self.best_epoch + 1} (loss={self.best_loss:.4f})"
            )
            if self.restore_best and self.best_network_params is not None:
                for k, v in training_network.collect_params().items():
                    if k in self.best_network_params:
                        v.set_data(self.best_network_params[k])
                print("  Obnovene parametre z najlepsej epochy.")
            return False

        return True


def run_deepar_electricity_replication():
    # 1. Nacitanie datasetu
    print("Stahujem dataset 'electricity'...")
    dataset = get_dataset("electricity", regenerate=False)

    # 2. Early stopping callback
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=1e-4,
        restore_best=True
    )

    # 3. Konfiguracia podla clanku (Tabulka 3)
    estimator = DeepAREstimator(
        freq=dataset.metadata.freq,
        prediction_length=24,        # z clanku
        context_length=168,          # z clanku
        num_layers=3,                # z clanku
        num_cells=40,                # z clanku
        distr_output=GaussianOutput(),
        scaling=True,
        batch_size=64,               # z clanku
        trainer=Trainer(
            epochs=200,              # horny strop, early stopping zastavi skor
            learning_rate=1e-3,      # z clanku
            hybridize=True,
            callbacks=[early_stopping],
            add_default_callbacks=True,
        )
    )

    start_time = time.time()

    print("Trenujem DeepAR s early stopping...")
    predictor = estimator.train(dataset.train)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Cas trenovania: {minutes}m {seconds}s")

    # 4. Predikcia
    print("\nGenerujem predikcie (200 samples podla clanku)...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=200              # z clanku (Supplementary)
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    # 5. Vyhodnotenie — vsetky metriky pocita GluonTS Evaluator
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    # 6. Ulozenie metrics_full.json — kompletne metriky z Evaluatora
    with open("metrics_full.json", "w", encoding="utf-8") as jf:
        serializable_metrics = {
            k: float(v) if hasattr(v, '__float__') else v
            for k, v in agg_metrics.items()
        }
        json.dump(serializable_metrics, jf, indent=4)

    # 7. Ulozenie results.txt — len info ktore NIE SU v metrics_full.json
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("TRENING: ELECTRICITY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Najlepsia epocha:       {early_stopping.best_epoch + 1}\n")
        f.write(f"Najlepsia val loss:     {early_stopping.best_loss:.4f}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Cas trenovania:         {minutes}m {seconds}s\n")

    # 8. Vypis do konzoly
    print("\n" + "=" * 40)
    print("VYSLEDKY: ELECTRICITY")
    print("=" * 40)
    print(f"ND    (v clanku: 0.07):  {agg_metrics['ND']:.4f}")
    print(f"NRMSE (v clanku: ~1.00): {agg_metrics['NRMSE']:.4f}")
    print(f"RMSE:                    {agg_metrics['RMSE']:.4f}")
    print(f"MASE:                    {agg_metrics['MASE']:.4f}")
    print(f"Najlepsia epocha:        {early_stopping.best_epoch + 1}")
    print("=" * 40)
    print("Ulozene: results.txt, metrics_full.json")

    # 9. Graf
    plot_idx = 10
    plt.figure(figsize=(12, 6))
    target_series = tss[plot_idx][-100:]
    plt.plot(
        target_series.index.to_timestamp(),
        target_series.values,
        label="Actual",
        color="black"
    )
    forecasts[plot_idx].plot(intervals=(0.5, 0.9), color='g')
    plt.title(f"DeepAR Electricity: ID {plot_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("electricity_result.png")
    print("Graf ulozeny: electricity_result.png")


if __name__ == "__main__":
    run_deepar_electricity_replication()