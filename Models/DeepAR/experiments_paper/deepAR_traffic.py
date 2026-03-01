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
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.best_network_params = None

    def on_epoch_end(self, epoch_no, epoch_loss, training_network, trainer, best_epoch_info, ctx) -> bool:
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
            print(f"  - Bez zlepsenia: {self.epochs_without_improvement}/{self.patience} "
                  f"(loss={current_loss:.4f}, best={self.best_loss:.4f})")

        if self.epochs_without_improvement >= self.patience:
            print(f"\n>>> Early stopping po epoche {epoch_no + 1}. "
                  f"Najlepsia epocha: {self.best_epoch + 1} (loss={self.best_loss:.4f})")
            if self.restore_best and self.best_network_params is not None:
                for k, v in training_network.collect_params().items():
                    if k in self.best_network_params:
                        v.set_data(self.best_network_params[k])
                print("  Obnovene parametre z najlepsej epochy.")
            return False
        return True


def run_deepar_traffic_replication():
    # 1. Nacitanie datasetu
    print("Stahujem dataset 'traffic'...")
    dataset = get_dataset("traffic", regenerate=False)

    print(f"freq: {dataset.metadata.freq}")
    print(f"prediction_length: {dataset.metadata.prediction_length}")
    first_train = next(iter(dataset.train))
    print(f"Dlzka prvej trenovacej serie: {len(first_train['target'])}")
    print(f"Pocet trenovacich serii: {sum(1 for _ in dataset.train)}")

    # 2. Early stopping callback
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4, restore_best=True)

    # 3. Konfiguracia podla clanku (Tabulka 3)
    estimator = DeepAREstimator(
        freq=dataset.metadata.freq,
        prediction_length=24,        # z clanku: decoder length = 24
        context_length=168,          # z clanku: encoder length = 168 (1 tyzden)
        num_layers=3,                # z clanku
        num_cells=40,                # z clanku
        distr_output=GaussianOutput(),   # z clanku: domain [0,1]
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
        num_samples=200
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # 5. Vyhodnotenie
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    # 6. Ulozenie metrics_full.json
    with open("metrics_full_traffic.json", "w", encoding="utf-8") as jf:
        serializable_metrics = {
            k: float(v) if hasattr(v, '__float__') else v
            for k, v in agg_metrics.items()
        }
        json.dump(serializable_metrics, jf, indent=4)

    # 7. Ulozenie results_traffic.txt
    with open("results_traffic.txt", "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("TRENING: TRAFFIC\n")
        f.write("=" * 40 + "\n")
        f.write(f"Najlepsia epocha:   {early_stopping.best_epoch + 1}\n")
        f.write(f"Najlepsia val loss: {early_stopping.best_loss:.4f}\n")
        f.write(f"Cas trenovania:     {minutes}m {seconds}s\n")
        f.write("=" * 40 + "\n")
        f.write("Kompletne metriky su v metrics_full_traffic.json\n")

    # 8. Vypis do konzoly
    print("\n" + "=" * 40)
    print("VYSLEDKY: TRAFFIC")
    print("=" * 40)
    print(f"ND    (v clanku: 0.17): {agg_metrics['ND']:.4f}")
    print(f"NRMSE (v clanku: 0.42): {agg_metrics['NRMSE']:.4f}")
    print(f"RMSE:                   {agg_metrics['RMSE']:.4f}")
    print(f"MASE:                   {agg_metrics['MASE']:.4f}")
    print(f"Najlepsia epocha:       {early_stopping.best_epoch + 1}")
    print(f"Cas trenovania:         {minutes}m {seconds}s")
    print("=" * 40)
    print("Ulozene: results_traffic.txt, metrics_full_traffic.json")

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
    plt.title(f"DeepAR Traffic: ID {plot_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("traffic_result.png")
    print("Graf ulozeny: traffic_result.png")


if __name__ == "__main__":
    run_deepar_traffic_replication()