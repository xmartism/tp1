# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.mx.distribution import GaussianOutput # Zmena na Gaussovo rozdelenie

def run_deepar_electricity_replication():
    # 1. Nacitanie datasetu Electricity
    print("Stahujem dataset 'electricity'...")
    dataset = get_dataset("electricity", regenerate=False)
    
    # 2. Konfiguracia presne podla clanku (Sekcia 4.2)
    estimator = DeepAREstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        context_length=168,           # 1 týžden histórie (klúcové pre Electricity)
        num_layers=3,                 # Podla clánku
        num_cells=40,                 # Podla clánku
        distr_output=GaussianOutput(), 
        scaling=True,
        trainer=Trainer(
            epochs=150,                # Dostatok casu na hlbokú konvergenciu
            learning_rate=1e-3,
            num_batches_per_epoch=500, # Viac vzoriek v každom kroku
            hybridize=True             # Zrýchlenie na MXNet backende
        )
    )

    print("Trenujem DeepAR na elektrine... Sledujte pokles Loss.")
    predictor = estimator.train(dataset.train)

    # 3. Predikcia
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, 
        predictor=predictor, 
        num_samples=100
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    # 4. Vyhodnotenie
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

    agg_metrics, item_metrics = evaluator(tss, forecasts)
    nd = agg_metrics['abs_error'] / agg_metrics['abs_target_sum']

    # 1. Uloženie do textového súboru pre rýchlu kontrolu
    with open("results.txt", "w") as f:
        f.write("="*40 + "\n")
        f.write("VYSLEDKY REPLIKACIE: ELECTRICITY\n")
        f.write("="*40 + "\n")
        f.write(f"ND (Normalized Deviation): {nd:.4f}\n")
        f.write(f"RMSE:                     {agg_metrics['RMSE']:.4f}\n")
        f.write(f"Mean wQuantileLoss:       {agg_metrics['mean_wQuantileLoss']:.4f}\n")
        f.write(f"MASE:                     {agg_metrics['MASE']:.4f}\n")
        f.write("="*40 + "\n")

    # 2. Uloženie kompletných metrík do JSON (vhodné pre dalšie spracovanie)
    import json
    with open("metrics_full.json", "w") as jf:
        # Prekonvertujeme numpy/float hodnoty na cisté floaty pre JSON
        serializable_metrics = {k: float(v) if hasattr(v, '__float__') else v 
                               for k, v in agg_metrics.items()}
        json.dump(serializable_metrics, jf, indent=4)

    print("\nMetriky boli uspesne ulozene do 'results.txt' a 'metrics_full.json'")
    print("\n" + "="*40)
    print("VYSLEDKY: ELECTRICITY")
    print("="*40)
    print(f"ND (v clanku 0.07):  {nd:.4f}")
    print(f"RMSE (v clanku 0.39): {agg_metrics['RMSE']:.4f}")
    print("="*40)

    # 5. Graf
    plot_idx = 10 
    plt.figure(figsize=(12, 6))
    target_series = tss[plot_idx][-100:] # Poslednych 100 hodin
    plt.plot(target_series.index.to_timestamp(), target_series.values, label="Actual", color="black")
    forecasts[plot_idx].plot(intervals=(0.5, 0.9), color='g')
    plt.title(f"DeepAR Electricity: ID {plot_idx}")
    plt.savefig("electricity_result.png")
    print("Graf ulozeny: electricity_result.png")

if __name__ == "__main__":
    run_deepar_electricity_replication()