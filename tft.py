import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer

import matplotlib.pyplot as plt

# === 1. Načítanie CSV ===
csv_path = "data.csv"  # <- uprav podľa potreby
df = pd.read_csv(csv_path)

# pridáme id série (ak je len jedna séria)
df["series_id"] = 0

# zabezpečíme, že time_idx je int
df["time_idx"] = df["time"].astype(int)


# === 2. Definícia parametrov ===
max_encoder_length = 30   # koľko krokov spätne
max_prediction_length = 7 # koľko krokov dopredu predpovedáme

# Rozdelenie dát na trénovaciu a validačnú množinu (80% / 20%)
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
val_df = df[train_size:]

# === 3. Vytvorenie TimeSeriesDataSet ===
train_dataset = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="value",
    group_ids=["series_id"],
    time_varying_unknown_reals=["value"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)

val_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    val_df,
    stop_randomization=True
)

# Dataloader-y
batch_size = 16
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)
#"""
# === 4. Tréning TFT modelu ===
trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    enable_model_summary=True
)

# model je LightningModule
tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # pre quantile loss
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
)

#trainer.fit(tft, train_dataloader, val_dataloader)
tft = TemporalFusionTransformer.load_from_checkpoint("tft_model.ckpt")
# === 5. Vyhodnotenie a vizualizácia ===

# Predikcie na validačných dátach
raw_predictions, x,*_ = tft.predict(
    val_dataloader,
    mode="raw",
    return_x=True
)

# === 5a. Plot predikcií (ako v Stallion tutoriáli) ===
print("Vizualizácia predikcií pre prvé 3 príklady:")

for idx in range(3):
    fig, ax = plt.subplots(figsize=(10, 4))
    tft.plot_prediction(
        x,
        raw_predictions,
        idx=idx,
        add_loss_to_title=True,
        ax=ax
    )
    plt.show()

# === 5b. (Voliteľné) – plot zjednodušených predikcií ===
# Priemerná predikcia (napr. mediánový kvantil)
predictions = tft.predict(val_dataloader)
print("\nUkážka numerických predikcií:")
print(predictions[:10])
