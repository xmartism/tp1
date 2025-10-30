import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
import matplotlib.pyplot as plt

# === 1. Načítanie CSV ===
csv_path = "apple1d.csv"  # uprav podľa potreby
df = pd.read_csv(csv_path)
print(df.columns)
# === 2. Úprava dát ===
# Dátum vo formáte mesiac/deň/rok (napr. 1/2/1990)

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Zoradíme podľa dátumu
df = df.sort_values("Date").reset_index(drop=True)

# Vytvoríme číselný index pre čas (potrebný pre TFT)
df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days

# Pridáme identifikátor série (ak je len jedna)
df["series_id"] = 0

# Cieľová premenná – budeme predpovedať Close
df["target"] = df["Close"]

# === 3. Definícia parametrov ===
max_encoder_length = 60   # koľko dní spätne berieme ako vstup
max_prediction_length = 30 # koľko dní dopredu predpovedáme

# Rozdelenie dát (80 % tréning, 20 % validácia)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# === 4. Vytvorenie TimeSeriesDataSet ===
train_dataset = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="target",
    group_ids=["series_id"],
    time_varying_known_reals=["time_idx"],  # čas ako známa veličina
    time_varying_unknown_reals=["Open", "High", "Low", "Close", "Volume", "target"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    allow_missing_timesteps=True,
)

val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, stop_randomization=True)

# === 5. DataLoadery ===
batch_size = 16
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)

# === 6. Tréning TFT modelu ===
trainer = Trainer(
    max_epochs=30,
    accelerator="gpu",
    enable_model_summary=True
)

tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=8,
    dropout=0.2,
    hidden_continuous_size=16,
    output_size=7,  # pre quantile loss
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
)

trainer.fit(tft, train_dataloader, val_dataloader)

# === 7. Predikcie a vizualizácia ===
raw_predictions, x, *_ = tft.predict(
    val_dataloader,
    mode="raw",
    return_x=True
)

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

# === 8. Numerické predikcie (napr. mediánový kvantil) ===
predictions = tft.predict(val_dataloader)
print("\nUkážka numerických predikcií:")
print(predictions[:10])
