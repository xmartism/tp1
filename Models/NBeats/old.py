import torch
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, mse
import requests
import io
import zipfile

torch.set_float32_matmul_precision('medium')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import smape
from darts.dataprocessing.transformers import Scaler


print("Sťahujem dataset M4 (Mesačné dáta - Train časť)...")

url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv"
df_m4 = pd.read_csv(url)

H = 18
LOOKBACK = 54
N_SERIES = 1

smapes = []

print(f"Spúšťam test. Hľadám prvých {N_SERIES} dostatočne dlhých radov...\n")

valid_series_count = 0
row_index = 0

while valid_series_count < N_SERIES and row_index < len(df_m4):

    series_name = df_m4.iloc[row_index, 0]
    values = df_m4.iloc[row_index, 1:].dropna().values.astype(float)
    series = TimeSeries.from_values(values)

    if len(series) < LOOKBACK + 2 * H:
        row_index += 1
        continue

    train, val = series[:-H], series[-H:]

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)

    model = NBEATSModel(
        input_chunk_length=LOOKBACK,
        output_chunk_length=H,
        generic_architecture=True,
        num_stacks=30,
        layer_widths=512,
        n_epochs=15,
        batch_size=1024,
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "auto",
            "enable_progress_bar": False,
            "logger": False
        }
    )

    model.fit(train_scaled, verbose=False)

    pred_scaled = model.predict(n=H)
    pred = scaler.inverse_transform(pred_scaled)

    error = smape(val, pred)
    smapes.append(error)

    print(f"Rad {series_name} (Dĺžka: {len(series)}) | sMAPE = {error:.2f}%")

    valid_series_count += 1
    row_index += 1

final_smape = np.mean(smapes)

print("\n" + "=" * 50)
print(f"VÁŠ PRIEMERNÝ sMAPE (vzorka {len(smapes)} radov): {final_smape:.3f}%")
print("CIEĽ Z ČLÁNKU (priemer za všetky mesačné rady): 12.048%")
print("=" * 50 + "\n")

plt.figure(figsize=(12, 6))
train[-100:].plot(label='Tréningové dáta (História)')
val.plot(label='Skutočnosť', color='black', linestyle='--')
pred.plot(label='N-BEATS Predikcia', color='red', lw=2)

plt.title(f'N-BEATS Predikcia pre rad {series_name} (sMAPE: {error:.2f}%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

df_sin = pd.read_csv('sinus_1000_10waves.csv')
series_sin = TimeSeries.from_dataframe(df_sin, value_cols='value')
train_sin, val_sin = series_sin[:-200], series_sin[-200:]

scaler_sin = Scaler()
train_sin_scaled = scaler_sin.fit_transform(train_sin)
val_sin_scaled = scaler_sin.transform(val_sin)

model_sin = NBEATSModel(
    input_chunk_length=50,
    output_chunk_length=10,
    n_epochs=50,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "auto"}
)
model_sin.fit(train_sin_scaled, verbose=True)

pred_sin_scaled = model_sin.predict(n=200)
pred_sin = scaler_sin.inverse_transform(pred_sin_scaled)

print(f"Sinus MAPE: {mape(val_sin, pred_sin):.2f}%")
print(f"Sinus MSE: {mse(val_sin, pred_sin):.2f}")
print(f"Sinus MAE: {mae(val_sin, pred_sin):.2f}")

plt.figure(figsize=(12, 6))
train_sin[-100:].plot(label='Tréning')
val_sin.plot(label='Skutočnosť')
pred_sin.plot(label='N-BEATS Predikcia', lw=2)
plt.title('Experiment 1: Sin')
plt.legend()
plt.show()


url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open('jena_climate_2009_2016.csv'))


df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

df = df.set_index('Date Time')
df = df[~df.index.duplicated(keep='first')]

print("Prepočítavam dáta na presné hodinové priemery...")
df_hourly = df.resample('1h').mean()

df_hourly = df_hourly.interpolate(method='linear')
df_hourly = df_hourly.reset_index()

print("Vytváram TimeSeries (dataset je teraz čistý)...")

series_target = TimeSeries.from_dataframe(
    df_hourly,
    time_col='Date Time',
    value_cols='T (degC)',
    freq='h'
)
series_covariates = TimeSeries.from_dataframe(
    df_hourly,
    time_col='Date Time',
    value_cols=['p (mbar)', 'rho (g/m**3)', 'Tdew (degC)'],
    freq='h'
)

val_len = 2000
train_target, val_target = series_target[:-val_len], series_target[-val_len:]
train_cov, val_cov = series_covariates[:-val_len], series_covariates[-val_len:]

scaler_target = Scaler()
scaler_cov = Scaler()

train_target_scaled = scaler_target.fit_transform(train_target)
val_target_scaled = scaler_target.transform(val_target)

train_cov_scaled = scaler_cov.fit_transform(train_cov)
val_cov_scaled = scaler_cov.transform(val_cov)

model = NBEATSModel(
    input_chunk_length=168,
    output_chunk_length=24,
    generic_architecture=True,
    num_stacks=30,
    layer_widths=512,
    n_epochs=20,
    batch_size=2048*2,
    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
)

print("Trénujem model na počasí (s tlakom a vlhkosťou)...")

model.fit(
    series=train_target_scaled,
    past_covariates=train_cov_scaled,
    verbose=True
)


pred_scaled = model.predict(n=24, series=train_target_scaled, past_covariates=train_cov_scaled)
prediction = scaler_target.inverse_transform(pred_scaled)


actual = val_target[:24]
print(f"MAE (Chyba v °C): {mae(actual, prediction):.2f} °C")
print(f"MSE  (Chyba na druhú): {mse(actual, prediction):.2f}")

plt.figure(figsize=(12, 6))
actual.plot(label='Skutočná Teplota')
prediction.plot(label='Predpoveď N-BEATS', color='green')
plt.title('Predpoveď počasia na 24h')
plt.legend()
plt.show()



col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

df = pd.read_csv('1D_AAPL.txt', header=None, names=col_names)
df['Date'] = pd.to_datetime(df['Date'])

series = TimeSeries.from_dataframe(df, 'Date', 'Close', freq='D', fill_missing_dates=True)
series = fill_missing_values(series, fill='auto')

train, val = series.split_before(len(series) - 30)

scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)

model = NBEATSModel(
    input_chunk_length=90,
    output_chunk_length=30,
    generic_architecture=True,
    num_stacks=30,
    num_blocks=1,
    layer_widths=512,
    n_epochs=40,
    batch_size=2048,
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": [0],
    }
)

model.fit(train_scaled, verbose=True)

pred_scaled = model.predict(len(val))
prediction = scaler.inverse_transform(pred_scaled)

chyba_mape = mape(val, prediction)
chyba_mae = mae(val, prediction)
chyba_mse = mse(val, prediction)

print(f"MAPE (Relatívna chyba): {chyba_mape:.2f}%")
print(f"MAE (Priemerná chyba v USD): {chyba_mae:.2f} $")
print(f"MSE (Chyba na druhú): {chyba_mse:.2f}")

plt.figure(figsize=(12, 6))
train[-365:].plot(label='História (1 rok)')
val.plot(label='Realita')
prediction.plot(label='N-BEATS Predpoveď', color='green')
plt.title(f'Denné dáta')
plt.legend()
plt.show()