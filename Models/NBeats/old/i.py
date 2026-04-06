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

df_sin = pd.read_csv('testData/sinus_1000_10waves.csv')
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

df = pd.read_csv('testData/1D_AAPL.txt', header=None, names=col_names)
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