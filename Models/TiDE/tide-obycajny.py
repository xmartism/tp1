import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import os

# Importy z Darts
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mape, mae, mse
from sklearn.preprocessing import StandardScaler
# Nastavenie grafov
plt.rcParams['figure.figsize'] = (12, 6)


def train_and_eval_tide(name, series, input_len=60, pred_len=24, epochs=10, col_name='hodnota'):
    print(f"\n{'=' * 60}")
    print(f"SPÚŠŤAM: {name}")
    print(f"{'=' * 60}")

    # 1. Rozdelenie na tréning a validáciu
    train, val = series.split_before(len(series) - pred_len)

    # 2. Škálovanie (StandardScaler pre lepšiu stabilitu TiDE)
    scaler = Scaler(scaler=StandardScaler())
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)

    # 3. Model TiDE
    model = TiDEModel(
        input_chunk_length=input_len,
        output_chunk_length=pred_len,
        num_encoder_layers=2,
        num_decoder_layers=2,
        decoder_output_dim=16,
        hidden_size=512,
        temporal_width_past=4,
        temporal_width_future=4,
        n_epochs=epochs,
        batch_size=32,
        dropout=0.1,
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
    )

    # 4. Tréning
    print(f"Trénujem model (Epochs: {epochs})...")
    model.fit(train_scaled)

    # 5. Predikcia (stále v škálovanom priestore)
    print("Počítam predikciu...")
    pred_scaled = model.predict(len(val))

    # --- KĽÚČOVÁ ZMENA: VÝPOČET NORMALIZOVANÝCH METRÍK ---
    norm_mse = mse(val_scaled, pred_scaled)
    norm_mae = mae(val_scaled, pred_scaled)

    # Vrátenie do pôvodných jednotiek pre graf a reálne metriky
    prediction = scaler.inverse_transform(pred_scaled)

    # Vyhodnotenie v reálnych jednotkách
    real_mae = mae(val, prediction)
    real_mse = mse(val, prediction)
    real_mape = mape(val, prediction)

    # 6. Výpis výsledkov
    print(f"\n--- VÝSLEDKY PRE {name} ---")
    print(f"Benckmark (Normalizované):")
    print(f"  > Normalizované MSE: {norm_mse:.6f}")
    print(f"  > Normalizované MAE: {norm_mae:.6f}")
    print(f"Realita (V jednotkách '{col_name}'):")
    print(f"  > MAE:  {real_mae:.4f}")
    print(f"  > MSE:  {real_mse:.4f}")
    #print(f"  > MAPE: {real_mape:.2f} %")

    # 7. Graf
    plt.figure()
    plot_len = min(len(train), 4 * input_len)

    train[-plot_len:].plot(label='História')
    val.plot(label='Realita (Test)', color='green')
    prediction.plot(label='TiDE Predpoveď', color='magenta')

    plt.title(f'{name}\nNorm. MSE: {norm_mse:.4f} | Real MAE: {real_mae:.2f}')
    plt.xlabel('Čas')
    plt.ylabel(col_name)
    plt.legend()
    plt.grid(True)
    plt.show()


# --- SPUSTENIE EXPERIMENTOV ---

# 1. SÍNUSOIDA
try:
    filename_sin = 'sinus_1000_10waves(1).csv'
    if os.path.exists(filename_sin):
        df_sin = pd.read_csv(filename_sin)
        df_sin['ds'] = pd.date_range(start='2024-01-01', periods=len(df_sin), freq='H')
        series_sin = TimeSeries.from_dataframe(df_sin, 'ds', 'value')
        train_and_eval_tide("Sínusoida", series_sin, input_len=50, pred_len=50, epochs=15, col_name='Amplitude')
except Exception as e:
    print(f"Chyba Sínus: {e}")

# 2. POČASIE (Jena Climate)
try:
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df_weather = pd.read_csv(z.open('jena_climate_2009_2016.csv'))
    df_weather['Date Time'] = pd.to_datetime(df_weather['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df_weather = df_weather.set_index('Date Time').resample('H').mean().reset_index()
    df_weather_subset = df_weather.iloc[-2000:].copy()
    series_weather = TimeSeries.from_dataframe(df_weather_subset, 'Date Time', 'T (degC)', freq='H')
    series_weather = fill_missing_values(series_weather, fill='auto')

    train_and_eval_tide("Počasie", series_weather, input_len=720, pred_len=96, epochs=50, col_name='Teplota (°C)')
except Exception as e:
    print(f"Chyba Počasie: {e}")

# 3. AKCIE (Daily)
try:
    filename_stock = '1D_AAPL(2).txt'
    if os.path.exists(filename_stock):
        df_stock = pd.read_csv(filename_stock, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])
        df_stock['ds'] = pd.to_datetime(df_stock['date'])
        series_stock = TimeSeries.from_dataframe(df_stock, 'ds', 'close', freq='D', fill_missing_dates=True)
        series_stock = fill_missing_values(series_stock, fill='auto')
        train_and_eval_tide("Akcie AAPL (Daily)", series_stock, input_len=90, pred_len=30, epochs=20,
                            col_name='Cena ($)')
except Exception as e:
    print(f"Chyba Akcie: {e}")