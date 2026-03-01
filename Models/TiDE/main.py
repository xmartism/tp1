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

# Nastavenie grafov
plt.rcParams['figure.figsize'] = (12, 6)


def train_and_eval_tide(name, series, input_len=60, pred_len=24, epochs=10, col_name='hodnota', show_mape=True):
    print(f"\n{'=' * 50}")
    print(f"SPÚŠŤAM: {name}")
    print(f"{'=' * 50}")

    # 1. Rozdelenie na tréning a validáciu
    train, val = series.split_before(len(series) - pred_len)

    # 2. Škálovanie (TiDE potrebuje 0-1)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)

    # 3. Model TiDE
    model = TiDEModel(
        input_chunk_length=input_len,
        output_chunk_length=pred_len,
        num_encoder_layers=2,
        num_decoder_layers=2,
        decoder_output_dim=16,
        hidden_size=128,
        temporal_width_past=4,
        temporal_width_future=4,
        n_epochs=epochs,
        batch_size=32,
        dropout=0.1,
        random_state=42,
        # GPU nastavenie
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
    )

    # 4. Tréning
    print(f"Trénujem model (Epochs: {epochs})...")
    model.fit(train_scaled)

    # 5. Predikcia
    print("Počítam predikciu...")
    pred_scaled = model.predict(len(val))

    # Vrátenie do pôvodných jednotiek
    prediction = scaler.inverse_transform(pred_scaled)

    # 6. Vyhodnotenie
    chyba_mae = mae(val, prediction)
    chyba_mse = mse(val, prediction)

    # Výpis výsledkov
    print(f"\n--- VÝSLEDKY PRE {name} ---")
    if show_mape:
        chyba_mape = mape(val, prediction)
        print(f"MAPE (Percentuálna chyba): {chyba_mape:.2f} %")

    print(f"MAE  (Absolútna chyba):    {chyba_mae:.4f}")
    print(f"MSE  (Kvadratická chyba):  {chyba_mse:.6f}")

    # 7. Graf
    plt.figure()
    plot_len = min(len(train), 4 * input_len)

    train[-plot_len:].plot(label='História')
    val.plot(label='Realita', color='green')
    prediction.plot(label='TiDE Predpoveď', color='magenta', low_quantile=0.05, high_quantile=0.95)

    # Dynamický nadpis podľa toho, či chceme vidieť MAPE
    if show_mape:
        title_text = f'{name}\nMAPE: {chyba_mape:.2f}% | MAE: {chyba_mae:.4f}'
    else:
        title_text = f'{name}\nMAE: {chyba_mae:.4f} | MSE: {chyba_mse:.4f}'

    plt.title(title_text)
    plt.xlabel('Čas')
    plt.ylabel(col_name)
    plt.legend()
    plt.grid(True)
    plt.show()


# 1. SÍNUSOIDA

try:
    filename_sin = 'sinus_1000_10waves(1).csv'
    if os.path.exists(filename_sin):
        df_sin = pd.read_csv(filename_sin)
        df_sin['ds'] = pd.date_range(start='2024-01-01', periods=len(df_sin), freq='H')
        series_sin = TimeSeries.from_dataframe(df_sin, 'ds', 'value')

        # TU JE ZMENA: show_mape=False
        train_and_eval_tide("Sínusoida", series_sin, input_len=50, pred_len=50, epochs=15, col_name='Amplitude',
                            show_mape=False)
except Exception as e:
    print(f"Chyba Sínus: {e}")

# 2. POČASIE

try:
    print("\nSťahujem dáta o počasí...")
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df_weather = pd.read_csv(z.open('jena_climate_2009_2016.csv'))
    df_weather['Date Time'] = pd.to_datetime(df_weather['Date Time'], format='%d.%m.%Y %H:%M:%S')

    df_weather = df_weather.set_index('Date Time').resample('H').mean().reset_index()
    df_weather_subset = df_weather.iloc[-2000:].copy()

    series_weather = TimeSeries.from_dataframe(df_weather_subset, 'Date Time', 'T (degC)', freq='H')
    series_weather = fill_missing_values(series_weather, fill='auto')

    train_and_eval_tide("Počasie", series_weather, input_len=72, pred_len=24, epochs=15, col_name='Teplota (°C)',
                        show_mape=False)
except Exception as e:
    print(f"Chyba Počasie: {e}")

# 3. AKCIE 1-DAY 

try:
    filename_stock = '1D_AAPL(2).txt'
    if os.path.exists(filename_stock):
        print(f"\nNačítavam akcie: {filename_stock}")

        # ZMENA: Odstránený stĺpec 'time', načítavame len dátum a ceny
        df_stock = pd.read_csv(filename_stock, header=None,
                               names=['date', 'open', 'high', 'low', 'close', 'volume'])

        # ZMENA: Konverzia iba dátumu (bez času)
        df_stock['ds'] = pd.to_datetime(df_stock['date'])

        # ZMENA: freq='D' pre denné dáta (namiesto 30min)
        series_stock = TimeSeries.from_dataframe(df_stock, 'ds', 'close', freq='D', fill_missing_dates=True)
        series_stock = fill_missing_values(series_stock, fill='auto')

        # Nastavené pre denné dáta:
        # input_len=96 (pozerá sa 3 mesiace dozadu)
        # pred_len=30 (predpovedá 1 mesiac dopredu)
        train_and_eval_tide("Akcie (1D Daily)", series_stock, input_len=96, pred_len=30, epochs=20, col_name='Cena ($)',
                            show_mape=False)
    else:
        print("Súbor s akciami chýba.")
except Exception as e:
    print(f"Chyba Akcie: {e}")