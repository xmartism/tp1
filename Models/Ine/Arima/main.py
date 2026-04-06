import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


data = pd.read_csv("sinus.csv")


if not {"time", "value"}.issubset(data.columns):
    raise ValueError("CSV musí obsahovať stĺpce 'time' a 'value'")

time = data["time"]
values = data["value"]


train_size = int(len(values) * 0.8)
train, test = values[:train_size], values[train_size:]
time_train, time_test = time[:train_size], time[train_size:]


# (p, d, q) môžeš upraviť podľa potreby
model = ARIMA(train, order=(3, 0, 2))
model_fit = model.fit()


forecast = model_fit.forecast(steps=len(test))

plt.figure(figsize=(10, 5))
plt.plot(time, values, label="Skutočné dáta")
plt.plot(time_test, forecast, label="Predikcia", color="red")
plt.axvline(x=time.iloc[train_size], color="gray", linestyle="--", label="Rozdelenie Train/Test")
plt.title("ARIMA predikcia na dátach z 'sinus.csv'")
plt.xlabel("Čas")
plt.ylabel("Hodnota")
plt.legend()
plt.show()
