import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Генеруємо приклад даних ---
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
trend = np.linspace(100, 200, len(dates))  # тренд
seasonal = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)  # сезонність
noise = np.random.normal(0, 5, len(dates))
visits = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'visits': visits})
df.set_index('date', inplace=True)

# --- 2. Побудова моделі ARIMA ---
# Параметри p,d,q можна оптимізувати, тут приклад: (1,1,1)
model = ARIMA(df['visits'], order=(1,1,1))
model_fit = model.fit()

# --- 3. Прогноз на наступні 60 днів ---
forecast = model_fit.forecast(steps=60)
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60)
forecast_series = pd.Series(forecast, index=forecast_dates)

# --- 4. Візуалізація ---
plt.figure(figsize=(12,5))
plt.plot(df['visits'], label='Фактичні відвідування')
plt.plot(forecast_series, label='Прогноз на майбутнє', color='red')
plt.title('Прогноз майбутньої активності користувачів')
plt.xlabel('Дата')
plt.ylabel('Кількість відвідувань')
plt.legend()
plt.show()
