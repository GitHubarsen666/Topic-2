import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 1. Генеруємо приклад даних (щомісячний часовий ряд) ---
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
trend = np.linspace(100, 200, len(dates))  # зростаючий тренд
seasonal = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)  # сезонний ефект
noise = np.random.normal(0, 5, len(dates))  # випадковий шум
visits = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'visits': visits})
df.set_index('date', inplace=True)

# --- 2. Декомпозиція для виділення тренду ---
decomposition = seasonal_decompose(df['visits'], model='additive', period=30)
trend_component = decomposition.trend.dropna()  # видаляємо NaN
seasonal_component = decomposition.seasonal

# --- 3. Побудова лінійної регресії для тренду ---
X = np.arange(len(trend_component)).reshape(-1,1)
y = trend_component.values
model = LinearRegression()
model.fit(X, y)
trend_pred = model.predict(X)

# --- 4. Побудова повної математичної моделі ---
# Повна модель = тренд (регресія) + сезонність
seasonal_for_model = seasonal_component[trend_component.index]
model_values = trend_pred + seasonal_for_model.values

# --- 5. Візуалізація результатів ---
plt.figure(figsize=(12,5))
plt.plot(df['visits'], label='Фактичні відвідування', alpha=0.5)
plt.plot(trend_component.index, trend_pred, label='Тренд (регресія)', color='red')
plt.plot(trend_component.index, model_values, label='Модель (тренд + сезонність)', color='green')
plt.title('Математична модель економічного процесу (активність користувачів)')
plt.xlabel('Дата')
plt.ylabel('Кількість відвідувань')
plt.legend()
plt.show()
