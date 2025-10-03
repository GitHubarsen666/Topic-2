import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 1. Генеруємо приклад даних ---
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
trend = np.linspace(50, 150, len(dates))  # зростаючий тренд
seasonal = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)  # річний сезонний ефект
noise = np.random.normal(0, 5, len(dates))  # випадковий шум
visits = trend + seasonal + noise

# створюємо DataFrame
df = pd.DataFrame({'date': dates, 'visits': visits})
df.set_index('date', inplace=True)

# --- 2. Візуалізація часового ряду ---
plt.figure(figsize=(12,5))
plt.plot(df['visits'], label='Щоденні відвідування')
plt.title('Активність користувачів електронної бібліотеки')
plt.xlabel('Дата')
plt.ylabel('Кількість відвідувань')
plt.legend()
plt.show()

# --- 3. Декомпозиція часового ряду ---
decomposition = seasonal_decompose(df['visits'], model='additive', period=30)
trend_component = decomposition.trend
seasonal_component = decomposition.seasonal
residual_component = decomposition.resid

# --- 4. Візуалізація компонентів ---
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(trend_component, color='blue')
plt.title('Тренд активності користувачів')

plt.subplot(3,1,2)
plt.plot(seasonal_component, color='orange')
plt.title('Сезонність (періодичні коливання)')

plt.subplot(3,1,3)
plt.plot(residual_component, color='green')
plt.title('Рандомні коливання (залишки)')

plt.tight_layout()
plt.show()
