import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.filters.hp_filter import hpfilter

# --- 1. Генеруємо приклад даних ---
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
trend = np.linspace(100, 200, len(dates))
seasonal = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = np.random.normal(0, 5, len(dates))
visits = trend + seasonal + noise

df = pd.DataFrame({'visits': visits}, index=dates)

# --- 2. Згладжування методом скользячого середнього (7 днів) ---
df['MA_7'] = df['visits'].rolling(window=7).mean()

# --- 3. Експоненційне згладжування ---
exp_model = ExponentialSmoothing(df['visits'], trend='add', seasonal='add', seasonal_periods=30)
df['ExpSmooth'] = exp_model.fit().fittedvalues

# --- 4. Фільтр Ходріка–Прескотта ---
trend_hp, cycle_hp = hpfilter(df['visits'], lamb=1600)
df['HP_trend'] = trend_hp
df['HP_cycle'] = cycle_hp

# --- 5. Візуалізація ---
plt.figure(figsize=(12,6))
plt.plot(df['visits'], label='Фактичні дані', alpha=0.5)
plt.plot(df['MA_7'], label='Скользяче середнє (7 днів)')
plt.plot(df['ExpSmooth'], label='Експоненційне згладжування')
plt.plot(df['HP_trend'], label='HP фільтр (тренд)')
plt.title('Методи згладжування та фільтрації часового ряду')
plt.xlabel('Дата')
plt.ylabel('Кількість відвідувань')
plt.legend()
plt.show()
