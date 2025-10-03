import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# --- 1. Генеруємо приклад даних ---
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

# Активність користувачів
trend_visits = np.linspace(100, 200, len(dates))
seasonal_visits = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise_visits = np.random.normal(0, 5, len(dates))
visits = trend_visits + seasonal_visits + noise_visits

# Кількість нових матеріалів
trend_materials = np.linspace(10, 30, len(dates))
seasonal_materials = 5 * np.sin(2 * np.pi * dates.dayofyear / 365 + np.pi/6)
noise_materials = np.random.normal(0, 2, len(dates))
materials = trend_materials + seasonal_materials + noise_materials

# Створюємо DataFrame
df = pd.DataFrame({'visits': visits, 'materials': materials}, index=dates)

# --- 2. Візуалізація часових рядів ---
plt.figure(figsize=(12,5))
plt.plot(df['visits'], label='Активність користувачів')
plt.plot(df['materials'], label='Нові матеріали')
plt.title('Часові ряди: активність vs. нові матеріали')
plt.xlabel('Дата')
plt.ylabel('Кількість')
plt.legend()
plt.show()

# --- 3. Кореляційний аналіз ---
correlation = df['visits'].corr(df['materials'])
print(f"Коефіцієнт кореляції Пірсона: {correlation:.2f}")

# --- 4. Тест причинності Грейнджера ---
# maxlag=7 перевіряє, чи впливає ряд на інший протягом тижня
granger_test = grangercausalitytests(df[['visits', 'materials']], maxlag=7, verbose=True)
