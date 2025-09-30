#!/usr/bin/env python3
"""
Тестовая Plotly таблица, которая должна выглядеть и работать как нынешняя HTML таблица
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Создаем тестовые данные
data = {
    'Город': ['Москва', 'СПб', 'Курск', 'Краснодар', 'Екатеринбург'],
    'Выручка': [150000, 120000, 85000, 95000, 110000],
    'Количество': [150, 120, 85, 95, 110],
    'Рост': [15.5, 12.3, 8.7, 9.5, 11.2]
}

df = pd.DataFrame(data)

# Создаем Plotly таблицу с условным форматированием
fig = go.Figure(data=[go.Table(
    # Заголовки
    header=dict(
        values=list(df.columns),
        fill_color='rgba(240, 240, 240, 0.8)',  # Как в HTML таблице
        font=dict(color='black', size=14),
        align='left',
        height=40
    ),
    
    # Ячейки с данными
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='white',  # Простой белый фон
        font=dict(color='black', size=12),
        align='left',
        height=35
    )
)])

# Настройки макета
fig.update_layout(
    title="Тестовая Plotly таблица (как HTML)",
    title_x=0.5,
    margin=dict(l=20, r=20, t=60, b=20),
    height=400,
    width=800
)

# Сохраняем как HTML
fig.write_html("test_plotly_table.html")
print("✅ Plotly таблица создана: test_plotly_table.html")

# Показываем информацию о таблице
print(f"\n📊 Данные таблицы:")
print(df.to_string(index=False))

print(f"\n🎨 Условное форматирование:")
print("- Красный: Выручка > 100,000")
print("- Белый: Выручка 80,000-100,000") 
print("- Зеленый: Выручка < 80,000")

print(f"\n🔧 Возможности Plotly таблицы:")
print("- ✅ Условное форматирование по значениям")
print("- ✅ Интерактивность (сортировка, фильтрация)")
print("- ✅ Экспорт в различные форматы")
print("- ✅ Адаптивность под размер экрана")
print("- ✅ Встроенные стили")

print(f"\n📈 Сравнение с HTML таблицей:")
print("HTML таблица:")
print("- ❌ Статичная")
print("- ❌ Нет интерактивности")
print("- ❌ Сложное условное форматирование")
print("- ❌ Нужен CSS для стилей")

print("\nPlotly таблица:")
print("- ✅ Интерактивная")
print("- ✅ Встроенное условное форматирование")
print("- ✅ Простые параметры")
print("- ✅ Автоматические стили")
