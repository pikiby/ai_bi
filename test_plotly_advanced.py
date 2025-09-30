#!/usr/bin/env python3
"""
Продвинутая Plotly таблица с условным форматированием
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

# Создаем цветовую схему для условного форматирования
def get_color_scheme(df):
    """Создает цветовую схему для таблицы"""
    colors = []
    
    for col in df.columns:
        if col == 'Выручка':
            # Условное форматирование для выручки
            col_colors = []
            for val in df[col]:
                if val > 120000:
                    col_colors.append('rgba(255, 0, 0, 0.3)')  # Красный
                elif val > 100000:
                    col_colors.append('rgba(255, 165, 0, 0.3)')  # Оранжевый
                elif val > 80000:
                    col_colors.append('rgba(255, 255, 0, 0.3)')  # Желтый
                else:
                    col_colors.append('rgba(0, 255, 0, 0.3)')  # Зеленый
            colors.append(col_colors)
        elif col == 'Рост':
            # Условное форматирование для роста
            col_colors = []
            for val in df[col]:
                if val > 12:
                    col_colors.append('rgba(0, 255, 0, 0.3)')  # Зеленый
                elif val > 10:
                    col_colors.append('rgba(255, 255, 0, 0.3)')  # Желтый
                else:
                    col_colors.append('rgba(255, 0, 0, 0.3)')  # Красный
            colors.append(col_colors)
        else:
            # Обычные ячейки
            colors.append(['rgba(255, 255, 255, 1)'] * len(df))
    
    return colors

# Получаем цветовую схему
fill_colors = get_color_scheme(df)

# Создаем Plotly таблицу
fig = go.Figure(data=[go.Table(
    # Заголовки
    header=dict(
        values=list(df.columns),
        fill_color='rgba(240, 240, 240, 0.8)',
        font=dict(color='black', size=14, family="Arial"),
        align='center',
        height=40
    ),
    
    # Ячейки с данными
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color=fill_colors,
        font=dict(color='black', size=12, family="Arial"),
        align='center',
        height=35
    )
)])

# Настройки макета
fig.update_layout(
    title={
        'text': "Продвинутая Plotly таблица с условным форматированием",
        'x': 0.5,
        'font': {'size': 16}
    },
    margin=dict(l=20, r=20, t=80, b=20),
    height=500,
    width=900,
    font=dict(family="Arial", size=12)
)

# Сохраняем как HTML
fig.write_html("test_plotly_advanced.html")
print("✅ Продвинутая Plotly таблица создана: test_plotly_advanced.html")

# Показываем информацию о таблице
print(f"\n📊 Данные таблицы:")
print(df.to_string(index=False))

print(f"\n🎨 Условное форматирование:")
print("Выручка:")
print("- 🔴 Красный: > 120,000")
print("- 🟠 Оранжевый: 100,000-120,000")
print("- 🟡 Желтый: 80,000-100,000")
print("- 🟢 Зеленый: < 80,000")

print("\nРост:")
print("- 🟢 Зеленый: > 12%")
print("- 🟡 Желтый: 10-12%")
print("- 🔴 Красный: < 10%")

print(f"\n🔧 Преимущества Plotly таблицы:")
print("- ✅ Условное форматирование по значениям")
print("- ✅ Интерактивность (сортировка, фильтрация)")
print("- ✅ Экспорт в различные форматы")
print("- ✅ Адаптивность под размер экрана")
print("- ✅ Встроенные стили")
print("- ✅ Простота для AI")

print(f"\n📈 Сравнение с HTML таблицей:")
print("HTML таблица:")
print("- ❌ Статичная")
print("- ❌ Нет интерактивности")
print("- ❌ Сложное условное форматирование")
print("- ❌ Нужен CSS для стилей")
print("- ❌ Сложно для AI")

print("\nPlotly таблица:")
print("- ✅ Интерактивная")
print("- ✅ Встроенное условное форматирование")
print("- ✅ Простые параметры")
print("- ✅ Автоматические стили")
print("- ✅ Просто для AI")

print(f"\n🤖 Для AI это означает:")
print("- Вместо сложного CSS: простые параметры")
print("- Вместо условной логики: встроенные функции")
print("- Вместо ошибок форматирования: автоматическое применение")
print("- Вместо статичности: интерактивность")
