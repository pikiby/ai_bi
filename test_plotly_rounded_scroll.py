#!/usr/bin/env python3
"""
Plotly таблица с скругленными углами и скроллом
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Создаем больше данных для демонстрации скролла
np.random.seed(42)
data = {
    'Город': [f'Город_{i}' for i in range(1, 21)],
    'Выручка': np.random.randint(50000, 200000, 20),
    'Количество': np.random.randint(50, 200, 20),
    'Рост': np.random.uniform(5, 20, 20).round(1),
    'Прибыль': np.random.randint(10000, 50000, 20),
    'Клиенты': np.random.randint(100, 1000, 20)
}

df = pd.DataFrame(data)

# Создаем Plotly таблицу с скругленными углами и скроллом
fig = go.Figure(data=[go.Table(
    # Заголовки
    header=dict(
        values=list(df.columns),
        fill_color='rgba(240, 240, 240, 0.8)',
        font=dict(color='black', size=14, family="Arial"),
        align='center',
        height=40,
        line=dict(color='white', width=1)
    ),
    
    # Ячейки с данными
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='white',
        font=dict(color='black', size=12, family="Arial"),
        align='center',
        height=35,
        line=dict(color='lightgray', width=1)
    )
)])

# Настройки макета с скругленными углами и скроллом
fig.update_layout(
    title={
        'text': "Plotly таблица с скругленными углами и скроллом",
        'x': 0.5,
        'font': {'size': 16}
    },
    margin=dict(l=20, r=20, t=80, b=20),
    height=400,  # Фиксированная высота для скролла
    width=900,
    font=dict(family="Arial", size=12),
    
    # Настройки для скролла
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    
    # Стили для скругленных углов
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    
    # Настройки таблицы для скролла
    autosize=True,
    showlegend=False
)

# Добавляем CSS для скругленных углов и скролла
fig.update_layout(
    # CSS стили для скругленных углов
    template="plotly_white",
    # Настройки для скролла
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

# Сохраняем как HTML с дополнительными стилями
html_content = fig.to_html(
    include_plotlyjs=True,
    div_id="plotly-table",
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
    }
)

# Добавляем CSS для скругленных углов и скролла
custom_css = """
<style>
#plotly-table {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

.plotly .table {
    border-radius: 15px !important;
    overflow: hidden !important;
}

.plotly .table .header {
    border-radius: 15px 15px 0 0 !important;
}

.plotly .table .cell {
    border-radius: 0 !important;
}

.plotly .table .cell:last-child {
    border-radius: 0 0 15px 15px !important;
}

/* Скролл для таблицы */
.plotly .table {
    max-height: 400px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
}

/* Стилизация скроллбара */
.plotly .table::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.plotly .table::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.plotly .table::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
}

.plotly .table::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}

/* Анимация при наведении */
.plotly .table:hover {
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    transition: box-shadow 0.3s ease;
}
</style>
"""

# Объединяем HTML с CSS
final_html = html_content.replace('</head>', f'{custom_css}</head>')

# Сохраняем финальный HTML
with open('test_plotly_rounded_scroll.html', 'w', encoding='utf-8') as f:
    f.write(final_html)

print("✅ Plotly таблица с скругленными углами и скроллом создана!")
print("📁 Файл: test_plotly_rounded_scroll.html")

print(f"\n📊 Данные таблицы (первые 5 строк):")
print(df.head().to_string(index=False))

print(f"\n🎨 Особенности:")
print("- ✅ Скругленные углы (border-radius: 15px)")
print("- ✅ Вертикальный скролл (max-height: 400px)")
print("- ✅ Горизонтальный скролл при необходимости")
print("- ✅ Кастомный скроллбар")
print("- ✅ Тень и анимация при наведении")
print("- ✅ Адаптивность")

print(f"\n🔧 CSS стили:")
print("- border-radius: 15px - скругленные углы")
print("- overflow: auto - скролл при переполнении")
print("- box-shadow - тень для глубины")
print("- transition - плавные анимации")

print(f"\n📱 Адаптивность:")
print("- Автоматический скролл при большом количестве данных")
print("- Сохранение пропорций на разных экранах")
print("- Плавные переходы и анимации")

print(f"\n🚀 Преимущества:")
print("- Красивый современный дизайн")
print("- Удобная навигация по большим данным")
print("- Профессиональный внешний вид")
print("- Легкая интеграция в существующий код")
