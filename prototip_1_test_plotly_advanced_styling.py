#!/usr/bin/env python3
"""
Прототип стандартной Plotly таблицы
- Левая колонка по левому краю, все правые по правому
- По горизонтали выравнивание по центру
- Цвет - прозрачный
- Заголовок серый, непрозрачный
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Создаем данные с разными типами для демонстрации стилей (в 3 раза больше)
np.random.seed(42)
data = {
    'Город': [f'Город_{i}' for i in range(1, 46)],  # 45 строк вместо 15
    'Выручка': np.random.randint(50000, 200000, 45),
    'Количество': np.random.randint(50, 200, 45),
    'Рост': np.random.uniform(5, 20, 45).round(1),
    'Прибыль': np.random.randint(10000, 50000, 45),
    'Клиенты': np.random.randint(100, 1000, 45)
}

df = pd.DataFrame(data)

# Создаем Plotly таблицу с заголовками
fig = go.Figure(data=[go.Table(
    # Заголовки - серые, непрозрачные
    header=dict(
        values=list(df.columns),
        fill_color='rgba(128, 128, 128, 1.0)',  # Серый, непрозрачный
        font=dict(color='white', size=14, family="Arial", weight='bold'),
        align='center',  # По горизонтали по центру
        height=30,
        line=dict(color='white', width=1)
    ),
    
    # Ячейки с данными - прозрачные, с правильным выравниванием
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='rgba(255, 255, 255, 0.0)',  # Прозрачный
        font=dict(color='black', size=12, family="Arial"),
        align=['left'] + ['right'] * (len(df.columns) - 1),  # Левая колонка влево, остальные вправо
        height=25,
        line=dict(color='lightgray', width=1)
    )
)])

# Настройки макета
fig.update_layout(
    margin=dict(l=0, r=20, t=20, b=20),
    font=dict(family="Arial", size=12),
    
    # Настройки для скролла
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    
    # Стили для скругленных углов
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    
    # Настройки таблицы
    autosize=True,
    showlegend=False,
    
    # Принудительное изменение размера
    width=800,
    height=400
)

# Сохраняем как HTML с дополнительными стилями
html_content = fig.to_html(
    include_plotlyjs=True,
    div_id="plotly-prototype-table",
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
)

# CSS стили для стандартной таблицы
prototype_css = """
<style>
/* Основные стили */
#plotly-prototype-table {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
    background: white !important;
    padding: 5px !important;
    margin: 10px 10px 10px 0 !important;
    width: 800px !important;
    height: 400px !important;
}

/* Принудительное изменение размера таблицы */
#plotly-prototype-table .plotly {
    width: 100% !important;
    height: 100% !important;
}

#plotly-prototype-table .plotly .table {
    width: 100% !important;
    table-layout: fixed !important;
}

.plotly .table {
    border-radius: 10px !important;
    overflow: hidden !important;
    background: transparent !important;
    border: 1px solid #e0e0e0 !important;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    table-layout: fixed !important;
}

.plotly .table .header {
    border-radius: 10px 10px 0 0 !important;
    background: rgba(128, 128, 128, 1.0) !important; /* Серый, непрозрачный */
    color: white !important;
    font-weight: bold !important;
    text-align: center !important;
}

.plotly .table .cell {
    border-radius: 0 !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    width: 16.66% !important;
    box-sizing: border-box !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
}

.plotly .table .cell:hover {
    background-color: rgba(240, 240, 240, 0.5) !important;
}

/* Вертикальное выравнивание по центру */
.plotly .table .cell {
    vertical-align: middle !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

.plotly .table .cell:nth-child(1) {
    justify-content: flex-start !important; /* Левая колонка влево */
}

.plotly .table .cell:nth-child(n+2) {
    justify-content: flex-end !important; /* Правые колонки вправо */
}

/* Скролл для таблицы */
.plotly .table {
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
}

/* Кастомный скроллбар */
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

/* Адаптивность */
@media (max-width: 768px) {
    #plotly-prototype-table {
        margin: 5px 5px 5px 0 !important;
        padding: 5px !important;
        width: 100% !important;
        height: 300px !important;
    }
    
    .plotly .table {
        max-height: 300px !important;
    }
    
    .plotly .table .cell {
        width: 16.66% !important;
        word-wrap: break-word !important;
    }
}

/* Дополнительные эффекты */
.plotly .table .header:hover {
    background: rgba(100, 100, 100, 1.0) !important;
    transition: all 0.2s ease !important;
}

/* Границы таблицы */
.plotly .table {
    border: 1px solid #e0e0e0 !important;
    background: transparent !important;
    border-radius: 15px !important;
}

/* Скругление для последней строки */
.plotly .table .cell:last-child {
    border-radius: 0 0 10px 10px !important;
}
</style>
"""

# Объединяем HTML с CSS
final_html = html_content.replace('</head>', f'{prototype_css}</head>')

# Сохраняем финальный HTML
with open('prototip_1_test_plotly_advanced_styling.html', 'w', encoding='utf-8') as f:
    f.write(final_html)

print("✅ Прототип стандартной Plotly таблицы создан!")
print("📁 Файл: prototip_1_test_plotly_advanced_styling.html")

print(f"\n📊 Данные таблицы (первые 5 строк):")
print(df.head().to_string(index=False))

print(f"\n🎨 Стандартные стили:")
print("- ✅ Левая колонка по левому краю")
print("- ✅ Все правые колонки по правому краю")
print("- ✅ По горизонтали выравнивание по центру")
print("- ✅ Цвет - прозрачный")
print("- ✅ Заголовок серый, непрозрачный")
print("- ✅ Скругленные углы (border-radius: 10px)")
print("- ✅ Скролл при необходимости")
print("- ✅ Адаптивность")

print(f"\n🔧 CSS возможности:")
print("- text-align: left - левая колонка")
print("- text-align: right - правые колонки")
print("- background: transparent - прозрачный фон")
print("- background: rgba(128, 128, 128, 1.0) - серый заголовок")
print("- border-radius: 10px - скругленные углы")
print("- overflow: auto - скролл")

print(f"\n📱 Адаптивность:")
print("- Автоматическое масштабирование")
print("- Оптимизация для мобильных устройств")
print("- Сохранение функциональности")

print(f"\n🚀 Преимущества:")
print("- Стандартный внешний вид")
print("- Простота и читаемость")
print("- Удобная навигация")
print("- Легкая интеграция")
print("- Высокая производительность")
