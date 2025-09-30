#!/usr/bin/env python3
"""
Продвинутая Plotly таблица с множеством стилей
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Создаем данные с разными типами для демонстрации стилей
np.random.seed(42)
data = {
    'Город': [f'Город_{i}' for i in range(1, 16)],
    'Выручка': np.random.randint(50000, 200000, 15),
    'Количество': np.random.randint(50, 200, 15),
    'Рост': np.random.uniform(5, 20, 15).round(1),
    'Прибыль': np.random.randint(10000, 50000, 15),
    'Клиенты': np.random.randint(100, 1000, 15)
}

df = pd.DataFrame(data)

# Создаем цветовую схему для условного форматирования
def get_advanced_colors(df):
    """Создает продвинутую цветовую схему"""
    colors = []
    
    for col in df.columns:
        if col == 'Выручка':
            # Градиент для выручки
            col_colors = []
            max_val = df[col].max()
            min_val = df[col].min()
            for val in df[col]:
                ratio = (val - min_val) / (max_val - min_val)
                if ratio > 0.8:
                    col_colors.append('rgba(255, 0, 0, 0.3)')  # Красный
                elif ratio > 0.6:
                    col_colors.append('rgba(255, 165, 0, 0.3)')  # Оранжевый
                elif ratio > 0.4:
                    col_colors.append('rgba(255, 255, 0, 0.3)')  # Желтый
                elif ratio > 0.2:
                    col_colors.append('rgba(144, 238, 144, 0.3)')  # Светло-зеленый
                else:
                    col_colors.append('rgba(0, 255, 0, 0.3)')  # Зеленый
            colors.append(col_colors)
        elif col == 'Рост':
            # Цвета для роста
            col_colors = []
            for val in df[col]:
                if val > 15:
                    col_colors.append('rgba(0, 255, 0, 0.4)')  # Зеленый
                elif val > 10:
                    col_colors.append('rgba(255, 255, 0, 0.4)')  # Желтый
                else:
                    col_colors.append('rgba(255, 0, 0, 0.4)')  # Красный
            colors.append(col_colors)
        else:
            # Обычные ячейки
            colors.append(['rgba(255, 255, 255, 1)'] * len(df))
    
    return colors

# Получаем цветовую схему
fill_colors = get_advanced_colors(df)

# Создаем Plotly таблицу
fig = go.Figure(data=[go.Table(
    # Заголовки с градиентом
    header=dict(
        values=list(df.columns),
        fill_color='rgba(70, 130, 180, 0.8)',  # Синий градиент
        font=dict(color='white', size=14, family="Arial", weight='bold'),
        align='center',
        height=45,
        line=dict(color='white', width=2)
    ),
    
    # Ячейки с данными
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color=fill_colors,
        font=dict(color='black', size=12, family="Arial"),
        align='center',
        height=40,
        line=dict(color='white', width=1)
    )
)])

# Настройки макета
fig.update_layout(
    title={
        'text': "Продвинутая Plotly таблица с множеством стилей",
        'x': 0.5,
        'font': {'size': 18, 'color': 'darkblue'}
    },
    margin=dict(l=20, r=20, t=100, b=20),
    height=500,
    width=1000,
    font=dict(family="Arial", size=12),
    
    # Настройки для скролла
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    
    # Стили для скругленных углов
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    
    # Настройки таблицы
    autosize=True,
    showlegend=False
)

# Сохраняем как HTML с дополнительными стилями
html_content = fig.to_html(
    include_plotlyjs=True,
    div_id="plotly-advanced-table",
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True
    }
)

# Продвинутые CSS стили
advanced_css = """
<style>
/* Основные стили */
#plotly-advanced-table {
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 20px !important;
    margin: 20px !important;
}

.plotly .table {
    border-radius: 20px !important;
    overflow: hidden !important;
    background: white !important;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.1) !important;
}

.plotly .table .header {
    border-radius: 20px 20px 0 0 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: bold !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
}

.plotly .table .cell {
    border-radius: 0 !important;
    transition: all 0.3s ease !important;
}

.plotly .table .cell:hover {
    background-color: rgba(102, 126, 234, 0.1) !important;
    transform: scale(1.02) !important;
}

/* Скролл для таблицы */
.plotly .table {
    max-height: 450px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
}

/* Кастомный скроллбар */
.plotly .table::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

.plotly .table::-webkit-scrollbar-track {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 6px;
}

.plotly .table::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 6px;
    border: 2px solid white;
}

.plotly .table::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}

/* Анимации */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

#plotly-advanced-table {
    animation: fadeIn 0.8s ease-out !important;
}

/* Адаптивность */
@media (max-width: 768px) {
    #plotly-advanced-table {
        margin: 10px !important;
        padding: 10px !important;
    }
    
    .plotly .table {
        max-height: 300px !important;
    }
}

/* Дополнительные эффекты */
.plotly .table .header:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    transform: scale(1.01) !important;
    transition: all 0.3s ease !important;
}

/* Градиентные границы */
.plotly .table {
    border: 3px solid transparent !important;
    background: linear-gradient(white, white) padding-box,
                linear-gradient(135deg, #667eea 0%, #764ba2 100%) border-box !important;
}
</style>
"""

# Объединяем HTML с CSS
final_html = html_content.replace('</head>', f'{advanced_css}</head>')

# Сохраняем финальный HTML
with open('test_plotly_advanced_styling.html', 'w', encoding='utf-8') as f:
    f.write(final_html)

print("✅ Продвинутая Plotly таблица создана!")
print("📁 Файл: test_plotly_advanced_styling.html")

print(f"\n📊 Данные таблицы (первые 5 строк):")
print(df.head().to_string(index=False))

print(f"\n🎨 Продвинутые стили:")
print("- ✅ Скругленные углы (border-radius: 20px)")
print("- ✅ Градиентные фоны и границы")
print("- ✅ Анимации и переходы")
print("- ✅ Кастомный скроллбар с градиентом")
print("- ✅ Эффекты при наведении")
print("- ✅ Адаптивность для мобильных")
print("- ✅ Тени и глубина")

print(f"\n🔧 CSS возможности:")
print("- border-radius: 20px - скругленные углы")
print("- linear-gradient - градиентные фоны")
print("- box-shadow - тени и глубина")
print("- transition - плавные анимации")
print("- @keyframes - кастомные анимации")
print("- @media - адаптивность")

print(f"\n📱 Адаптивность:")
print("- Автоматическое масштабирование")
print("- Оптимизация для мобильных устройств")
print("- Сохранение функциональности")

print(f"\n🚀 Преимущества:")
print("- Профессиональный внешний вид")
print("- Современный дизайн")
print("- Удобная навигация")
print("- Легкая интеграция")
print("- Высокая производительность")
