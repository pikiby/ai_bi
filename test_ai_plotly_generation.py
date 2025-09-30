#!/usr/bin/env python3
"""
Пример того, как AI может генерировать Plotly таблицы
"""

import plotly.graph_objects as go
import pandas as pd
import json

# Создаем тестовые данные
data = {
    'Город': ['Москва', 'СПб', 'Курск', 'Краснодар', 'Екатеринбург'],
    'Выручка': [150000, 120000, 85000, 95000, 110000],
    'Количество': [150, 120, 85, 95, 110],
    'Рост': [15.5, 12.3, 8.7, 9.5, 11.2]
}

df = pd.DataFrame(data)

# Пример того, как AI может генерировать стили для Plotly
def ai_generate_plotly_styles(user_request, df):
    """
    Симуляция того, как AI может генерировать стили для Plotly таблицы
    """
    styles = {
        'header_fill_color': 'rgba(240, 240, 240, 0.8)',
        'cells_fill_color': 'white',
        'font_color': 'black',
        'font_size': 12,
        'conditional_formatting': []
    }
    
    # Анализируем запрос пользователя
    if 'красн' in user_request.lower():
        if 'столбец' in user_request.lower():
            # Выделить столбец красным
            if 'Курск' in user_request:
                # Найти столбец с Курском
                for col in df.columns:
                    if 'Курск' in df[col].values:
                        styles['conditional_formatting'].append({
                            'column': col,
                            'condition': f'value == "Курск"',
                            'color': 'red'
                        })
        elif 'значение' in user_request.lower():
            # Выделить значения красным
            if 'выше 8' in user_request:
                styles['conditional_formatting'].append({
                    'column': 'Рост',
                    'condition': 'value > 8',
                    'color': 'red'
                })
    
    return styles

# Примеры запросов пользователя
user_requests = [
    "сделать красным столбец с надписью Курск",
    "выделить красным значения выше 8",
    "сделать первую строку красной",
    "добавить чередование строк"
]

print("🤖 Примеры того, как AI может генерировать Plotly таблицы:")
print("=" * 60)

for i, request in enumerate(user_requests, 1):
    print(f"\n{i}. Запрос: '{request}'")
    
    # AI генерирует стили
    styles = ai_generate_plotly_styles(request, df)
    
    print(f"   AI генерирует стили:")
    print(f"   {json.dumps(styles, indent=2, ensure_ascii=False)}")
    
    # Показываем, как это будет выглядеть в Plotly
    if styles['conditional_formatting']:
        print(f"   → Plotly код:")
        for rule in styles['conditional_formatting']:
            print(f"     fill_color = ['red' if {rule['condition']} else 'white' for val in df['{rule['column']}']]")
    else:
        print(f"   → Plotly код: fill_color='white'")

print(f"\n🎯 Преимущества для AI:")
print("=" * 60)
print("✅ Простые параметры вместо сложного CSS")
print("✅ Встроенное условное форматирование")
print("✅ Автоматическое применение стилей")
print("✅ Меньше ошибок форматирования")
print("✅ Интерактивность из коробки")

print(f"\n📊 Сравнение сложности:")
print("=" * 60)
print("HTML + CSS (текущая система):")
print("- ❌ Сложные селекторы")
print("- ❌ Условная логика в CSS")
print("- ❌ Ошибки форматирования")
print("- ❌ Статичность")

print("\nPlotly (предлагаемая система):")
print("- ✅ Простые параметры")
print("- ✅ Встроенные функции")
print("- ✅ Автоматическое применение")
print("- ✅ Интерактивность")

print(f"\n🔧 Пример интеграции в app.py:")
print("=" * 60)
print("""
# Вместо сложного CSS генерации:
def _generate_plotly_table(df, style_meta):
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color=style_meta.get('header_fill_color', 'lightgray')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=style_meta.get('cells_fill_color', 'white')
        )
    )])
    return fig

# AI генерирует простые параметры:
table_style = {
    'header_fill_color': 'rgba(240, 240, 240, 0.8)',
    'cells_fill_color': 'white',
    'conditional_formatting': [
        {'column': 'Рост', 'condition': 'value > 8', 'color': 'red'}
    ]
}
""")

print(f"\n🚀 Рекомендация:")
print("=" * 60)
print("Переход на Plotly таблицы решит большинство проблем:")
print("- AI будет генерировать простые параметры")
print("- Меньше ошибок форматирования")
print("- Больше возможностей для пользователя")
print("- Проще поддержка и развитие")
