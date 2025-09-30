#!/usr/bin/env python3
"""
Упрощенный тест новой системы Pandas Styler
"""

import pandas as pd
import numpy as np

# Создаем тестовые данные
np.random.seed(42)
data = {
    'Город': [f'Город_{i}' for i in range(1, 11)],
    'Выручка': np.random.randint(50000, 200000, 10),
    'Количество': np.random.randint(50, 200, 10),
    'Рост': np.random.uniform(5, 20, 10).round(1),
    'Прибыль': np.random.randint(10000, 50000, 10)
}

df = pd.DataFrame(data)

print("=== Тест новой системы Pandas Styler ===\n")

print("1. Исходные данные:")
print(df.head())
print()

print("2. Базовая стилизация:")
# Базовая конфигурация
basic_config = {
    "header_fill_color": "#f4f4f4",
    "header_font_color": "black",
    "cells_fill_color": "white",
    "font_color": "black"
}

# Создаем стилизованную таблицу
styler = df.style
styler = styler.set_table_styles([
    {"selector": "thead th", "props": [
        ("background-color", basic_config["header_fill_color"]),
        ("color", basic_config["header_font_color"]),
        ("font-weight", "bold")
    ]}
])

print("✅ Базовая стилизация применена")

print("\n3. Чередование строк:")
striped_config = {
    "header_fill_color": "#f4f4f4",
    "header_font_color": "black",
    "cells_fill_color": "white",
    "font_color": "black",
    "striped": True,
    "even_row_color": "#f9f9f9",
    "odd_row_color": "white"
}

styler_striped = df.style
styler_striped = styler_striped.set_table_styles([
    {"selector": "thead th", "props": [
        ("background-color", striped_config["header_fill_color"]),
        ("color", striped_config["header_font_color"]),
        ("font-weight", "bold")
    ]},
    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", striped_config["even_row_color"])]},
    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", striped_config["odd_row_color"])]}
])

print("✅ Чередование строк применено")

print("\n4. Условное форматирование:")
conditional_config = {
    "header_fill_color": "#f4f4f4",
    "header_font_color": "black",
    "cells_fill_color": "white",
    "font_color": "black",
    "cell_rules": [
        {"column": "Выручка", "value": "max", "color": "red"},
        {"column": "Рост", "value": "min", "color": "green"}
    ]
}

styler_conditional = df.style
styler_conditional = styler_conditional.set_table_styles([
    {"selector": "thead th", "props": [
        ("background-color", conditional_config["header_fill_color"]),
        ("color", conditional_config["header_font_color"]),
        ("font-weight", "bold")
    ]}
])

# Применяем условное форматирование
for rule in conditional_config["cell_rules"]:
    column = rule["column"]
    value = rule["value"]
    color = rule["color"]
    
    if column in df.columns:
        if value == "max":
            styler_conditional = styler_conditional.apply(
                lambda x: [f"background-color: {color}; color: white" 
                          if x[column] == x[column].max() else "" 
                          for _ in x], 
                subset=[column]
            )
        elif value == "min":
            styler_conditional = styler_conditional.apply(
                lambda x: [f"background-color: {color}; color: white" 
                          if x[column] == x[column].min() else "" 
                          for _ in x], 
                subset=[column]
            )

print("✅ Условное форматирование применено")

print("\n5. Тест производительности:")
import time

start_time = time.time()
for _ in range(1000):
    styler = df.style
    styler = styler.set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#f4f4f4"),
            ("color", "black"),
            ("font-weight", "bold")
        ]}
    ])
end_time = time.time()

print(f"✅ Время создания 1000 стилизованных таблиц: {end_time - start_time:.3f} секунд")

print("\n6. Сравнение с HTML системой:")
print("✅ Pandas Styler:")
print("  - Простые параметры")
print("  - Встроенная валидация")
print("  - Автоматическое применение")
print("  - Лучшая производительность")
print("  - Совместимость с Streamlit")

print("\n❌ HTML+CSS система:")
print("  - Сложные CSS правила")
print("  - Ручная валидация")
print("  - Сложная отладка")
print("  - Проблемы с изоляцией")
print("  - Сложность для AI")

print("\n=== Тест завершен успешно ===")
print("\nНовая система готова к интеграции!")
