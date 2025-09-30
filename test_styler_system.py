#!/usr/bin/env python3
"""
Тест новой системы Streamlit + Pandas Styler
"""

import pandas as pd
import streamlit as st
import numpy as np

# Импортируем новую систему
from streamlit_styler_system import (
    create_styled_dataframe,
    generate_styler_config,
    render_styled_table,
    STANDARD_STYLER_CONFIG
)

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

st.title("Тест новой системы Streamlit + Pandas Styler")

st.markdown("### 1. Базовая таблица")
st.dataframe(df, use_container_width=True)

st.markdown("### 2. Таблица с базовыми стилями")
basic_config = {
    "header_fill_color": "#f4f4f4",
    "header_font_color": "black",
    "cells_fill_color": "white",
    "font_color": "black"
}

styled_df = create_styled_dataframe(df, basic_config)
st.dataframe(styled_df, use_container_width=True)

st.markdown("### 3. Таблица с чередованием строк")
striped_config = {
    "header_fill_color": "#f4f4f4",
    "header_font_color": "black",
    "cells_fill_color": "white",
    "font_color": "black",
    "striped": True,
    "even_row_color": "#f9f9f9",
    "odd_row_color": "white"
}

styled_df_striped = create_styled_dataframe(df, striped_config)
st.dataframe(styled_df_striped, use_container_width=True)

st.markdown("### 4. Таблица с условным форматированием")
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

styled_df_conditional = create_styled_dataframe(df, conditional_config)
st.dataframe(styled_df_conditional, use_container_width=True)

st.markdown("### 5. AI генерация конфигурации")
user_requests = [
    "сделать чередование строк",
    "выделить максимум красным",
    "выделить минимум зеленым"
]

for request in user_requests:
    st.markdown(f"**Запрос:** {request}")
    config = generate_styler_config(request, df)
    st.json(config)
    
    styled_df_ai = create_styled_dataframe(df, config)
    st.dataframe(styled_df_ai, use_container_width=True)
    st.markdown("---")

st.markdown("### 6. Полная интеграция")
render_styled_table(
    df, 
    conditional_config,
    title="Тестовая таблица",
    caption="Таблица с условным форматированием"
)

st.markdown("### 7. Сравнение производительности")
import time

# Тест скорости
start_time = time.time()
for _ in range(100):
    styled_df = create_styled_dataframe(df, basic_config)
end_time = time.time()

st.info(f"Время создания 100 стилизованных таблиц: {end_time - start_time:.3f} секунд")

st.markdown("### 8. Информация о системе")
st.markdown("""
**Новая система Streamlit + Pandas Styler:**

✅ **Преимущества:**
- Простые параметры для AI
- Встроенная интерактивность Streamlit
- Автоматическое применение стилей
- Меньше ошибок форматирования
- Лучшая производительность

✅ **Возможности:**
- Базовые стили (цвета, шрифты)
- Условное форматирование
- Чередование строк
- Выделение ячеек
- AI генерация конфигурации

✅ **Интеграция:**
- Совместимость с существующим кодом
- Простая миграция
- Расширяемость
""")
