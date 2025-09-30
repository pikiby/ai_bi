#!/usr/bin/env python3
"""
Новая система таблиц на основе Streamlit + Pandas Styler
Заменит текущую HTML+CSS систему
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
import re

# ==================== КОНСТАНТЫ ====================

# Стандартные стили для Pandas Styler
STANDARD_STYLER_CONFIG = {
    "header_style": {
        "background-color": "#f4f4f4",
        "color": "black",
        "font-weight": "bold"
    },
    "cell_style": {
        "background-color": "white",
        "color": "black"
    },
    "border": "1px solid #ddd",
    "text-align": "left"
}

# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def create_styled_dataframe(df: pd.DataFrame, style_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Создает стилизованный DataFrame с помощью Pandas Styler.
    
    Args:
        df: Исходный DataFrame
        style_config: Конфигурация стилей
        
    Returns:
        Стилизованный DataFrame
    """
    styler = df.style
    
    # Применяем базовые стили
    styler = _apply_base_styles(styler, style_config)
    
    # Применяем условное форматирование
    styler = _apply_conditional_formatting(styler, df, style_config)
    
    # Применяем выделение ячеек
    styler = _apply_cell_highlighting(styler, df, style_config)
    
    # Применяем чередование строк
    styler = _apply_row_striping(styler, style_config)
    
    return styler

def _apply_base_styles(styler, style_config: Dict[str, Any]):
    """Применяет базовые стили к таблице"""
    
    # Стили заголовков
    header_style = style_config.get("header_style", STANDARD_STYLER_CONFIG["header_style"])
    styler = styler.set_table_styles([
        {"selector": "thead th", "props": list(header_style.items())}
    ])
    
    # Стили ячеек
    cell_style = style_config.get("cell_style", STANDARD_STYLER_CONFIG["cell_style"])
    styler = styler.set_table_styles([
        {"selector": "tbody td", "props": list(cell_style.items())}
    ])
    
    # Границы
    border = style_config.get("border", STANDARD_STYLER_CONFIG["border"])
    styler = styler.set_table_styles([
        {"selector": "table", "props": [("border", border)]}
    ])
    
    return styler

def _apply_conditional_formatting(styler, df: pd.DataFrame, style_config: Dict[str, Any]):
    """Применяет условное форматирование"""
    
    cell_rules = style_config.get("cell_rules", [])
    
    for rule in cell_rules:
        column = rule.get("column")
        value = rule.get("value")
        color = rule.get("color", "red")
        
        if column and column in df.columns:
            if value == "max":
                # Выделить максимальные значения
                styler = styler.apply(
                    lambda x: [f"background-color: {color}; color: white" 
                              if x[column] == x[column].max() else "" 
                              for _ in x], 
                    subset=[column]
                )
            elif value == "min":
                # Выделить минимальные значения
                styler = styler.apply(
                    lambda x: [f"background-color: {color}; color: white" 
                              if x[column] == x[column].min() else "" 
                              for _ in x], 
                    subset=[column]
                )
            elif isinstance(value, (int, float)):
                # Выделить конкретные значения
                styler = styler.apply(
                    lambda x: [f"background-color: {color}; color: white" 
                              if x[column] == value else "" 
                              for _ in x], 
                    subset=[column]
                )
    
    return styler

def _apply_cell_highlighting(styler, df: pd.DataFrame, style_config: Dict[str, Any]):
    """Применяет выделение конкретных ячеек"""
    
    highlighted_cells = style_config.get("highlighted_cells", [])
    
    for cell in highlighted_cells:
        row_idx = cell.get("row")
        col_name = cell.get("column")
        color = cell.get("color", "yellow")
        
        if col_name and col_name in df.columns and row_idx is not None:
            styler = styler.apply(
                lambda x: [f"background-color: {color}" 
                          if x.name == row_idx else "" 
                          for _ in x], 
                subset=[col_name]
            )
    
    return styler

def _apply_row_striping(styler, style_config: Dict[str, Any]):
    """Применяет чередование строк"""
    
    if style_config.get("striped", False):
        even_color = style_config.get("even_row_color", "#f9f9f9")
        odd_color = style_config.get("odd_row_color", "white")
        
        styler = styler.set_table_styles([
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", even_color)]},
            {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", odd_color)]}
        ])
    
    return styler

# ==================== ФУНКЦИИ ДЛЯ AI ====================

def generate_styler_config(user_request: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Генерирует конфигурацию стилей на основе запроса пользователя.
    Упрощенная версия для AI.
    
    Args:
        user_request: Запрос пользователя
        df: DataFrame для анализа
        
    Returns:
        Конфигурация стилей
    """
    config = {
        "header_style": STANDARD_STYLER_CONFIG["header_style"].copy(),
        "cell_style": STANDARD_STYLER_CONFIG["cell_style"].copy(),
        "striped": False,
        "cell_rules": [],
        "highlighted_cells": []
    }
    
    # Простой анализ запроса
    user_request_lower = user_request.lower()
    
    # Чередование строк
    if any(word in user_request_lower for word in ["чередование", "полосатые", "striped"]):
        config["striped"] = True
        config["even_row_color"] = "#f5f5f5"
        config["odd_row_color"] = "white"
    
    # Выделение максимума
    if "максимум" in user_request_lower or "max" in user_request_lower:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                config["cell_rules"].append({
                    "column": col,
                    "value": "max",
                    "color": "red"
                })
    
    # Выделение минимума
    if "минимум" in user_request_lower or "min" in user_request_lower:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                config["cell_rules"].append({
                    "column": col,
                    "value": "min",
                    "color": "green"
                })
    
    # Выделение конкретных значений
    if "красн" in user_request_lower:
        config["header_style"]["background-color"] = "red"
        config["header_style"]["color"] = "white"
    
    return config

def create_table_code_template(df: pd.DataFrame, style_config: Dict[str, Any]) -> str:
    """
    Создает шаблон кода для AI генерации.
    
    Args:
        df: DataFrame
        style_config: Конфигурация стилей
        
    Returns:
        Шаблон кода
    """
    template = f'''
# Streamlit + Pandas Styler таблица
import pandas as pd
import streamlit as st

# Данные
data = {df.to_dict('records')}
df = pd.DataFrame(data)

# Конфигурация стилей
style_config = {style_config}

# Создание стилизованной таблицы
styler = df.style

# Применение стилей
styler = apply_styler_styles(styler, df, style_config)

# Отображение
st.dataframe(styler, use_container_width=True)
'''
    return template

# ==================== ИНТЕГРАЦИЯ С STREAMLIT ====================

def render_styled_table(df: pd.DataFrame, style_config: Dict[str, Any], 
                       title: str = "", caption: str = ""):
    """
    Отображает стилизованную таблицу в Streamlit.
    
    Args:
        df: DataFrame
        style_config: Конфигурация стилей
        title: Заголовок таблицы
        caption: Подпись таблицы
    """
    if title:
        st.markdown(f"**{title}**")
    
    # Создаем стилизованную таблицу
    styled_df = create_styled_dataframe(df, style_config)
    
    # Отображаем в Streamlit
    st.dataframe(styled_df, use_container_width=True)
    
    if caption:
        st.caption(caption)

# ==================== УТИЛИТЫ ====================

def parse_style_request(user_request: str) -> Dict[str, Any]:
    """
    Парсит запрос пользователя и извлекает параметры стилизации.
    
    Args:
        user_request: Запрос пользователя
        
    Returns:
        Словарь с параметрами стилизации
    """
    config = {}
    
    # Поиск цветов
    color_patterns = {
        "красн": "red",
        "зелен": "green", 
        "син": "blue",
        "желт": "yellow"
    }
    
    for pattern, color in color_patterns.items():
        if pattern in user_request.lower():
            config["color"] = color
            break
    
    # Поиск типов форматирования
    if "максимум" in user_request.lower():
        config["highlight"] = "max"
    elif "минимум" in user_request.lower():
        config["highlight"] = "min"
    
    # Чередование строк
    if any(word in user_request.lower() for word in ["чередование", "полосатые"]):
        config["striped"] = True
    
    return config

def validate_style_config(config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Валидирует и исправляет конфигурацию стилей.
    
    Args:
        config: Конфигурация стилей
        df: DataFrame
        
    Returns:
        Исправленная конфигурация
    """
    # Проверяем существование колонок
    if "cell_rules" in config:
        valid_rules = []
        for rule in config["cell_rules"]:
            if rule.get("column") in df.columns:
                valid_rules.append(rule)
        config["cell_rules"] = valid_rules
    
    # Проверяем цвета
    valid_colors = ["red", "green", "blue", "yellow", "orange", "purple"]
    if "color" in config and config["color"] not in valid_colors:
        config["color"] = "red"  # По умолчанию
    
    return config

# ==================== ЭКСПОРТ ФУНКЦИЙ ====================

__all__ = [
    "create_styled_dataframe",
    "generate_styler_config", 
    "create_table_code_template",
    "render_styled_table",
    "parse_style_request",
    "validate_style_config",
    "STANDARD_STYLER_CONFIG"
]
