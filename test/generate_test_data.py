#!/usr/bin/env python3
"""Простой скрипт для генерации и вывода тестовой таблицы."""

from pathlib import Path

import pandas as pd


def generate_test_data() -> pd.DataFrame:
    """Возвращает DataFrame с фиксированными тестовыми данными."""

    data = [
        {
            "Город": "Москва",
            "Партнер": "Партнер_001",
            "Активные пользователи": 1250,
            "Подписки Android": 45,
            "Подписки iOS": 32,
            "Доход Android (₽)": 2350,
            "Доход iOS (₽)": 1890,
            "Конверсия (%)": 6.2,
        },
        {
            "Город": "Санкт-Петербург",
            "Партнер": "Партнер_002",
            "Активные пользователи": 890,
            "Подписки Android": 28,
            "Подписки iOS": 19,
            "Доход Android (₽)": 1680,
            "Доход iOS (₽)": 1120,
            "Конверсия (%)": 5.3,
        },
        {
            "Город": "Новосибирск",
            "Партнер": "Партнер_003",
            "Активные пользователи": 420,
            "Подписки Android": 15,
            "Подписки iOS": 12,
            "Доход Android (₽)": 750,
            "Доход iOS (₽)": 680,
            "Конверсия (%)": 6.4,
        },
        {
            "Город": "Екатеринбург",
            "Партнер": "Партнер_004",
            "Активные пользователи": 380,
            "Подписки Android": 12,
            "Подписки iOS": 8,
            "Доход Android (₽)": 580,
            "Доход iOS (₽)": 420,
            "Конверсия (%)": 5.3,
        },
        {
            "Город": "Казань",
            "Партнер": "Партнер_005",
            "Активные пользователи": 290,
            "Подписки Android": 9,
            "Подписки iOS": 6,
            "Доход Android (₽)": 420,
            "Доход iOS (₽)": 350,
            "Конверсия (%)": 5.2,
        },
    ]

    return pd.DataFrame(data)


def render_html_table(df: pd.DataFrame) -> str:
    """Формирует HTML-страницу с базовой стилизацией таблицы."""

    table_html = df.to_html(index=False, classes="test-table")

    return f"""<!DOCTYPE html>
<html lang=\"ru\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Тестовая таблица</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 24px;
        }}

        .test-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0;
        }}

        .test-table th {{
            background-color: #f0f0f0;
            color: #333;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
            font-weight: bold;
        }}

        .test-table td {{
            padding: 10px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}

        .test-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .test-table tr:hover {{
            background-color: #f0f8ff;
        }}
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Тестовая таблица данных</h1>
        {table_html}
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    df = generate_test_data()
    print(df.to_string(index=False))

    html_path = Path(__file__).with_name("generated_test_table.html")
    html_path.write_text(render_html_table(df), encoding="utf-8")
    print(f"\nHTML-версия сохранена: {html_path}")
