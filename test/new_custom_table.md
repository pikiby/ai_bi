# Упрощение файла для тестирования таблиц
_Exported on 9/29/2025 at 12:15:08 GMT-3 from Cursor (1.6.45)_

---

**User**

Общая задача. Я тестирую работу визуализации таблиц имитируя запросы пользователя
@generate_test_data.py @test_table_styling.html 

Сделай файл @generate_test_data.py максимально простым. Чтобы он выводил только таблицу 

---

**Cursor**

Предлагаю упростить `test/generate_test_data.py` так, чтобы скрипт:

- генерировал DataFrame (можно оставить текущую функцию генерации, но убрать лишние стилизации и тесты),
- в `main` просто выводил таблицу в консоль (например, через `print(df)` или `print(df.to_string(index=False))`),
- удалил всё лишнее: тестовые вызовы Styler, текстовые сообщения, список стилей и т.д.

Подтвердите, пожалуйста, что именно такое упрощение и вывод в консоль вам подходят, и я внесу изменения.

---

**User**

да давай попробуем

---

**Cursor**

```python
*** Begin Patch
*** Delete File: /home/boris/Documents/Work/analytics/ai_bi/test/generate_test_data.py
*** End Patch
```

```python
*** Begin Patch
*** Delete File: /home/boris/Documents/Work/analytics/ai_bi/test/generate_test_data.py
*** End Patch
```

```python
#!/usr/bin/env python3
"""Простой скрипт для генерации и вывода тестовой таблицы."""

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


if __name__ == "__main__":
    df = generate_test_data()
    print(df.to_string(index=False))

```

Упростил `test/generate_test_data.py`: оставил только генерацию фиксированного набора строк и вывод таблицы в консоль без какого‑либо Styler и вспомогательных функций. Логика теперь минимальная — `generate_test_data()` возвращает готовый `DataFrame`, а в `main` он печатается через `to_string` без индекса.

```10:32:test/generate_test_data.py
    data = [
        {
            "Город": "Москва",
            "Партнер": "Партнер_001",
            "Активные пользователи": 1250,
            ...
```

Тесты не запускал; при необходимости можно проверить командой `python3 test/generate_test_data.py`.

```python
    data = [
        {
            "Город": "Москва",
            "Партнер": "Партнер_001",
            "Активные пользователи": 1250,
            ...
```

---

**User**

сделай так чтобы созданная таблица выводилась  В HTML чтобы я понял как она выглядит

---

**Cursor**

```python
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


if __name__ == "__main__":
    df = generate_test_data()
    print(df.to_string(index=False))

    html_path = Path(__file__).with_name("generated_test_table.html")
    html_path.write_text(df.to_html(index=False), encoding="utf-8")
    print(f"\nHTML-версия сохранена: {html_path}")

```

Готово: теперь `test/generate_test_data.py` не только печатает DataFrame в консоль, но и сохраняет его в `test/generated_test_table.html` (UTF-8, без индекса) с сообщением о пути. При запуске `python3 test/generate_test_data.py` увидите таблицу и ссылку на файл.

---

**User**

сделай базувую стилизацию как тут@test_table_styling.html 

---

**Cursor**

```python
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

```

Добавил базовую стилизацию, аналогичную `test/test_table_styling.html`: теперь `render_html_table()` возвращает полноценную HTML-страницу с контейнером, шрифтами, отступами и таблицей с полосатой подсветкой строк и hover-эффектом. Файл `test/generated_test_table.html` при запуске `python3 test/generate_test_data.py` создаётся/перезаписывается уже с этими стилями.

---

**User**

Отлично теперь пользователь пишет что хочет сделат  таблицу сине-белой. сделай изменения в @generate_test_data.py  и перезапусти этот файл

---

**Cursor**

```python
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
            background-color: #1e3a8a;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #1e3a8a;
            font-weight: bold;
        }}

        .test-table td {{
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
            text-align: left;
        }}

        .test-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        .test-table tr:nth-child(odd) {{
            background-color: white;
        }}

        .test-table tr:hover {{
            background-color: #dbeafe;
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

```

Готово! Изменил стилизацию на сине-белую:

- **Заголовки**: тёмно-синий фон (#1e3a8a) с белым текстом
- **Строки**: чередование белого и светло-серого (#f8fafc) 
- **Hover**: светло-голубой (#dbeafe)
- **Границы**: синие для заголовков, серые для ячеек

Скрипт перезапущен, HTML-файл обновлён с новой сине-белой стилизацией.

---

**User**

Пользователь пишет выдели максималльные значения в каждом блоке красным

---

**Cursor**

```python
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

    # Находим максимальные значения в числовых колонках
    styled_df = df.copy()
    
    # Выделяем максимальные значения красным
    for col in styled_df.columns:
        if col not in ['Город', 'Партнер']:  # Только числовые колонки
            max_val = styled_df[col].max()
            styled_df[col] = styled_df[col].apply(
                lambda x: f'<span style="color: red; font-weight: bold;">{x}</span>' if x == max_val else str(x)
            )

    table_html = styled_df.to_html(index=False, classes="test-table", escape=False)

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
            background-color: #1e3a8a;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #1e3a8a;
            font-weight: bold;
        }}

        .test-table td {{
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
            text-align: left;
        }}

        .test-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        .test-table tr:nth-child(odd) {{
            background-color: white;
        }}

        .test-table tr:hover {{
            background-color: #dbeafe;
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

```

Готово! Добавил выделение максимальных значений красным цветом:

- **Алгоритм**: для каждой числовой колонки (кроме "Город" и "Партнер") находится максимальное значение
- **Выделение**: максимальные значения оборачиваются в `<span>` с красным цветом и жирным шрифтом
- **Результат**: в HTML-файле максимальные значения в каждой колонке будут выделены красным

Скрипт перезапущен, HTML-файл обновлён с выделением максимумов.

---

**User**

Пользователь пишет

Сделай таблицу по умолчанию

---

**Cursor**

```python
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
            background-color: #1e3a8a;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #1e3a8a;
            font-weight: bold;
        }}

        .test-table td {{
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
            text-align: left;
        }}

        .test-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        .test-table tr:nth-child(odd) {{
            background-color: white;
        }}

        .test-table tr:hover {{
            background-color: #dbeafe;
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

```

```python
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

```

Готово! Вернул таблицу к стандартному виду по умолчанию:

- **Убрал** выделение максимальных значений красным
- **Вернул** базовую стилизацию: серые заголовки, чередующиеся строки, стандартные границы
- **Убрал** сине-белую цветовую схему

Теперь таблица имеет классический вид без специальных выделений.

