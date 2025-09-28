# def_render_download_buttons

## Описание функции

Функция `_render_download_buttons()` отрисовывает кнопки скачивания для таблиц и графиков.

## Сигнатура функции

```python
def _render_download_buttons(data, item: dict, data_type: str):
```

## Параметры

- `data` - Данные для скачивания (DataFrame или Figure)
- `item: dict` - Элемент результата
- `data_type: str` - Тип данных ("table" или "chart")

## Возвращаемое значение

Функция не возвращает значения (void).

## Алгоритм работы

1. Определение типа данных
2. Создание кнопок для таблиц (CSV, XLSX)
3. Создание кнопок для графиков (HTML)
4. Настройка уникальных ключей

## Используемые функции

- `_df_to_csv_bytes()` - конвертация в CSV
- `_df_to_xlsx_bytes()` - конвертация в XLSX
- `fig.to_html()` - конвертация графика в HTML
