#!/usr/bin/env python3
"""
Тест экспорта стилей pandas в Excel
Проверяем корректность цветов
"""
import pandas as pd
import io

# Создаем тестовый DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

print("pandas version:", pd.__version__)

try:
    import openpyxl
    print("openpyxl version:", openpyxl.__version__)
except ImportError:
    print("openpyxl NOT installed!")
    exit(1)

# Тест 1: Простой светло-синий фон в HEX формате
print("\n=== Тест 1: HEX цвет (#ADD8E6) ===")
styler1 = df.style.applymap(lambda x: 'background-color: #ADD8E6')

buf1 = io.BytesIO()
try:
    with pd.ExcelWriter(buf1, engine='openpyxl') as writer:
        styler1.to_excel(writer, sheet_name='Test1', index=False)
    print("✓ Экспорт успешен")
    with open('test_hex_color.xlsx', 'wb') as f:
        f.write(buf1.getvalue())
    print("✓ Файл сохранен: test_hex_color.xlsx")
except Exception as e:
    print(f"✗ Ошибка: {e}")

# Тест 2: RGBA цвет (должен НЕ работать)
print("\n=== Тест 2: RGBA цвет (rgba(173,216,230,0.5)) ===")
styler2 = df.style.applymap(lambda x: 'background-color: rgba(173,216,230,0.5)')

buf2 = io.BytesIO()
try:
    with pd.ExcelWriter(buf2, engine='openpyxl') as writer:
        styler2.to_excel(writer, sheet_name='Test2', index=False)
    print("✓ Экспорт успешен (но цвет может быть черным!)")
    with open('test_rgba_color.xlsx', 'wb') as f:
        f.write(buf2.getvalue())
    print("✓ Файл сохранен: test_rgba_color.xlsx")
except Exception as e:
    print(f"✗ Ошибка: {e}")

# Тест 3: Проверка что именно передается в openpyxl
print("\n=== Тест 3: Проверка стилей Styler ===")
styler3 = df.style.applymap(lambda x: 'background-color: #ADD8E6; color: #000000')
# Вычисляем стили
ctx = styler3._compute()
print("Применённые стили (первая ячейка):")
print(ctx.ctx)

print("\n=== Готово ===")
print("Проверь файлы test_hex_color.xlsx и test_rgba_color.xlsx")

