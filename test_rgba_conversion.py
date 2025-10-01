#!/usr/bin/env python3
"""
Воспроизведение проблемы с rgba → hex конвертацией
Тестируем то же что делает приложение
"""
import pandas as pd
import io
import sys

print("=" * 60)
print("ТЕСТ КОНВЕРТАЦИИ RGBA В EXCEL")
print("=" * 60)

# Проверка библиотек
try:
    import openpyxl
    print(f"✓ pandas: {pd.__version__}")
    print(f"✓ openpyxl: {openpyxl.__version__}")
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    sys.exit(1)

# Тестовые данные (как в приложении)
df = pd.DataFrame({
    'Город': ['Москва', 'Питер', 'Новосибирск'],
    'Население': [12000000, 5400000, 1600000]
})

print(f"\nDataFrame:\n{df}\n")

# ========================================
# ТЕСТ 1: Прямое применение rgba (как делает LLM)
# ========================================
print("ТЕСТ 1: rgba цвет (как генерирует LLM)")
print("-" * 60)

styled_df = df.style.set_properties(
    subset=[df.columns[0]], 
    **{'background-color': 'rgba(144,238,144,0.3)'}
)

# Проверяем _todo
if hasattr(styled_df, '_todo'):
    print(f"_todo операций: {len(styled_df._todo)}")
    for i, (func, args, kwargs) in enumerate(styled_df._todo):
        print(f"  {i}: {func.__name__}, kwargs: {kwargs}")

# Вычисляем стили
ctx = styled_df._compute()
if hasattr(ctx, 'ctx') and ctx.ctx:
    print(f"\nПервая строка стилей:")
    for i, style in enumerate(ctx.ctx[0][:2]):
        print(f"  Столбец {i}: {repr(style)}")

# Экспортируем в Excel
try:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        styled_df.to_excel(writer, sheet_name='Test', index=False)
    
    with open('test_rgba_direct.xlsx', 'wb') as f:
        f.write(buf.getvalue())
    print(f"\n✓ Файл сохранен: test_rgba_direct.xlsx ({len(buf.getvalue())} байт)")
    print("  ⚠️  Проверь - цвет должен быть ЧЕРНЫМ!")
except Exception as e:
    print(f"\n✗ Ошибка: {e}")

# ========================================
# ТЕСТ 2: С конвертацией rgba → hex
# ========================================
print("\n" + "=" * 60)
print("ТЕСТ 2: rgba → hex конвертация")
print("-" * 60)

styled_df2 = df.style.set_properties(
    subset=[df.columns[0]], 
    **{'background-color': 'rgba(144,238,144,0.3)'}
)

# Применяем конвертацию (как в приложении)
import re

def convert_color_value(value):
    if not isinstance(value, str):
        return value
    
    match = re.search(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)', value)
    if match:
        r, g, b = map(int, match.groups())
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        result = re.sub(r'rgba?\([^)]+\)', hex_color, value)
        print(f"Конвертация: {value} → {result}")
        return result
    return value

# Пересоздаем styler
new_styler = df.style

if hasattr(styled_df2, '_todo'):
    print(f"Обработка {len(styled_df2._todo)} операций...")
    
    for func, args, kwargs in styled_df2._todo:
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                new_dict = {k: convert_color_value(v) for k, v in value.items()}
                new_kwargs[key] = new_dict
            else:
                new_kwargs[key] = convert_color_value(value)
        
        result = func(new_styler, *args, **new_kwargs)
        if result is not None:
            new_styler = result

# Проверяем результат
ctx2 = new_styler._compute()
if hasattr(ctx2, 'ctx') and ctx2.ctx:
    print(f"\nПервая строка стилей ПОСЛЕ конвертации:")
    for i, style in enumerate(ctx2.ctx[0][:2]):
        print(f"  Столбец {i}: {repr(style)}")

# Экспортируем
try:
    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine='openpyxl') as writer:
        new_styler.to_excel(writer, sheet_name='Test', index=False)
    
    with open('test_rgba_converted.xlsx', 'wb') as f:
        f.write(buf2.getvalue())
    print(f"\n✓ Файл сохранен: test_rgba_converted.xlsx ({len(buf2.getvalue())} байт)")
    print("  ✓ Проверь - цвет должен быть СВЕТЛО-ЗЕЛЕНЫМ!")
except Exception as e:
    print(f"\n✗ Ошибка: {e}")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print("\nФайлы для проверки:")
print("  1. test_rgba_direct.xlsx    - БЕЗ конвертации (черный)")
print("  2. test_rgba_converted.xlsx - С конвертацией (зеленый)")
print("\nОткрой оба файла и сравни цвета!")

