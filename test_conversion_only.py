#!/usr/bin/env python3
"""
Проверка ТОЛЬКО логики конвертации (без openpyxl)
"""
import pandas as pd
import re

print("=" * 60)
print("ТЕСТ ЛОГИКИ КОНВЕРТАЦИИ RGBA → HEX")
print("=" * 60)

# Тестовые данные
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

print(f"\nDataFrame:\n{df}\n")

# ========================================
# Создаем styled_df как делает LLM
# ========================================
print("Создаем styled_df с rgba (как LLM):")
styled_df = df.style.set_properties(**{'background-color': 'rgba(173, 216, 230, 0.3)'})

# Проверяем _todo
print(f"\n_todo содержит: {len(styled_df._todo)} операций")
for i, (func, args, kwargs) in enumerate(styled_df._todo):
    print(f"  Операция {i}:")
    print(f"    func: {func.__name__}")
    print(f"    args: {args}")
    print(f"    kwargs: {kwargs}")

# ========================================
# ФУНКЦИЯ КОНВЕРТАЦИИ (из app.py)
# ========================================
def convert_color_value(value):
    """Конвертирует rgba/rgb в hex"""
    if not isinstance(value, str):
        return value
    
    match = re.search(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)', value)
    if match:
        r, g, b = map(int, match.groups())
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        result = re.sub(r'rgba?\([^)]+\)', hex_color, value)
        print(f"    [КОНВЕРТАЦИЯ] {value} → {result}")
        return result
    
    return value

# ========================================
# Применяем конвертацию
# ========================================
print("\nПрименяем конвертацию:")
new_styler = df.style

for func, args, kwargs in styled_df._todo:
    print(f"\nОбработка операции: {func.__name__}")
    print(f"  kwargs ДО: {kwargs}")
    
    new_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, dict):
            # Словарь стилей
            new_dict = {k: convert_color_value(v) for k, v in value.items()}
            new_kwargs[key] = new_dict
        else:
            new_kwargs[key] = convert_color_value(value)
    
    print(f"  kwargs ПОСЛЕ: {new_kwargs}")
    
    # Применяем операцию
    result = func(new_styler, *args, **new_kwargs)
    if result is not None:
        new_styler = result

# ========================================
# Проверяем результат
# ========================================
print("\n" + "=" * 60)
print("ПРОВЕРКА РЕЗУЛЬТАТА:")
print("=" * 60)

# Вычисляем стили исходного styler
ctx_old = styled_df._compute()
print("\nСтили ИСХОДНОГО styler (с rgba):")
if hasattr(ctx_old, 'ctx') and ctx_old.ctx:
    for row_idx, row_styles in enumerate(ctx_old.ctx[:2]):  # первые 2 строки
        print(f"  Строка {row_idx}:")
        for col_idx, style in enumerate(row_styles):
            if style:
                print(f"    Столбец {col_idx}: {style}")

# Вычисляем стили нового styler
ctx_new = new_styler._compute()
print("\nСтили НОВОГО styler (с hex):")
if hasattr(ctx_new, 'ctx') and ctx_new.ctx:
    for row_idx, row_styles in enumerate(ctx_new.ctx[:2]):  # первые 2 строки
        print(f"  Строка {row_idx}:")
        for col_idx, style in enumerate(row_styles):
            if style:
                print(f"    Столбец {col_idx}: {style}")

# ========================================
# ВЫВОД
# ========================================
print("\n" + "=" * 60)
print("ИТОГ:")
print("=" * 60)

has_rgba_old = any('rgba(' in str(style) for row in ctx_old.ctx for style in row)
has_rgba_new = any('rgba(' in str(style) for row in ctx_new.ctx for style in row)
has_hex_new = any('#' in str(style) for row in ctx_new.ctx for style in row)

print(f"\nИсходный styler содержит rgba: {has_rgba_old}")
print(f"Новый styler содержит rgba: {has_rgba_new}")
print(f"Новый styler содержит hex: {has_hex_new}")

if not has_rgba_new and has_hex_new:
    print("\n✅ КОНВЕРТАЦИЯ РАБОТАЕТ!")
    print("   rgba успешно заменен на hex")
else:
    print("\n❌ КОНВЕРТАЦИЯ НЕ РАБОТАЕТ!")
    print("   rgba не заменился на hex")

