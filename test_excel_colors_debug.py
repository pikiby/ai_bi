#!/usr/bin/env python3
"""
Детальный тест экспорта цветов в Excel
Создает несколько файлов с разными методами применения стилей
"""
import pandas as pd
import sys
import io

print("=" * 60)
print("ТЕСТ ЭКСПОРТА СТИЛЕЙ PANDAS В EXCEL")
print("=" * 60)

# Проверка версий
print(f"\npandas версия: {pd.__version__}")

try:
    import openpyxl
    print(f"openpyxl версия: {openpyxl.__version__}")
except ImportError:
    print("❌ openpyxl НЕ установлен! Установите: pip install openpyxl")
    sys.exit(1)

# Тестовые данные
df = pd.DataFrame({
    'Город': ['Москва', 'Санкт-Петербург', 'Новосибирск'],
    'Население': [12000000, 5400000, 1600000],
    'Площадь': [2500, 1400, 500]
})

print(f"\nТестовый DataFrame:\n{df}\n")

# ============================================================
# ТЕСТ 1: HEX цвет через applymap
# ============================================================
print("ТЕСТ 1: Светло-синий через HEX (#ADD8E6)")
print("-" * 60)

styler1 = df.style.applymap(lambda x: 'background-color: #ADD8E6')

# Проверим что внутри styler
ctx1 = styler1._compute()
print(f"Количество строк стилей: {len(ctx1.ctx) if hasattr(ctx1, 'ctx') else 'N/A'}")
if hasattr(ctx1, 'ctx') and ctx1.ctx:
    first_row_styles = ctx1.ctx[0] if ctx1.ctx else []
    print(f"Стили первой строки: {first_row_styles[:2] if first_row_styles else 'пусто'}")

try:
    buf1 = io.BytesIO()
    with pd.ExcelWriter(buf1, engine='openpyxl') as writer:
        styler1.to_excel(writer, sheet_name='Тест1_HEX', index=False)
    
    with open('test1_hex_applymap.xlsx', 'wb') as f:
        f.write(buf1.getvalue())
    print(f"✅ УСПЕХ! Файл: test1_hex_applymap.xlsx ({len(buf1.getvalue())} байт)")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")

# ============================================================
# ТЕСТ 2: RGBA цвет через applymap (должен дать черный/неправильный)
# ============================================================
print("\nТЕСТ 2: Светло-синий через RGBA")
print("-" * 60)

styler2 = df.style.applymap(lambda x: 'background-color: rgba(173, 216, 230, 0.5)')

ctx2 = styler2._compute()
if hasattr(ctx2, 'ctx') and ctx2.ctx:
    first_row_styles = ctx2.ctx[0] if ctx2.ctx else []
    print(f"Стили первой строки: {first_row_styles[:2] if first_row_styles else 'пусто'}")

try:
    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine='openpyxl') as writer:
        styler2.to_excel(writer, sheet_name='Тест2_RGBA', index=False)
    
    with open('test2_rgba_applymap.xlsx', 'wb') as f:
        f.write(buf2.getvalue())
    print(f"✅ Экспорт выполнен: test2_rgba_applymap.xlsx")
    print(f"⚠️  ВНИМАНИЕ: цвет может быть ЧЕРНЫМ или неправильным!")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")

# ============================================================
# ТЕСТ 3: RGB цвет через applymap
# ============================================================
print("\nТЕСТ 3: Светло-синий через RGB (без альфа-канала)")
print("-" * 60)

styler3 = df.style.applymap(lambda x: 'background-color: rgb(173, 216, 230)')

try:
    buf3 = io.BytesIO()
    with pd.ExcelWriter(buf3, engine='openpyxl') as writer:
        styler3.to_excel(writer, sheet_name='Тест3_RGB', index=False)
    
    with open('test3_rgb_applymap.xlsx', 'wb') as f:
        f.write(buf3.getvalue())
    print(f"✅ Экспорт выполнен: test3_rgb_applymap.xlsx")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")

# ============================================================
# ТЕСТ 4: set_properties (альтернативный метод)
# ============================================================
print("\nТЕСТ 4: HEX через set_properties (на весь DataFrame)")
print("-" * 60)

styler4 = df.style.set_properties(**{'background-color': '#ADD8E6'})

try:
    buf4 = io.BytesIO()
    with pd.ExcelWriter(buf4, engine='openpyxl') as writer:
        styler4.to_excel(writer, sheet_name='Тест4_Props', index=False)
    
    with open('test4_hex_properties.xlsx', 'wb') as f:
        f.write(buf4.getvalue())
    print(f"✅ УСПЕХ: test4_hex_properties.xlsx")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")

# ============================================================
# ТЕСТ 5: Комбинированные стили (как в реальном коде)
# ============================================================
print("\nТЕСТ 5: Комбинация стилей (цвет фона + цвет текста)")
print("-" * 60)

styler5 = df.style.applymap(
    lambda x: 'background-color: #ADD8E6; color: #000000'
)

try:
    buf5 = io.BytesIO()
    with pd.ExcelWriter(buf5, engine='openpyxl') as writer:
        styler5.to_excel(writer, sheet_name='Тест5_Combo', index=False)
    
    with open('test5_combo_styles.xlsx', 'wb') as f:
        f.write(buf5.getvalue())
    print(f"✅ УСПЕХ: test5_combo_styles.xlsx")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")

print("\n" + "=" * 60)
print("ТЕСТЫ ЗАВЕРШЕНЫ!")
print("=" * 60)
print("\nСозданные файлы:")
print("  1. test1_hex_applymap.xlsx      - HEX цвет (должен работать)")
print("  2. test2_rgba_applymap.xlsx     - RGBA (может быть черным!)")
print("  3. test3_rgb_applymap.xlsx      - RGB без альфа")
print("  4. test4_hex_properties.xlsx    - через set_properties")
print("  5. test5_combo_styles.xlsx      - комбинация стилей")
print("\nОткрой файлы в Excel и проверь цвета!")
print("Правильный цвет: СВЕТЛО-СИНИЙ")
print("Если видишь ЧЕРНЫЙ - значит формат цвета не поддерживается")

