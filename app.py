# app.py
# Полноценное приложение Streamlit: динамическая подгрузка промптов, роутер режимов,
# SQL (ClickHouse), RAG, безопасный Plotly, история результатов и экспорт.

import os
import re
import json
import io
import zipfile
from datetime import datetime
from textwrap import dedent

import numpy as np

import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio  # для to_image(fig, format="png")

from openai import OpenAI
from clickhouse_client import ClickHouse_client
import retriever

# >>> Горячая подгрузка prompts.py при каждом обращении
import importlib
import prompts

# --- Каталоги и БД по умолчанию (для режима catalog) ---
KB_DOCS_DIR = os.getenv("KB_DOCS_DIR", "kb_docs")     # папка с описаниями дашбордов (если есть)
DEFAULT_DB = os.getenv("CLICKHOUSE_DB", "db1")        # база ClickHouse для полного каталога таблиц


pio.templates.default = "plotly"

# ----------------------- Базовые настройки страницы -----------------------

# Из уважения к предпочтениям — без emoji в иконке
st.set_page_config(page_title="Ассистент аналитики")

# ----------------------- Константы окружения -----------------------

CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")   # было, вероятно: "./chroma"
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")  # было, вероятно: "kb_default"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
CATALOG_TABLES_FILE = os.getenv("KB_CATALOG_TABLES_FILE", os.path.join("docs", "kb_catalog_tables.md"))
CATALOG_DASHBOARDS_FILE = os.getenv("KB_CATALOG_DASHBOARDS_FILE", os.path.join("docs", "kb_catalog_dashboards.md"))
KB_T_AI_GLOBAL_REPORT_FILE = os.path.join("docs", "kb_t_ai_global_report.md")
SQL_SEMANTIC_GUARD = os.getenv("SQL_SEMANTIC_GUARD", "1") == "1"

# --- Авто-индексация при старте (однократно на процесс) ---
# Включается флагом окружения: KB_AUTO_INGEST_ON_START=1
@st.cache_resource  # гарантирует запуск ровно один раз на процесс Streamlit
def _auto_ingest_once():
    from ingest import run_ingest  # ленивый импорт, чтобы не тянуть зависимости зря
    return run_ingest()

# Автоматическая индексация базы знаний при запуске приложения
# Включается переменной окружения KB_AUTO_INGEST_ON_START=1
# Индексирует документы из папки docs/ в ChromaDB для RAG-поиска
if os.getenv("KB_AUTO_INGEST_ON_START", "0") == "1":
    try:
        stats = _auto_ingest_once()
        # Короткое уведомление в сайдбар без «воды»
        st.sidebar.success(
            f'Индекс обновлён: файлов {stats.get("files",0)}, '
            f'чанков {stats.get("chunks",0)}, добавлено {stats.get("added",0)}'
        )
    except Exception as e:
        st.sidebar.error(f"Автоиндексация не удалась: {e}")


# ----------------------- Клиент OpenAI -----------------------

# >>> клиент создаём один раз
client = OpenAI()

# ----------------------- Состояние приложения -----------------------
# st.session_state - встроенный механизм Streamlit для сохранения данных между взаимодействиями
# Без него все переменные сбрасываются при каждом клике/обновлении страницы
# Используем паттерн "ленивая инициализация" - создаем только при первом обращении

# история чата
if "messages" not in st.session_state:
    st.session_state["messages"] = []  

# история результатов (таблицы/графики)
if "results" not in st.session_state:
    st.session_state["results"] = []   

#  последний df (polars), для построения графиков
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

# Сохраняем последние метаданные SQL, чтобы всплывающие графики имели заголовок и SQL даже при ошибках
if "last_sql_meta" not in st.session_state:
    st.session_state["last_sql_meta"] = {}

# кэш последнего непустого RAG-контекста (используем как фолбэк)
if "last_rag_ctx" not in st.session_state:
    st.session_state["last_rag_ctx"] = ""

# кэш схемы БД для быстрого доступа
if "db_schema_cache" not in st.session_state:
    st.session_state["db_schema_cache"] = {}

# глобальные стили для новых таблиц
if "next_table_style" not in st.session_state:
    st.session_state["next_table_style"] = None

# ----------------------- Вспомогательные функции -----------------------

# КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ: Горячая перезагрузка системных промптов
# ЦЕЛЬ: Позволяет изменять LLM-поведение без перезапуска приложения
# МЕСТА ИСПОЛЬЗОВАНИЯ: При каждом запросе пользователя (строка 1035), проверка предупреждений (строка 1008)
# ВАЖНОСТЬ: Обеспечивает гибкость разработки, надежность через fallback-значения, отладку через предупреждения
def _reload_prompts():
    importlib.reload(prompts)
    warn = []

    def _get(name, default):
        if hasattr(prompts, name):
            return getattr(prompts, name)
        else:
            warn.append(name)
            return default

    p_map = {
        "router": _get(
            "ROUTER_PROMPT",
            "Ты — маршрутизатор. Верни ровно один блок ```mode\nsql\n``` где sql|rag|plotly|catalog."
        ),
        "sql": _get(
            "RULES_SQL",
            "Режим SQL. Верни лаконичный ответ с одним блоком ```sql ...``` и не добавляй ничего лишнего."
        ),
        "rag": _get(
            "RULES_RAG",
            "Режим RAG. Сначала верни блок ```rag <краткий_запрос>```, без пояснений."
        ),
        "plotly": _get(
            "RULES_PLOTLY",
            "Режим PLOTLY. Верни ровно один блок ```plotly``` с кодом, создающим переменную fig."
        ),
    }
    return p_map, warn


# КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ: Сохранение результатов в историю приложения
# ЦЕЛЬ: Обеспечивает персистентность данных между обновлениями страницы, возможность экспорта, контекст для LLM
# МЕСТА ИСПОЛЬЗОВАНИЯ: После SQL-запросов (строка 1251), после создания графиков (строка 1399)
# ВАЖНОСТЬ: Фундаментальная для сохранения результатов работы, экспорта данных, непрерывности UX
def _push_result(kind: str, df_pl: pl.DataFrame | None = None,
                 fig: go.Figure | None = None, meta: dict | None = None):
    st.session_state["results"].append({
        "kind": kind,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "df_pl": df_pl,     # polars.DataFrame (для таблицы/экспорта)
        "fig": fig,         # plotly.graph_objects.Figure (для графика/экспорта)
        "meta": meta or {},
        "msg_idx": st.session_state.get("last_assistant_idx"),
    })

# Нумерация таблиц для UI: определяет порядковый номер таблицы среди всех результатов
# ЗАЧЕМ: Создание заголовков "Таблица 1", "Таблица 2", уникальных ключей, каталогизация
# ГДЕ ИСПОЛЬЗУЕТСЯ: Отображение заголовков (строка 287), каталог таблиц (строка 222), уникальные ключи редакторов
def _table_number_for(item: dict) -> int:
    n = 0
    for it in st.session_state.get("results", []):
        if it.get("kind") == "table":
            n += 1
        if it is item:
            break
    return n


# КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ: Парсинг SQL-запросов для извлечения метаданных
# ЧТО ДЕЛАЕТ: Анализирует SQL-код и извлекает структурированную информацию (таблицы, колонки, период, лимит)
# ЗАЧЕМ НУЖНО: Автоматические подписи к таблицам, контекст для LLM, умные заголовки, каталогизация результатов
# ГДЕ ИСПОЛЬЗУЕТСЯ: 
#   - Подписи под таблицами (строка 301) - показывает источники и периоды
#   - Каталог таблиц для LLM (строка 227) - обеспечивает контекст для новых запросов  
#   - Умные заголовки (строка 277) - "Топ 10 по «city»" вместо "Результаты запроса"
#   - Контекст для графиков - LLM знает структуру данных для визуализации
# ВАЖНОСТЬ: Фундаментальная для понимания контекста, улучшения UX, автоматизации подписей
# БЕЗ НЕЁ: Приложение было бы "глухим" - не помнило бы контекст и не могло строить на предыдущих результатах
def _extract_sql_info(sql: str | None, pdf: pd.DataFrame | None) -> dict:
 
    if not sql:
        sql = ""
    info = {"tables": [], "columns": [], "period": None, "limit": None}

    # Таблицы из FROM и JOIN
    for pat in [r"\bFROM\s+([a-zA-Z0-9_.`\"]+)", r"\bJOIN\s+([a-zA-Z0-9_.`\"]+)"]:
        info["tables"] += [m.group(1).strip("`\"") for m in re.finditer(pat, sql, flags=re.IGNORECASE)]

    # Столбцы из SELECT, если удалось вычленить; иначе — первые 6 из датафрейма
    m_sel = re.search(r"\bSELECT\s+(.*?)\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    cols = []
    if m_sel:
        raw = m_sel.group(1)
        parts = [p.strip() for p in raw.split(",")]
        for p in parts:
            # Берём алиас после AS, иначе — хвостовое слово (напр. users.city → city)
            m_as = re.search(r"\bAS\s+([a-zA-Z0-9_`\"]+)\b", p, flags=re.IGNORECASE)
            if m_as:
                c = m_as.group(1).strip("`\"")
            else:
                c = re.split(r"\s+", p)[-1]
                c = c.split(".")[-1].strip("`\"")
            # отфильтруем служебные * и функции
            if c != "*" and re.match(r"^[a-zA-Z0-9_]+$", c or ""):
                cols.append(c)
    if not cols and isinstance(pdf, pd.DataFrame):
        cols = list(pdf.columns[:6])
    info["columns"] = cols[:10]  # верхняя граница на всякий случай

    # Период: BETWEEN '...' AND '...' или пары сравнений с датой
    m_bt = re.search(r"\bBETWEEN\s*'([^']+)'\s*AND\s*'([^']+)'", sql, flags=re.IGNORECASE)
    if m_bt:
        info["period"] = f"{m_bt.group(1)} — {m_bt.group(2)}"
    else:
        dates = re.findall(r"'(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)'", sql)
        if len(dates) >= 2:
            info["period"] = f"{dates[0]} — {dates[1]}"

    # LIMIT n
    m_lim = re.search(r"\bLIMIT\s+(\d+)\b", sql, flags=re.IGNORECASE)
    if m_lim:
        info["limit"] = int(m_lim.group(1))

    return info

# КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ: Создание контекста для LLM из истории таблиц
# ПРИЧИНА: LLM не помнит предыдущие результаты, нужен контекст для преемственности диалога
# ЦЕЛЬ: Формирование "памяти" о всех ранее созданных таблицах для передачи LLM в качестве контекста
# ГДЕ ИСПОЛЬЗУЕТСЯ: 
#   - Перед роутером - LLM видит все таблицы и выбирает подходящий режим
#   - Перед SQL-генерацией - LLM понимает структуру предыдущих запросов
#   - Перед RAG-поиском - LLM знает, какие таблицы уже использовались
# ДЛЯ КАКИХ ЗАДАЧ: 
#   - Преемственность диалога: "Добавь к предыдущей таблице фильтр по Москве"
#   - Ссылки на таблицы: "Построй график по Таблице 2"
#   - Логичные продолжения: "Покажи детализацию" (LLM понимает контекст)
#   - Контекстные предложения: LLM может предлагать логичные продолжения
# ВАЖНОСТЬ: Фундаментальная для обеспечения контекста LLM, улучшения качества ответов, поддержки ссылок
# БЕЗ НЕЁ: LLM был бы "глухим" - не помнил бы контекст и не мог строить на предыдущих результатах
def _tables_index_hint() -> str:
    lines = []
    for item in st.session_state.get("results", []):
        if item.get("kind") != "table":
            continue
        n = _table_number_for(item)
        meta = item.get("meta") or {}
        title = (meta.get("title") or "").strip()
        explain = (meta.get("explain") or "").strip()
        sql = (meta.get("sql") or "").strip()
        # Скармливаем модели только самое нужное: название/пояснение и выжимку о периоде/источниках
        info = _extract_sql_info(sql, None)
        src = ", ".join(info.get("tables") or [])
        period = info.get("period") or "период не указан"
        head = f"Таблица {n}"
        if title:
            head += f": {title}"
        lines.append(f"{head}\nИсточник(и): {src or '—'}; Период: {period}\nSQL:\n{sql}\n")
    if not lines:
        return "Ранее таблицы не создавались."
    return "Справка по ранее созданным таблицам:\n\n" + "\n".join(lines)

# Очистка ответов LLM: удаляет служебные блоки (title, sql, plotly) из чата
# ЦЕЛЬ: Показать пользователю только понятный контент, скрыть технические детали
# ЗАДАЧИ: Разделение ответственности (служебные блоки обрабатываются отдельно), улучшение UX
# ГДЕ ИСПОЛЬЗУЕТСЯ: Отображение ответов ассистента в чате, обработка разных режимов (SQL/RAG/Plotly)
def _strip_llm_blocks(text: str) -> str:
    if not text:
        return text
    for tag in ("title", "explain", "sql", "rag", "python", "plotly", "table", "table_style"):
        text = re.sub(
            rf"```{tag}\s*.*?```",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ: Главная функция отрисовки результатов (рефакторинг)
# ПРИЧИНА РЕФАКТОРИНГА: Оригинальная функция была слишком большой (200+ строк) и нарушала принцип единственной ответственности
# ЦЕЛЬ: Роутинг отрисовки результатов к специализированным функциям с сохранением интерфейса
# ЗАДАЧИ: Улучшение читаемости, упрощение тестирования, переиспользование кода, упрощение поддержки
# ГДЕ ИСПОЛЬЗУЕТСЯ: Отображение новых результатов (сразу после создания), отображение истории (при загрузке страницы)
# ВАЖНОСТЬ: Фундаментальная для отображения результатов в интерфейсе, обеспечения персистентности данных
# БЕЗ НЕЁ: Результаты не отображались бы в интерфейсе пользователя
def _render_result(item: dict):
    """Главная функция отрисовки результатов - роутер к специализированным функциям"""
    kind = item.get("kind")
    
    if kind == "table":
        _render_table(item)
    elif kind == "chart":
        _render_chart(item)
    else:
        st.warning(f"Неизвестный тип результата: {kind}")



# ======================== Вспомогательные функции для _render_result ========================

# Отрисовка таблиц: координирует полную отрисовку таблиц от заголовка до кнопок скачивания
# АЛГОРИТМ: Валидация данных → Подготовка → Заголовок → Содержимое → Подпись → SQL → Скачивание
# ИСПОЛЬЗУЕТ: _get_title(), _render_table_content(), _render_table_caption(), _render_sql_block(), _render_download_buttons()
# ОБРАБОТКА ОШИБОК: Graceful degradation при некорректных данных, безопасная обработка отсутствующих метаданных
def _render_table(item: dict):
    df_pl = item.get("df_pl")
    if not isinstance(df_pl, pl.DataFrame):
        return
    
    pdf = df_pl.to_pandas()
    n = _table_number_for(item)
    meta = item.get("meta") or {}
    
    title = _get_title(meta, pdf, "sql")
    st.markdown(f"### Таблица {n}: {title}")
    
    _render_table_content(pdf, meta)
    _render_table_caption(meta, pdf)
    _render_sql_block(meta)
    _render_download_buttons(pdf, item, "table")


# Отрисовка графиков: координирует полную отрисовку Plotly-графиков с интерактивностью и экспортом
# АЛГОРИТМ: Валидация данных → Подготовка → Заголовок → График → Подпись → SQL → Код Plotly → Скачивание
# ИСПОЛЬЗУЕТ: _get_title(), _render_chart_caption(), _render_sql_block(), _render_plotly_code(), _render_download_buttons()
# ОСОБЕННОСТИ: Интерактивные графики с PNG-экспортом, fallback на контекст SQL, двойная документация (SQL + Plotly)
def _render_chart(item: dict):
    fig = item.get("fig")
    if not isinstance(fig, go.Figure):
        return
    
    meta = item.get("meta") or {}
    
    title = _get_title(meta, fallback_source="context")
    st.markdown(f"### {title}")
    
    st.plotly_chart(
        fig,
        theme=None,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 2}
        },
    )
    
    _render_chart_caption(meta)
    _render_sql_block(meta)
    _render_plotly_code(meta)
    _render_download_buttons(fig, item, "chart")


# УНИФИЦИРОВАННАЯ ФУНКЦИЯ: Получение заголовков для таблиц и графиков с умным fallback
# ЦЕЛЬ: Унификация логики заголовков, устранение дублирования кода, упрощение поддержки
# АЛГОРИТМ: Явный заголовок → Умный анализ SQL → Fallback на контекст → Дефолт
# ИСПОЛЬЗУЕТ: _extract_sql_info() для анализа SQL, st.session_state для контекста
# ОСОБЕННОСТИ: Поддерживает таблицы (с анализом данных) и графики (с контекстным fallback)
def _get_title(meta: dict, pdf: pd.DataFrame = None, fallback_source: str = "sql") -> str:
    # Проверяем явный заголовок
    title = (meta.get("title") or "").strip()
    if title:
        return title
    
    # Fallback 1: Умный анализ SQL (для таблиц)
    if fallback_source == "sql" and pdf is not None:
        sql = (meta.get("sql") or "").strip()
        if sql:
            info = _extract_sql_info(sql, pdf)
            
            if info.get("limit"):
                lead = None
                for c in info["columns"]:
                    if re.search(r"(city|город|category|катег|product|товар|region|регион|name|назв)", c, flags=re.IGNORECASE):
                        lead = c
                        break
                return f'Топ {info["limit"]}' + (f" по «{lead}»" if lead else "")
    
    # Fallback 2: Контекстный fallback (для графиков)
    if fallback_source == "context":
        title = st.session_state.get("last_sql_meta", {}).get("title", "").strip()
        if title:
            return title
    
    # Дефолтный заголовок
    return "Результаты запроса"


# ======================== ТАБЛИЧНЫЙ РЕНДЕР ========================

_TABLE_BASE_CSS = dedent(
    """
    <style>
        .ai-table-card {
            max-width: 1200px;
            margin: 0 auto 32px auto;
            background: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        }

        .ai-table-card .ai-table {
            width: 100%;
            border-collapse: collapse;
            margin: 0;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
        }

        .ai-table-card .ai-table th {
            background-color: #f0f0f0;
            color: #333333;
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #d1d5db;
            font-weight: 600;
        }

        .ai-table-card .ai-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e5e7eb;
            border-right: 1px solid #f3f4f6;
            text-align: left;
            vertical-align: top;
        }

        .ai-table-card .ai-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .ai-table-card .ai-table tr:hover {
            background-color: #f0f8ff;
        }

        .ai-table-card .ai-table tr:last-child td {
            border-bottom: none;
        }

        .ai-table-card .ai-table td:last-child,
        .ai-table-card .ai-table th:last-child {
            border-right: none;
        }
    </style>
    """
).strip()


def _compose_table_css(style_meta: dict | None) -> str:
    """Формирует CSS-блок с учётом пользовательских настроек."""

    css_parts = [_TABLE_BASE_CSS]
    style_meta = style_meta or {}

    header_bg = style_meta.get("header_fill_color")
    cells_bg = style_meta.get("cells_fill_color")
    align = (style_meta.get("align") or "left").lower()
    custom_css = style_meta.get("custom_css")

    if header_bg:
        css_parts.append(
            f"<style>.ai-table-card .ai-table th {{ background-color: {header_bg} !important; }}</style>"
        )
    if cells_bg:
        css_parts.append(
            f"<style>.ai-table-card .ai-table td {{ background-color: {cells_bg} !important; }}</style>"
        )
    if align in {"left", "center", "right"}:
        css_parts.append(
            f"<style>.ai-table-card .ai-table th, .ai-table-card .ai-table td {{ text-align: {align}; }}</style>"
        )
    if custom_css:
        css_parts.append(f"<style>{custom_css}</style>")

    return "\n".join(css_parts)


def _detect_percent_column(column_name: str) -> bool:
    name = str(column_name).lower()
    return any(token in name for token in ("%", "percent", "pct", "процент", "конвер"))


def _make_number_formatter(series: pd.Series) -> callable:
    """Хелпер для форматирования числовых колонок (тысячные разделители)."""

    non_na = series.dropna()
    if non_na.empty or pd.api.types.is_integer_dtype(series):
        decimals = 0
    else:
        if np.all(np.isclose(non_na % 1, 0)):
            decimals = 0
        else:
            max_abs = float(non_na.abs().max()) if not non_na.empty else 0.0
            decimals = 3 if max_abs < 1 else 2

    def format_number(val: float) -> str:
        if pd.isna(val):
            return "—"
        try:
            formatted = f"{float(val):,.{decimals}f}"
            if decimals:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
        except Exception:
            return str(val)

    return format_number


def _make_percent_formatter(series: pd.Series) -> callable:
    """Форматирование процентов (0-1 интерпретируем как доли)."""

    non_na = series.dropna()
    treat_as_ratio = False
    if not non_na.empty:
        max_abs = float(non_na.abs().max())
        treat_as_ratio = 0 < max_abs <= 1.0

    def format_percent(val: float) -> str:
        if pd.isna(val):
            return "—"
        try:
            value = float(val)
            if treat_as_ratio:
                value *= 100
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}%"
        except Exception:
            return str(val)

    return format_percent


def _build_table_formatters(pdf: pd.DataFrame, style_meta: dict | None) -> dict:
    """Комбинирует форматеры по умолчанию и пользовательские (если переданы)."""

    formatters: dict[str, callable] = {}
    custom = {}
    if style_meta and isinstance(style_meta.get("formatters"), dict):
        custom = style_meta.get("formatters") or {}

    for col in pdf.columns:
        if col in custom:
            fmt = custom[col]
            if callable(fmt):
                formatters[col] = fmt
            elif isinstance(fmt, str):
                def _wrap(template: str):
                    def _formatter(val):
                        if pd.isna(val):
                            return "—"
                        try:
                            return template.format(val)
                        except Exception:
                            return str(val)
                    return _formatter
                formatters[col] = _wrap(fmt)
            continue

        series = pdf[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            def _format_dt(val):
                if pd.isna(val):
                    return "—"
                try:
                    return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return str(val)
            formatters[col] = _format_dt
        elif pd.api.types.is_numeric_dtype(series):
            if _detect_percent_column(col):
                formatters[col] = _make_percent_formatter(series)
            else:
                formatters[col] = _make_number_formatter(series)

    return formatters


def _safe_apply_styler_code(styler: pd.io.formats.style.Styler, code: str | None) -> tuple[pd.io.formats.style.Styler, list[str]]:
    """Выполняет пользовательский styler_code в легкой песочнице."""

    warnings: list[str] = []
    if not code:
        return styler, warnings

    banned = re.compile(r"\b(import|open|exec|eval|subprocess|os\.|sys\.|pathlib|__|socket)\b", re.IGNORECASE)
    if banned.search(code):
        warnings.append("Styler-код отклонён: обнаружены небезопасные конструкции.")
        return styler, warnings

    local_env = {"styler": styler, "pd": pd, "np": np}
    safe_builtins = {"min": min, "max": max, "sum": sum, "len": len, "abs": abs, "round": round}

    try:
        exec(code, {"__builtins__": safe_builtins}, local_env)
        updated = local_env.get("styler", styler)
        if isinstance(updated, pd.io.formats.style.Styler):
            styler = updated
    except Exception as exc:
        warnings.append(f"Styler-код вызвал ошибку: {exc}")

    return styler, warnings


def _render_table_html(pdf: pd.DataFrame, style_meta: dict | None) -> tuple[str, list[str]]:
    """Готовит HTML-тело таблицы вместе с базовым оформлением."""

    pdf_copy = pdf.copy()
    for col in pdf_copy.select_dtypes(include=["object", "string"]).columns:
        pdf_copy[col] = pdf_copy[col].fillna("—")

    formatters = _build_table_formatters(pdf_copy, style_meta)

    styler = pdf_copy.style
    if hasattr(styler, "hide"):
        styler = styler.hide(axis="index")
    else:
        styler = styler.hide_index()

    styler = styler.set_table_attributes('class="ai-table" data-role="ai-table"')
    styler = styler.format(formatters, na_rep="—")

    align = (style_meta or {}).get("align") or "left"
    styler = styler.set_table_styles([
        {"selector": "th", "props": [("text-align", align)]},
        {"selector": "td", "props": [("text-align", align)]},
    ], overwrite=False)

    styler, warnings = _safe_apply_styler_code(styler, (style_meta or {}).get("styler_code"))

    table_html = styler.to_html()
    css_block = _compose_table_css(style_meta)

    return f"{css_block}\n<div class=\"ai-table-card\">{table_html}</div>", warnings


# Отрисовка содержимого таблицы с учетом стилей
def _render_table_content(pdf: pd.DataFrame, meta: dict):
    """Отрисовывает таблицу с базовым HTML и пользовательно-ориентированными хуками."""

    style_meta = (meta.get("table_style") or {})
    if not style_meta and "next_table_style" in st.session_state:
        style_meta = st.session_state["next_table_style"]
        meta["table_style"] = style_meta
        del st.session_state["next_table_style"]

    table_html, warnings = _render_table_html(pdf, style_meta)

    if style_meta:
        st.info(f"🎨 Применяем стиль: {style_meta}")
    for warning_msg in warnings:
        st.warning(warning_msg)

    # Высоту подбираем эвристикой: 40px на заголовок + строки таблицы
    estimated_height = min(720, 120 + 28 * (len(pdf) + 1))
    st.components.v1.html(table_html, height=estimated_height, scrolling=True)


# Отрисовка подписи таблицы
def _render_table_caption(meta: dict, pdf: pd.DataFrame):
    """Отрисовка подписи под таблицей с fallback на анализ SQL"""
    explain = (meta.get("explain") or "").strip()
    if explain:
        st.caption(explain)
    else:
        sql = (meta.get("sql") or "").strip()
        info = _extract_sql_info(sql, pdf)
        src = ", ".join(info.get("tables") or []) or "источник не указан"
        period = info.get("period") or "период не указан"
        st.caption(f"Источник: {src}. Период: {period}.")


# Отрисовка подписи графика
def _render_chart_caption(meta: dict):
    """Отрисовка подписи под графиком с fallback на последний SQL"""
    explain = (meta.get("explain") or "").strip()
    if explain:
        st.caption(explain)
    else:
        explain = st.session_state.get("last_sql_meta", {}).get("explain", "").strip()
        if explain:
            st.caption(explain)


# Отрисовка SQL блока
def _render_sql_block(meta: dict):
    """Отрисовка свернутого блока с SQL-кодом"""
    used_sql = (meta.get("sql") or "").strip()
    orig_sql = (meta.get("sql_original") or "").strip()
    
    if not used_sql and not orig_sql:
        return
    
    with st.expander("Показать SQL", expanded=False):
        if used_sql:
            st.markdown("**Использованный SQL**")
            st.code(used_sql, language="sql")
            if orig_sql and orig_sql != used_sql:
                st.markdown("**Исходный SQL от модели**")
                st.code(orig_sql, language="sql")
        elif orig_sql:
            st.code(orig_sql, language="sql")


# Отрисовка кода Plotly
def _render_plotly_code(meta: dict):
    """Отрисовка свернутого блока с кодом Plotly"""
    plotly_src = (meta.get("plotly_code") or "").strip()
    if not plotly_src:
        return
    
    with st.expander("Показать код Plotly", expanded=False):
        st.code(plotly_src, language="python")


# Отрисовка кнопок скачивания
def _render_download_buttons(data, item: dict, data_type: str):
    """Отрисовка кнопок скачивания для таблиц и графиков"""
    ts = (item.get("ts") or data_type).replace(":", "-")
    
    if data_type == "table":
        # CSV и XLSX кнопки для таблиц
        try:
            col_csv, col_xlsx, _ = st.columns([4, 4, 2], gap="small")
        except TypeError:
            col_csv, col_xlsx, _ = st.columns([4, 4, 2])
        
        with col_csv:
            st.download_button(
                "Скачать CSV",
                data=_df_to_csv_bytes(data),
                file_name=f"table_{ts}.csv",
                mime="text/csv",
                key=f"dl_csv_{ts}",
                use_container_width=True,
            )
        with col_xlsx:
            st.download_button(
                "Скачать XLSX",
                data=_df_to_xlsx_bytes(data, "Result"),
                file_name=f"table_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{ts}",
                use_container_width=True,
            )
    
    elif data_type == "chart":
        # HTML кнопка для графиков
        html_bytes = data.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
        
        try:
            col_html, _ = st.columns([4, 8], gap="small")
        except TypeError:
            col_html, _ = st.columns([4, 8])
        
        with col_html:
            st.download_button(
                "Скачать график",
                data=html_bytes,
                file_name=f"chart_{ts}.html",
                mime="text/html",
                key=f"dl_html_{ts}",
                use_container_width=True,
            )


# ======================== КОНЕЦ РЕФАКТОРИНГА ========================

def _df_to_csv_bytes(pdf: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pdf.to_csv(buf, index=False)
    return buf.getvalue()


def _df_to_xlsx_bytes(pdf: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pdf.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def _build_plotly_table(pdf: pd.DataFrame) -> go.Figure:
    """Создаёт Plotly-таблицу с темным стилем (контрастная шапка и строки)."""

    values = [pdf[c].tolist() for c in pdf.columns] if not pdf.empty else [[] for _ in pdf.columns]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[str(c) for c in pdf.columns],
                    fill_color="#111827",
                    font=dict(color="#f9fafb", size=13),
                    align="left",
                ),
                cells=dict(
                    values=values,
                    fill_color="#1f2933",
                    font=dict(color="#f9fafb"),
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=12, b=0), height=min(560, 80 + 24 * len(pdf)))
    return fig

def _default_plotly_table_code(df: pd.DataFrame) -> str:
    """Генерирует базовый темный Plotly-код с явным перечислением колонок."""
    columns = [str(c) for c in df.columns]
    header_vals = ", ".join(repr(col) for col in columns)
    cell_vals = ",\n        ".join(f"df[{repr(col)}]" for col in columns)

    code = (
        "fig = go.Figure(data=[go.Table(\n"
        "    header=dict(\n"
        f"        values=[{header_vals}],\n"
        "        fill_color=\"#111827\",\n"
        "        font=dict(color=\"#f9fafb\", size=13),\n"
        "        align=\"left\"\n"
        "    ),\n"
        "    cells=dict(\n"
        "        values=[\n"
        f"        {cell_vals}\n"
        "        ],\n"
        "        fill_color=\"#1f2933\",\n"
        "        font=dict(color=\"#f9fafb\"),\n"
        "        align=\"left\"\n"
        "    )\n"
        ")])\n"
        "fig.update_layout(margin=dict(l=0, r=0, t=12, b=0), height=min(560, 80 + 24 * len(df)))\n"
    )
    return code

def _history_zip_bytes() -> bytes:
    """Собрать ZIP с историей результатов (таблицы: csv+xlsx+sql, графики: html)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(st.session_state["results"], start=1):
            base = f"{idx:03d}_{item['kind']}_{item['ts'].replace(':','-')}"
            if item["kind"] == "table" and isinstance(item.get("df_pl"), pl.DataFrame):
                pdf = item["df_pl"].to_pandas()
                zf.writestr(f"{base}.csv", _df_to_csv_bytes(pdf))
                zf.writestr(f"{base}.xlsx", _df_to_xlsx_bytes(pdf, "Result"))
                sql = (item.get("meta") or {}).get("sql")
                if sql:
                    zf.writestr(f"{base}.sql.txt", sql.encode("utf-8"))
            elif item["kind"] == "chart" and isinstance(item.get("fig"), go.Figure):
                html_buf = io.StringIO()
                item["fig"].write_html(html_buf, include_plotlyjs="cdn", full_html=True)
                zf.writestr(f"{base}.html", html_buf.getvalue().encode("utf-8"))
    return buf.getvalue()

def _read_text_file_quiet(path: str, max_bytes: int = 256 * 1024) -> str:
    """Читает текстовый файл (если есть). Возвращает строку или пустую строку при ошибке."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    except Exception:
        return ""

# --- 1) Узнаём: это ошибка по схеме? (таблица/колонка/идентификатор) ---
def _is_schema_error(err_text: str) -> bool:
    # ClickHouse коды и фразы, когда LLM «промахнулся» по схеме
    SIGNS = ("Unknown table", "Unknown column", "Unknown identifier",
        "There is no column",          # <<< добавьте эту фразу
        "Code: 60",  # table doesn't exist
        "Code: 47",  # unknown identifier
        )  # unknown identifier
    t = (err_text or "").strip()
    return any(s in t for s in SIGNS)


# --- 1a) Узнаём: это ошибка регулярного выражения RE2 (ClickHouse)? ---
def _is_re2_error(err_text: str) -> bool:
    t = (err_text or "").lower()
    return (
        "unterminated subpattern" in t
        or "missing ), unterminated subpattern" in t
        or ("re2" in t and "missing" in t)
        or "invalid escape" in t
    )


# --- 1b) Узнаём: это ошибка агрегации ClickHouse? (184/215)
def _is_aggregation_error(err_text: str) -> bool:
    t = (err_text or "").upper()
    return (
        "ILLEGAL_AGGREGATION" in t
        or "NOT_AN_AGGREGATE" in t
        or "CODE: 184" in t
        or "CODE: 215" in t
    )

# --- 2) Готовим компактный хинт-снимок схемы (через ClickHouse_client.get_schema) ---
def _schema_hint(ch_client, database: str = "db1", max_tables: int = 12, max_cols: int = 8) -> str:
    """
    Берём актуальную схему и сворачиваем в короткий system-хинт.
    """
    sch = ch_client.get_schema(database)  # <-- используем уже существующий get_schema
    lines = [f"Схема БД `{database}` (сокращённо):"]
    for i, (table, cols) in enumerate(sch.items()):
        if i >= max_tables:
            break
        cols_s = ", ".join(f"`{c}` {t}" for c, t in (cols[:max_cols] if cols else []))
        lines.append(f"- `{database}.{table}`: {cols_s}" if cols_s else f"- `{database}.{table}`: (пусто)")
    return "\n".join(lines)


# ======================== Семантическая защита SQL (из базы знаний) ========================

def _read_kb_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _parse_md_table_rows(md_section: str) -> list[tuple[str, str]]:
    """Возвращает список пар (column_name, type) из markdown-таблицы в секции."""
    rows = []
    # Ищем строки вида: | `col` | Type | ... |
    for m in re.finditer(r"^\|\s*`([^`]+)`\s*\|\s*([A-Za-z0-9_]+)\s*\|", md_section, flags=re.MULTILINE):
        col = m.group(1).strip()
        typ = m.group(2).strip()
        if col:
            rows.append((col, typ))
    return rows


def _extract_section(md: str, title_regex: str) -> str:
    """Возвращает текст секции, начиная с заголовка до следующего заголовка того же уровня."""
    m = re.search(title_regex, md, flags=re.IGNORECASE)
    if not m:
        return ""
    start = m.start()
    # Следующий заголовок уровня ###
    m2 = re.search(r"^###\s+", md[start+3:], flags=re.MULTILINE)
    end = (start + 3 + m2.start()) if m2 else len(md)
    return md[start:end]


def _build_metrics_meta_from_kb(md_path: str) -> dict:
    """Строит мета-словарь категорий метрик из KB. Ключи категорий:
    - subscriptions
    - payments_amount
    - payments_count
    - active_users
    """
    md = _read_kb_file(md_path)
    if not md:
        return {
            "subscriptions": set(),
            "payments_amount": set(),
            "payments_count": set(),
            "active_users": set(),
        }

    subs_sec = _extract_section(md, r"^###\s+Метрики\s+подписок")
    pay_day_sec = _extract_section(md, r"^###\s+Метрики\s+платежей\s*\(дневные\)")
    pay_cum_sec = _extract_section(md, r"^###\s+Кумулятивные\s+метрики\s+платежей\s*\(месячные\)")
    act_sec = _extract_section(md, r"^###\s+Метрики\s+активных\s+пользователей")

    subs_rows = _parse_md_table_rows(subs_sec)
    pay_day_rows = _parse_md_table_rows(pay_day_sec)
    pay_cum_rows = _parse_md_table_rows(pay_cum_sec)
    act_rows = _parse_md_table_rows(act_sec)

    subscriptions = {c for c, _ in subs_rows} | {"paying_users", "paying_users_day"}

    payments_amount = set()
    payments_count = set()

    def classify_payment_rows(rows: list[tuple[str, str]]):
        for col, typ in rows:
            low = col.lower()
            if typ.lower().startswith("float") or ("_pl" in low) or ("amount" in low):
                payments_amount.add(col)
            elif ("count" in low) or ("refunded" in low) or typ.lower().startswith("uint"):
                payments_count.add(col)
            else:
                # по умолчанию не относим
                pass

    classify_payment_rows(pay_day_rows)
    classify_payment_rows(pay_cum_rows)

    active_users = {c for c, _ in act_rows}

    # Явные корректировки
    payments_amount |= {"Android_PL", "IOS_PL", "Android_PL_cum", "IOS_PL_cum", "refunded_amount_appstore", "refunded_amount_yookassa"}

    return {
        "subscriptions": subscriptions,
        "payments_amount": payments_amount,
        "payments_count": payments_count,
        "active_users": active_users,
    }


def _get_kb_metrics_meta() -> dict:
    """Кэширует и возвращает метрики из KB."""
    key = "kb_metrics_meta"
    if key in st.session_state and isinstance(st.session_state[key], dict):
        return st.session_state[key]
    meta = _build_metrics_meta_from_kb(KB_T_AI_GLOBAL_REPORT_FILE)
    st.session_state[key] = meta
    return meta


def _infer_intent_category(sql_text: str, base_messages: list) -> str | None:
    """Грубая эвристика определения намерения: суммы оплат / кол-во оплат / подписки."""
    text = (sql_text or "") + "\n" + "\n".join(
        [m.get("content", "") for m in (base_messages or []) if isinstance(m, dict)]
    )
    low = text.lower()
    if any(w in low for w in ["выручк", "сумм", "доход", "revenue", "amount"]):
        return "payments_amount"
    if any(w in low for w in ["число оплат", "кол-во оплат", "количество оплат", "покупк", "transactions", "count"]):
        return "payments_count"
    if any(w in low for w in ["подписк", "subscr", "paying users"]):
        return "subscriptions"
    return None


def _semantic_guard_text(category: str, meta: dict) -> str:
    cat2ru = {
        "payments_amount": "суммы оплат/выручка",
        "payments_count": "количество оплат/покупок",
        "subscriptions": "количество подписок",
        "active_users": "активные пользователи",
    }
    all_known = set().union(*meta.values()) if meta else set()
    allowed = set(meta.get(category, set()))
    forbidden = all_known - allowed
    msg = [
        f"Семантическая категория запроса: {cat2ru.get(category, category)}.",
        "Сохраняй бизнес-смысл метрик. Запрещены подмены между оплатами и подписками.",
        "Если нет подходящих полей — верни ошибку и НЕ меняй категорию.",
        "Разрешено использовать только эти поля (как источники агрегатов): "
        + ", ".join(sorted(f"`{c}`" for c in allowed)) if allowed else "—",
    ]
    if forbidden:
        msg.append(
            "Запрещено использовать в этой задаче: " + ", ".join(sorted(f"`{c}`" for c in forbidden))
        )
    return "\n".join(msg)


def _extract_used_metrics(sql_text: str, meta: dict) -> set[str]:
    """Грубый парсер: вытаскивает имена известных метрик, упомянутых в SELECT/WHERE/ORDER."""
    if not sql_text:
        return set()
    known = set().union(*meta.values()) if meta else set()
    used = set()
    # Ищем бэктики `name`
    for m in re.finditer(r"`([A-Za-z0-9_]+)`", sql_text):
        name = m.group(1)
        if name in known:
            used.add(name)
    # Ищем без бэктиков (ограничим известными именами)
    tokens = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", sql_text))
    used |= (tokens & known)
    return used


def _validate_sql_semantics(sql_text: str, category: str, meta: dict) -> tuple[bool, str]:
    if not category or not meta:
        return True, ""
    allowed = set(meta.get(category, set()))
    all_known = set().union(*meta.values())
    forbidden = all_known - allowed
    used = _extract_used_metrics(sql_text, meta)
    bad = used & forbidden
    if bad:
        return False, "Семантическое нарушение: " + ", ".join(sorted(bad))
    # Доп. правило: для payments_amount должны присутствовать денежные поля (PL/amount)
    if category == "payments_amount":
        if not any(("_PL" in u) or ("amount" in u.lower()) for u in used):
            return False, "Ожидались денежные поля (PL/amount), но они не найдены."
    if category == "payments_count":
        if not any(("count" in u.lower()) or ("refunded" in u.lower()) for u in used):
            return False, "Ожидались счётчики покупок/оплат (count/refunded), но они не найдены."
    if category == "subscriptions":
        if not any(("sub" in u.lower()) or (u.startswith("paying_users")) for u in used):
            return False, "Ожидались метрики подписок, но они не найдены."
    return True, ""

# Возвращает список дашбордов из локальной папки документов.
# Ищем .md/.txt/.json/.yaml с упоминаниями dashboard/дашборд/DataLens.
# Ничего не режем, показываем всё, что нашли.

def _dashboards_catalog_from_docs(doc_dir: str) -> str:

    if not os.path.isdir(doc_dir):
        return "каталог документов не найден."

    items: list[str] = []
    for root, _, files in os.walk(doc_dir):
        for fn in files:
            low = fn.lower()
            if not low.endswith((".md", ".txt", ".json", ".yaml", ".yml")):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(128 * 1024)  # читаем до 128KB
            except Exception:
                continue

            # заголовок берем из первой Markdown-«решётки», если есть
            title = None
            m = re.search(r"^\s*#\s+(.+)$", txt, re.MULTILINE)
            if m:
                title = m.group(1).strip()

            # эвристика наличия дашборда
            if re.search(r"\b(dashboard|дашборд|дашборды|data\s*lens)\b", txt, re.IGNORECASE):
                items.append(f"- {title or fn}")

            # лёгкая поддержка JSON со списком дашбордов
            if low.endswith(".json"):
                try:
                    data = json.loads(txt)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, str) and re.search(r"dashboard|дашборд", v, re.I):
                                items.append(f"- {k}: {v}")
                except Exception:
                    pass

    if not items:
        return "описаний дашбордов не найдено."
    return "\n".join(items)


def _get_cached_schema(ch_client, database: str = "db1") -> dict:
    """
    Получает схему БД с кэшированием для быстрого доступа.
    """
    cache_key = f"schema_{database}"
    
    # Проверяем кэш
    if cache_key in st.session_state["db_schema_cache"]:
        return st.session_state["db_schema_cache"][cache_key]
    
    # Если нет в кэше, получаем из БД
    try:
        schema = ch_client.get_schema(database)
        st.session_state["db_schema_cache"][cache_key] = schema
        return schema
    except Exception as e:
        st.warning(f"Не удалось получить схему БД: {e}")
        return {}

def _check_table_exists(table_name: str, ch_client, database: str = "db1") -> bool:
    """
    Быстрая проверка существования таблицы в БД.
    """
    schema = _get_cached_schema(ch_client, database)
    return table_name in schema

def _enhanced_table_search(query: str, chroma_path: str, collection_name: str) -> list:
    """
    Улучшенный поиск таблиц в базе знаний с несколькими стратегиями.
    """
    hits = []
    try:
        # Стратегия 1: Прямой поиск по названию таблицы
        if 't_' in query.lower():
            table_name = None
            words = query.split()
            for word in words:
                if word.startswith('t_') and len(word) > 2:
                    table_name = word
                    break
            
            if table_name:
                hits += retriever.retrieve(
                    f"таблица {table_name} описание структура поля DDL",
                    k=5,
                    chroma_path=chroma_path,
                    collection_name=collection_name,
                )
        
        # Стратегия 2: Поиск по ключевым словам из запроса
        keywords = []
        if 'монетизация' in query.lower():
            keywords.append('монетизация партнеры')
        if 'партнер' in query.lower():
            keywords.append('партнеры статус')
        if 'блокировка' in query.lower():
            keywords.append('блокировка монетизация')
        
        for keyword in keywords:
            hits += retriever.retrieve(
                keyword, k=3,
                chroma_path=chroma_path,
                collection_name=collection_name,
            )
        
        # Стратегия 3: Общий поиск по запросу
        hits += retriever.retrieve(
            query, k=5,
            chroma_path=chroma_path,
            collection_name=collection_name,
        )
        
    except Exception as e:
        st.warning(f"Ошибка при поиске в базе знаний: {e}")
    
    # Убираем дубликаты по тексту
    seen_texts = set()
    unique_hits = []
    for hit in hits:
        text = hit.get("text", "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_hits.append(hit)
    
    return unique_hits

def _tables_catalog_from_db(ch_client, database: str = "db1") -> str:
    """
    Возвращает ПОЛНЫЙ каталог таблиц из ClickHouse (system.columns) без сокращений.
    Ничего не обрезаем по числу таблиц/колонок.
    """
    try:
        schema: dict[str, list[tuple[str, str]]] = ch_client.get_schema(database)
    except Exception as e:
        # важна явная диагностика, чтобы пользователь видел причину
        return f"не удалось получить схему БД `{database}` ({e})."

    lines: list[str] = [f"Всего таблиц: {len(schema)}"]
    # сортируем по имени таблицы для устойчивого вывода
    for table, cols in sorted(schema.items(), key=lambda x: x[0]):
        # показываем ВСЕ колонки (имя и тип), без усечений
        cols_fmt = ", ".join(f"`{name}`:{ctype}" for name, ctype in cols)
        lines.append(f"- `{database}`.`{table}` ({len(cols)} колонок): {cols_fmt}")
    return "\n".join(lines)


# --- 3) Выполнить SQL; при ошибке по схеме — запросить схему, перегенерировать SQL и повторить ---
def run_sql_with_auto_schema(sql_text: str,
                             base_messages: list,
                             ch_client,
                             llm_client,
                             prompts_map: dict,
                             model_name: str,
                             retry_delay: float = 0.35):
    """
    sql_text        — исходный SQL (сгенерированный LLM ранее).
    base_messages   — ваша история/сообщения пользователя, которые вы уже подаёте модели.
    ch_client       — экземпляр ClickHouse_client.
    llm_client      — клиент OpenAI (или совместимый) с .chat.completions.create(...)
    prompts_map     — ваш словарь с промптами; нужен ключ 'sql'.
    model_name      — имя модели.
    retry_delay     — короткая пауза перед финальным повтором.
    Возвращает (df, used_sql)
    """
    import re, time

    # 3.0. Предварительная защита: агрегаты в WHERE и CTE с агрегатами для дат запрещены
    try:
        # Проверяем любые CTE с агрегатами (запрещено)
        has_agg_cte = bool(re.search(r"WITH\s+\w+\s+AS\s*\(\s*SELECT\s+(max|sum|avg|count|min|anyLast|any|argMax|argMin)\s*\(", sql_text, flags=re.IGNORECASE))
        
        m_where = re.search(r"\bWHERE\b([\s\S]*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bSETTINGS\b|\bFORMAT\b|\bUNION\b|$)", sql_text, flags=re.IGNORECASE)
        where_part = (m_where.group(1) if m_where else "")
        has_agg_in_where = bool(re.search(r"\b(max|sum|avg|count|min|anyLast|any|argMax|argMin)\s*\(", where_part, flags=re.IGNORECASE))
        # Разрешаем агрегаты в WHERE только если это внутри подзапроса SELECT (скаляр)
        has_select_inside_where = ("select" in where_part.lower())
        # Дополнительно проверяем на скалярные подзапросы в WHERE
        has_scalar_subquery = bool(re.search(r"\([^)]*SELECT[^)]*\)", where_part, flags=re.IGNORECASE))
        
        if has_agg_cte or (has_agg_in_where and not (has_select_inside_where or has_scalar_subquery)):
            guard_hint = (
                "Запрещены агрегатные функции в любых CTE и на верхнем уровне в WHERE. "
                "Используй только явные даты, указанные пользователем: \n"
                "WHERE report_date = '2025-08-31' или WHERE report_date BETWEEN '2025-08-01' AND '2025-08-31'\n"
                "Если пользователь НЕ указал дату — используй: WHERE report_date = (SELECT max(report_date) FROM <таблица> WHERE <метрика> > 0). "
                "НЕ используй WITH для агрегатов. Верни только блок ```sql```.")

            regen_msgs_pre = (
                [
                    {"role": "system", "content": prompts_map["sql"]},
                    {"role": "system", "content": guard_hint},
                ]
                + base_messages
                + [
                    {"role": "user", "content": f"Исходный SQL (исправь согласно правилам):\n```sql\n{sql_text}\n```"}
                ]
            )

            try:
                regen_pre = llm_client.chat.completions.create(
                    model=model_name, messages=regen_msgs_pre, temperature=0
                ).choices[0].message.content
                m_pre = re.search(r"```sql\s*(.*?)```", regen_pre, flags=re.DOTALL | re.IGNORECASE)
                if m_pre:
                    sql_text = m_pre.group(1).strip()
            except Exception:
                # тихо продолжаем с исходным SQL, если не удалось перегенерировать
                pass
    except Exception:
        pass

    # 3.1. Первая попытка — просто выполнить
    try:
        df = ch_client.query_run(sql_text)
        return df, sql_text
    except Exception as e:
        err = str(e)
        
        # Проверяем и исправляем возможное дублирование префикса БД
        dup = f"{DEFAULT_DB}.{DEFAULT_DB}."
        if dup in sql_text:
            fixed_sql = sql_text.replace(dup, f"{DEFAULT_DB}.")
            try:
                df = ch_client.query_run(fixed_sql)
                return df, fixed_sql
            except Exception:
                pass  # Если и исправленный не работает, продолжаем обычную обработку
        
        # Если это ошибка RE2 (регулярки) — попробуем простой автофикс: заменить "match"/"regexp" на LIKE
        if _is_re2_error(err):
            try:
                sql_fixed_regex = re.sub(r"(?i)\bmatch\s*\(([^,]+),\s*'([^']*)'\)", r"\1 LIKE '%\2%'", sql_text)
                sql_fixed_regex = re.sub(r"(?i)\bmatch\s*\(([^,]+),\s*\"([^\"]*)\"\)", r"\1 LIKE '%\2%'", sql_fixed_regex)
                sql_fixed_regex = re.sub(r"(?i)\bregexp\s+\'([^']*)\'", r"LIKE '%\1%'", sql_fixed_regex)
                df = ch_client.query_run(sql_fixed_regex)
                return df, sql_fixed_regex
            except Exception:
                pass

        # Попытка авто-чинить ошибки агрегаций ClickHouse
        if _is_aggregation_error(err):
            try:
                agg_fix_hint = (
                    "Обнаружена ошибка агрегации ClickHouse (ILLEGAL_AGGREGATION/NOT_AN_AGGREGATE). "
                    "Перепиши SQL с учётом правил: "
                    "1) Не используй агрегаты в WHERE верхнего уровня. "
                    "2) Для 'конца прошлого месяца' используй скалярный подзапрос в WHERE: "
                    "WHERE `report_date` = (SELECT max(`report_date`) FROM <таблица> WHERE `report_date` < toStartOfMonth(today()) AND <метрика> > 0). "
                    "3) Если дату нужно вывести в SELECT — используй ТОТ ЖЕ скалярный подзапрос как значение, НЕ колонку `report_date`. "
                    "4) Не присваивай агрегату алиас с именем столбца таблицы (нельзя AS `report_date`). Используй нейтральный алиас. "
                    "5) Для исключения списков используй NOT IN ('..','..'), без arrayJoin в WHERE. Верни только блок ```sql```."
                )

                regen_msgs = (
                    [
                        {"role": "system", "content": prompts_map["sql"]},
                        {"role": "system", "content": agg_fix_hint},
                    ]
                    + base_messages
                )
                regen = llm_client.chat.completions.create(
                    model=model_name, messages=regen_msgs, temperature=0
                ).choices[0].message.content
                m = re.search(r"```sql\s*(.*?)```", regen, flags=re.DOTALL | re.IGNORECASE)
                if m:
                    sql_fixed = m.group(1).strip()
                    df = ch_client.query_run(sql_fixed)
                    return df, sql_fixed
            except Exception:
                pass

        if not _is_schema_error(err):
            # не схема — пробрасываем как есть
            raise

    # 3.2. Схема понадобилась — составим ТОЧНЫЙ хинт по ЗАДЕЙСТВОВАННЫМ таблицам
    # Извлекаем имена таблиц из исходного SQL
    try:
        tbl_names = []
        for pat in [r"\bFROM\s+([a-zA-Z0-9_.`\"]+)", r"\bJOIN\s+([a-zA-Z0-9_.`\"]+)"]:
            tbl_names += [m.group(1).strip('`"') for m in re.finditer(pat, sql_text, flags=re.IGNORECASE)]
        # нормализуем к именам таблиц без префикса БД для get_schema
        short_tbls = sorted(set([t.split(".")[-1] for t in tbl_names if t]))
    except Exception:
        short_tbls = []

    # Берём схему только по этим таблицам с полным списком колонок
    try:
        precise_schema = ch_client.get_schema(DEFAULT_DB, tables=short_tbls) if short_tbls else ch_client.get_schema(DEFAULT_DB)
    except Exception:
        precise_schema = {}

    # Собираем компактный, но полный по колонкам хинт
    lines = [f"Схема БД `{DEFAULT_DB}` для задействованных таблиц (полный список колонок):"]
    for table, cols in precise_schema.items():
        cols_s = ", ".join(f"`{name}` {ctype}" for name, ctype in cols)
        lines.append(f"- `{DEFAULT_DB}.{table}`: {cols_s}" if cols_s else f"- `{DEFAULT_DB}.{table}`: (пусто)")
    # Семантический guard (на основе KB) — добавляем к precise_hint и отдельным сообщением
    precise_hint = "\n".join(lines)
    guard_msgs_extra = []
    if SQL_SEMANTIC_GUARD:
        try:
            kb_meta = _get_kb_metrics_meta()
            category = _infer_intent_category(sql_text, base_messages)
            if category:
                sem_text = _semantic_guard_text(category, kb_meta)
                precise_hint = precise_hint + "\n\n" + "Семантические правила (из базы знаний):\n" + sem_text
                guard_msgs_extra = [{"role": "system", "content": sem_text}]
        except Exception:
            # Если что-то пошло не так — просто не добавляем guard
            guard_msgs_extra = []

    # Попробуем извлечь отсутствующие колонки из текста ошибки
    missing_cols = []
    try:
        miss_part = re.search(r"Missing columns:\s*([^\)]*?) while", err)
        if miss_part:
            missing_cols = re.findall(r"'([^']+)'", miss_part.group(1))
    except Exception:
        missing_cols = []

    guard_instr = (
        "Перегенерируй корректный ```sql``` СТРОГО по схемам выше. "
        + (f"Не используй отсутствующие колонки: {', '.join(missing_cols)}. " if missing_cols else "")
        + "Используй только перечисленные имена столбцов и таблиц, ничего не выдумывай. Верни только блок ```sql```."
    )

    regen_msgs = (
        [{"role": "system", "content": precise_hint},
         {"role": "system", "content": prompts_map["sql"]}]
        + guard_msgs_extra
        + base_messages
        + [{"role": "system", "content": guard_instr}]
    )

    regen = llm_client.chat.completions.create(
        model=model_name, messages=regen_msgs, temperature=0
    ).choices[0].message.content

    m = re.search(r"```sql\s*(.*?)```", regen, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError("Не удалось извлечь перегенерированный SQL из ответа модели.")

    sql2 = m.group(1).strip()
    # Защита от дублирования префикса БД после перегенерации SQL
    dup2 = f"{DEFAULT_DB}.{DEFAULT_DB}."
    if dup2 in sql2:
        sql2 = sql2.replace(dup2, f"{DEFAULT_DB}.")

    # Дополнительная семантическая проверка и, при необходимости, один повтор с усиленным guard
    if SQL_SEMANTIC_GUARD:
        try:
            kb_meta = _get_kb_metrics_meta()
            category = _infer_intent_category(sql_text, base_messages)
            ok, reason = _validate_sql_semantics(sql2, category, kb_meta)
            if not ok and category:
                fix_hint = (
                    "Предыдущий перегенерированный SQL нарушил семантические правила (" + reason + ").\n"
                    "Исправь SQL: используй только допустимые поля для категории; не подменяй смысл. Верни только блок ```sql```."
                )
                retry_msgs = (
                    [{"role": "system", "content": precise_hint},
                     {"role": "system", "content": prompts_map["sql"]}]
                    + [{"role": "system", "content": _semantic_guard_text(category, kb_meta)}]
                    + base_messages
                    + [{"role": "system", "content": guard_instr}, {"role": "system", "content": fix_hint}]
                )
                retry_reply = llm_client.chat.completions.create(
                    model=model_name, messages=retry_msgs, temperature=0
                ).choices[0].message.content
                m_retry = re.search(r"```sql\s*(.*?)```", retry_reply, flags=re.DOTALL | re.IGNORECASE)
                if m_retry:
                    sql2_try = m_retry.group(1).strip()
                    if dup2 in sql2_try:
                        sql2_try = sql2_try.replace(dup2, f"{DEFAULT_DB}.")
                    ok2, reason2 = _validate_sql_semantics(sql2_try, category, kb_meta)
                    if ok2:
                        sql2 = sql2_try
                    else:
                        # оставляем sql2 как есть — ниже упадём с ошибкой выполнения или вернём ошибку
                        pass
        except Exception:
            pass

    # 3.3. Короткий retry на сетевые/временные сбои при втором запуске
    for _ in range(2):  # одна пауза и вторая попытка
        try:
            df2 = ch_client.query_run(sql2)
            return df2, sql2
        except Exception as e2:
            last = str(e2)
            time.sleep(retry_delay)
    # если дошли сюда — оба раза не получилось
    raise RuntimeError(f"Повтор после обновления схемы тоже упал: {last}")

# ----------------------- Сайдбар -----------------------
with st.sidebar:
    # Единственная кнопка — переиндексация базы знаний
    # Комментарий: держим всё максимально компактно, без заголовков/разделителей,
    #               статус и краткий итог показываем прямо под кнопкой.
    if st.button("Переиндексировать базу", use_container_width=True):
        start_ts = datetime.now()
        with st.status("Индексируем…", expanded=False) as status:
            try:
                # Ленивая загрузка, чтобы не тянуть зависимости, пока кнопка не нажата
                from ingest import run_ingest

                stats = run_ingest()  # запускаем индексацию и получаем сводку
                dur = (datetime.now() - start_ts).total_seconds()

                # Короткий итог без «воды»
                status.update(label="Готово", state="complete")
                st.success(
                    f'Файлов: {stats.get("files", 0)} | '
                    f'Чанков: {stats.get("chunks", 0)} | '
                    f'Добавлено: {stats.get("added", 0)} | '
                    f'{dur:.1f} c'
                )
            except Exception as e:
                # Показываем краткую ошибку, чтобы не «шуметь» трейсами в интерфейсе
                status.update(label="Ошибка", state="error")
                st.error(f"Индексирование не удалось: {e}")

# ----------------------- Основной layout -----------------------

# Красивый стартовый блок с кратким описанием и быстрыми действиями.
# Появляется только если чат ещё пуст, чтобы не загромождать интерфейс во время работы.
if not st.session_state.get("messages"):
    with st.container():
        # Небольшой «hero»-блок с мягким градиентом и скруглением
        st.markdown(
            """
## Ассистент аналитики: база знаний + SQL

Этот помощник отвечает на вопросы из базы знаний и строит SQL‑запросы к вашей БД, а также может сделать интерактивные графики.

**Как формулировать запросы**
- Пишите обычным языком — ассистент сам поймет что ему делать.
- Запросы в базу данных можно уточнять, корректировать и объединять. По умолчанию асистент строит запросы на последнюю дату в таблице. 
- Можно описать как строить графики, менять их вид, раскрашивать в разные цвета.


**Примеры**
Запрос к базе знаний:
```
Расскажи какие таблицы доступны и что в них содержится.
```

Простой SQL‑запрос:
```
Сделай топ 10 городов по оплатам мобильного приложения.
```
Простой график:
```
Построй график по последнему запросу.
```
"""
        )

# Предупреждения о пропущенных блоках промптов (если есть)
_prompts_map, _prompts_warn = _reload_prompts()
if _prompts_warn:
    st.warning("В `prompts.py` отсутствуют: " + ", ".join(_prompts_warn))

# Рендер существующей истории чата
if st.session_state["messages"]:
    for i, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                # >>> все результаты, привязанные к этому ответу
                for item in st.session_state["results"]:
                    if item.get("msg_idx") == i:
                        _render_result(item)


# Поле ввода внизу
user_input = st.chat_input("Введите запрос…")

# ----------------------- Обработка нового запроса -----------------------

if user_input:
    # 0) Сохраняем и показываем реплику пользователя
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompts_map, _ = _reload_prompts()  # >>> Горячая подгрузка актуальных блоков

    # 1) Маршрутизация: ждём ровно ```mode ...``` где в тексте sql|rag|plotly
    router_msgs = (
        [{"role": "system", "content": _tables_index_hint()}] +  # <<< новая «память» по таблицам
        [{"role": "system", "content": prompts_map["router"]}] +
        st.session_state["messages"]
    )
    try:
        route = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=router_msgs,
            temperature=0.0,
        ).choices[0].message.content
    except Exception as e:
        route = "```mode\nsql\n```"
        st.warning(f"Роутер недоступен, переключаюсь в 'sql': {e}")

    m_mode = re.search(r"```mode\s*(.*?)```", route, re.DOTALL | re.IGNORECASE)
    mode = (m_mode.group(1).strip() if m_mode else "sql").lower()

    if mode not in {"sql", "rag", "plotly", "catalog"}:
        mode = "sql"  # >>> на случай 'pivot' или другого не реализованного режима

    final_reply = ""

    if mode == "catalog":
        # Детерминированный каталог из КУРАТОРСКИХ файлов без LLM
        text = user_input.lower()
        want_tables = any(w in text for w in ["таблиц", "таблица", "tables", "table"])
        want_dash = any(w in text for w in ["дашборд", "дашборды", "dashboard", "dashboards", "datalens"])
        if not (want_tables or want_dash):
            want_tables = True
            want_dash = True

        out = []
        if want_dash:
            dashboards_md = _read_text_file_quiet(CATALOG_DASHBOARDS_FILE)
            out.append("### Дашборды\n" + (dashboards_md.strip() or "—"))
        if want_tables:
            tables_md = _read_text_file_quiet(CATALOG_TABLES_FILE)
            out.append("### Таблицы\n" + (tables_md.strip() or "—"))

        final_reply = "\n\n".join(out) if out else "Каталог пуст."
        st.session_state["messages"].append({"role": "assistant", "content": final_reply})
        with st.chat_message("assistant"):
            st.markdown(final_reply)
        st.stop()

    # 2) Выполнение по выбранному режиму
    if mode == "rag":
        # 2a) Просим краткий RAG-запрос (блок ```rag ...```)
        rag_msgs = [{"role": "system", "content": prompts_map["rag"]}] + st.session_state["messages"]
        try:
            rag_draft = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=rag_msgs,
                temperature=0.2,
            ).choices[0].message.content
        except Exception as e:
            rag_draft = ""
            st.error(f"Не удалось получить RAG-запрос: {e}")

        m_rag = re.search(r"```rag\s*(.*?)```", rag_draft, re.DOTALL | re.IGNORECASE)
        rag_query = (m_rag.group(1).strip() if m_rag else "")

        hits = []
        if rag_query:
            try:
                # Используем улучшенный поиск для RAG-режима
                hits = _enhanced_table_search(
                    rag_query, 
                    chroma_path=CHROMA_PATH,
                    collection_name=COLLECTION_NAME
                )
            except Exception as e:
                st.warning(f"Не удалось получить контекст из базы знаний: {e}")

        context = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])

        # если нашли непустой контекст — сохраняем в кэш; если пусто — берём последний удачный
        if context.strip():
            st.session_state["last_rag_ctx"] = context
        else:
            cached = st.session_state.get("last_rag_ctx", "")
            if cached:
                context = cached

        # 2b) Финальный ответ/SQL с учётом контекста
        exec_msgs = (
            [{"role": "system", "content": _tables_index_hint()}]
            + [
                {"role": "system", "content": prompts_map["sql"]},
                {"role": "system", "content": "Если пользователь просит визуализацию (график/диаграмму), верни сразу после блока ```sql``` ещё и блок ```plotly```."},
                {"role": "system", "content": prompts_map["plotly"]},
            ]
            + st.session_state["messages"]
            + [{"role": "system", "content": f"Контекст базы знаний:\n{context}\nИнструкции: строго придерживайся контексту. Если нужных таблиц нет — скажи об этом и не пиши SQL."}]
        )
        try:
            final_reply = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=exec_msgs,
                temperature=0.2,
            ).choices[0].message.content
        except Exception as e:
            final_reply = "Не удалось получить ответ в режиме RAG."
            st.error(f"Ошибка на шаге ответа (RAG): {e}")

    elif mode == "sql":
        exec_msgs = (
            [{"role": "system", "content": _tables_index_hint()}]
            + [
                {"role": "system", "content": prompts_map["sql"]},
                # Небольшой мост: объясняем, что график — в той же реплике
                {"role": "system", "content": "Если пользователь просит визуализацию (график/диаграмму), верни сразу после блока ```sql``` ещё и блок ```plotly```."},
                {"role": "system", "content": prompts_map["plotly"]},
            ]
            + st.session_state["messages"]
        )
        try:
            final_reply = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=exec_msgs,
                temperature=0.2,
            ).choices[0].message.content
        except Exception as e:
            final_reply = "Не удалось получить ответ в режиме SQL."
            st.error(f"Ошибка на шаге ответа (SQL): {e}")

    

    elif mode == "plotly":
         # 333-Новая: передаём модели подсказку со списком доступных колонок и их типами
        cols_hint_msg = []
        try:
            if st.session_state.get("last_df") is not None:
                _pdf = st.session_state["last_df"].to_pandas()
                # Соберём компактную справку по колонкам для системы
                cols_hint_text = "Доступные столбцы и типы:\n" + "\n".join(
                    [f"- {c}: {str(_pdf[c].dtype)}" for c in _pdf.columns]
                )
                cols_hint_msg = [{"role": "system", "content": cols_hint_text}]
        except Exception:
            # Если по каким-то причинам last_df ещё не готов — просто не добавляем подсказку
            cols_hint_msg = []

        exec_msgs = (
            [{"role": "system", "content": prompts_map["plotly"]}]
            + cols_hint_msg
            + st.session_state["messages"]
        )
        try:
            final_reply = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=exec_msgs,
                temperature=0.2,
            ).choices[0].message.content
        except Exception as e:
            final_reply = "Не удалось получить код графика."
            st.error(f"Ошибка на шаге ответа (Plotly): {e}")

    # 3) Публикуем ответ ассистента в чат и сохраняем в историю
    st.session_state["messages"].append({"role": "assistant", "content": final_reply})
    # индекс этого сообщения ассистента (нужен для привязки результатов)
    st.session_state["last_assistant_idx"] = len(st.session_state["messages"]) - 1
    with st.chat_message("assistant"):
        # Не показываем служебные блоки title/explain/sql — они теперь рендерятся у таблицы
        cleaned = _strip_llm_blocks(final_reply)
        if cleaned:
            st.markdown(cleaned)
        created_chart = False
        created_table = False

        # 4) Если ассистент вернул SQL — выполняем ClickHouse и сохраняем таблицу
        m_sql = re.search(r"```sql\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        if m_sql:
            sql = m_sql.group(1).strip()
            # Удаляем только симметричные обёртки кавычками/бэктиками, не трогая внутренние `...`
            if len(sql) >= 2 and sql[0] in {'`', "'", '"'} and sql[-1] == sql[0]:
                sql = sql[1:-1].strip()
            # Пытаемся вытащить дополнительные блоки:
            m_title = re.search(r"```title\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
            m_explain = re.search(r"```explain\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
            meta_extra = {
                # В meta пойдёт ИСПОЛЬЗОВАННЫЙ SQL; исходный сохраним отдельно
                "sql": None,
                "sql_original": sql,
                "title": (m_title.group(1).strip() if m_title else None),
                "explain": (m_explain.group(1).strip() if m_explain else None),
            }
            try:
                ch = ClickHouse_client()
                df_any, used_sql = run_sql_with_auto_schema(
                    sql_text=sql,                        # ваш первоначальный SQL
                    base_messages=st.session_state["messages"],  # те же сообщения, что вы даёте модели
                    ch_client=ch,       # ваш клиент ClickHouse
                    llm_client=client,                   # ваш OpenAI-клиент
                    prompts_map=prompts_map,             # ваши системные промпты
                    model_name=OPENAI_MODEL              # имя модели
                )
                # Обновим meta и общий state, чтобы заголовок/SQL подсасывались при отрисовке
                meta_extra["sql"] = used_sql
                st.session_state["last_sql_meta"] = dict(meta_extra)
                if used_sql.strip() != sql.strip():
                    st.info("SQL был автоматически скорректирован по схеме (отсутствующие поля/алиасы исправлены).")
                if isinstance(df_any, pl.DataFrame):
                    df_pl = df_any
                else:
                    # на всякий случай: если драйвер вернул pandas
                    df_pl = pl.from_pandas(df_any) if isinstance(df_any, pd.DataFrame) else None

                st.session_state["last_df"] = df_pl
                if df_pl is not None:
                    # Сохраняем таблицу. Не генерируем/не показываем дополнительный Plotly go.Table-график.
                    meta_table = dict(meta_extra)
                    _push_result("table", df_pl=df_pl, meta=meta_table)
                    _render_result(st.session_state["results"][-1])
                    created_table = True
                else:
                    st.error("Драйвер вернул неожиданный формат данных.")
            except Exception as e:
                # Показать краткую ошибку и обязательно — заголовок/SQL, чтобы пользователь видел, что именно выполнялось
                st.session_state["last_sql_meta"] = dict(meta_extra)
                st.error(f"Ошибка выполнения SQL: {e}")
                title = (meta_extra.get("title") or "Результаты запроса").strip()
                explain = (meta_extra.get("explain") or "").strip()
                if title:
                    st.markdown(f"### {title}")
                if explain:
                    st.caption(explain)
                with st.expander("Показать SQL", expanded=False):
                    orig_sql = (meta_extra.get("sql_original") or "").strip()
                    if orig_sql and orig_sql != st.session_state.get("last_sql_meta", {}).get("sql", ""):
                        st.markdown("**Исходный SQL от модели**")
                        st.code(orig_sql, language="sql")

        # Убрали обработку блока table — режим упразднён

        # 5) Если ассистент вернул Plotly-код — исполняем его в песочнице и сохраняем график
        m_plotly = re.search(r"```plotly\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        m_python = re.search(r"```python\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        plotly_code = (m_plotly.group(1) if m_plotly else (m_python.group(1) if m_python else "")).strip()

        # Применение стилей к Streamlit-таблице из go.Table кода — всегда (даже если таблица уже создана)
        if plotly_code and re.search(r"go\.Table\(", plotly_code):
            try:
                m_hdr0 = re.search(r"header\s*=\s*dict\([^)]*?fill_color\s*=\s*([\"'])\s*([^\"']+)\s*\1", plotly_code, re.IGNORECASE | re.DOTALL)
                m_cells0 = re.search(r"cells\s*=\s*dict\([^)]*?fill_color\s*=\s*([\"'])\s*([^\"']+)\s*\1", plotly_code, re.IGNORECASE | re.DOTALL)
                hdr_color0 = (m_hdr0.group(2).strip() if m_hdr0 else None)
                cell_color0 = (m_cells0.group(2).strip() if m_cells0 else None)
                if hdr_color0 or cell_color0:
                    for it in reversed(st.session_state.get("results", [])):
                        if it.get("kind") == "table" and isinstance(it.get("df_pl"), pl.DataFrame):
                            meta_it = it.get("meta") or {}
                            meta_it["table_style"] = {"header_fill_color": hdr_color0, "cells_fill_color": cell_color0, "align": "left"}
                            it["meta"] = meta_it
                            try:
                                st.rerun()
                            except Exception:
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                            break
            except Exception:
                pass

        # Новый лёгкий формат для стилизации: блок ```table_style```
        m_tstyle = re.search(r"```table_style\s*([\s\S]*?)```", final_reply, re.IGNORECASE)
        if m_tstyle:
            try:
                block = m_tstyle.group(1)
                hdr_color1 = None
                cell_color1 = None
                align1 = None
                for line in block.splitlines():
                    if "header_fill_color" in line:
                        hdr_color1 = line.split(":", 1)[-1].strip().strip('"\'')
                    elif "cells_fill_color" in line:
                        cell_color1 = line.split(":", 1)[-1].strip().strip('"\'')
                    elif re.search(r"\balign\b", line):
                        align1 = line.split(":", 1)[-1].strip().strip('"\'')
                
                # ИСПРАВЛЕНИЕ: Сохраняем стили для новых таблиц
                if hdr_color1 or cell_color1 or align1:
                    style_data = {
                        "header_fill_color": hdr_color1, 
                        "cells_fill_color": cell_color1, 
                        "align": align1 or "left"
                    }
                    st.session_state["next_table_style"] = style_data
                    st.success(f"🎨 Стили сохранены для следующей таблицы: {style_data}")
                    # НЕ применяем к существующим таблицам - только сохраняем для новых
            except Exception:
                pass

        if plotly_code and not (created_chart or created_table):
            if st.session_state["last_df"] is None:
                st.info("Нет данных для графика: выполните SQL, чтобы получить df.")
            else:
                code = plotly_code  # берём уже извлечённый текст из ```plotly или ```python

                # Если это go.Table — не строим график (таблица уже применена как стиль)
                if re.search(r"go\.Table\(", code):
                    code = ""

                # Базовая защита: не допускаем опасные конструкции
                BANNED_RE = re.compile(
                    r"(?:\bopen\b|\bexec\b|\beval\b|subprocess|socket|"
                    r"os\.[A-Za-z_]+|sys\.[A-Za-z_]+|Path\(|write\(|remove\(|unlink\(|requests|httpx)",
                    re.IGNORECASE,
                )
                # >>> удалим любые import/from и комментарии/тройные строки
                code_clean = code
                code_clean = re.sub(r"(?m)^\s*(?:from\s+\S+\s+import\s+.*|import\s+.*)\s*$", "", code_clean)
                code_scan = code_clean
                # многострочные ''' ... ''' и """ ... """
                code_scan = re.sub(r"'''[\s\S]*?'''", "", code_scan)
                code_scan = re.sub(r'"""[\s\S]*?"""', "", code_scan)
                # однострочные комментарии: # ...
                code_scan = re.sub(r"(?m)#.*$", "", code_scan)

                if not code:
                    # ничего не рисуем и не шумим
                    pass
                elif BANNED_RE.search(code_scan):
                    st.error("Код графика отклонён (запрещённые конструкции).")
                else:
                    try:
                        pdf = st.session_state["last_df"].to_pandas()
                        # --- Хелперы проверки колонок ---
                        def col(*names):
                            """
                            Вернёт первое подходящее имя колонки из перечисленных.
                            Если ни одно не найдено — поднимет понятную ошибку.
                            """
                            for n in names:
                                if isinstance(n, str) and n in pdf.columns:
                                    return n
                            raise KeyError(f"Нет ни одной из колонок {names}. Доступны: {list(pdf.columns)}")

                        def has_col(name: str) -> bool:
                            return isinstance(name, str) and name in pdf.columns

                        COLS = list(pdf.columns)  # можно подсветить пользователю доступные имена

                        safe_globals = {
                            "__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list},
                            "pd": pd,
                            "px": px,
                            "go": go,
                            "df": pdf,   # исходные данные (только чтение)
                            "col": col,  # <<< добавили
                            "has_col": has_col,
                            "COLS": COLS,
                        }
                        local_vars = {}
                        exec(code_clean, safe_globals, local_vars)

                        fig = local_vars.get("fig")
                        if isinstance(fig, go.Figure):
                            _push_result("chart", fig=fig, meta={"plotly_code": plotly_code})
                            _render_result(st.session_state["results"][-1])
                        else:
                            st.error("Ожидается, что код в ```plotly``` создаст переменную fig (plotly.graph_objects.Figure).")
                    except Exception as e:
                        # Если ошибка связана с выбором колонок (наш helper col(...) кинул KeyError),
                        # попробуем один автоматический ретрай: напомним модели доступные колонки.
                        err_text = str(e)
                        needs_retry = isinstance(e, KeyError) or "Нет ни одной из колонок" in err_text

                        if needs_retry:
                            try:
                                _pdf = st.session_state["last_df"].to_pandas()
                                _cols_list = list(_pdf.columns)
                                retry_hint = (
                                    "Ошибка выбора колонок при построении графика: "
                                    + err_text
                                    + "\nДоступные колонки: "
                                    + ", ".join(map(str, _cols_list))
                                    + "\nСгенерируй НОВЫЙ код для переменной fig, используя ТОЛЬКО эти имена через col(...)."
                                )

                                retry_msgs = (
                                    [{"role": "system", "content": prompts_map["plotly"]},
                                    {"role": "system", "content": retry_hint}]
                                    + st.session_state["messages"]
                                )
                                retry_reply = client.chat.completions.create(
                                    model=OPENAI_MODEL,
                                    messages=retry_msgs,
                                    temperature=0,
                                ).choices[0].message.content

                                # Повторно ищем блок ```plotly``` и пытаемся исполнить
                                m_plotly_retry = re.search(r"```plotly\s*(.*?)```", retry_reply, re.DOTALL | re.IGNORECASE)
                                if m_plotly_retry:
                                    code_retry = m_plotly_retry.group(1).strip()

                                    # Повторная базовая проверка безопасности + удалим любые import/from
                                    code_retry_clean = re.sub(r"(?m)^\s*(?:from\s+\S+\s+import\s+.*|import\s+.*)\s*$", "", code_retry)
                                    code_scan2 = re.sub(r"'''[\s\S]*?'''", "", code_retry_clean)
                                    code_scan2 = re.sub(r'"""[\s\S]*?"""', "", code_scan2)
                                    code_scan2 = re.sub(r"(?m)#.*$", "", code_scan2)
                                    if BANNED_RE.search(code_scan2):
                                        st.error("Код графика (повтор) отклонён (запрещённые конструкции).")
                                    else:
                                        # Выполняем повторный код в том же «песочном» окружении
                                        # Собираем такое же безопасное окружение, как в первом запуске
                                        safe_globals_retry = {
                                            "__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list},
                                            "pd": pd,
                                            "px": px,
                                            "go": go,
                                            "df": pdf,      # данные только для чтения
                                            "col": col,
                                            "has_col": has_col,
                                            "COLS": COLS,
                                        }
                                        local_vars = {}
                                        exec(code_retry_clean, safe_globals_retry, local_vars)
                                        fig = local_vars.get("fig")

                                        if isinstance(fig, go.Figure):
                                            _push_result("chart", fig=fig, meta={"plotly_code": code_retry})
                                            _render_result(st.session_state["results"][-1])
                                        else:
                                            st.error("Повтор: код не создал переменную fig (plotly.graph_objects.Figure).")
                                else:
                                    st.error("Повтор: ассистент не вернул блок ```plotly```.")
                            except Exception as e2:
                                st.error(f"Повтор также не удался: {e2}")
                        else:
                            st.error(f"Ошибка выполнения кода графика: {e}")

# --- Кнопка скачивания архива В САМОМ НИЗУ ---
# ВАЖНО: размещена после обработки user_input, SQL/Plotly и _push_result(...),
# поэтому архив собирается с учётом самых свежих результатов уже на первом клике.
if st.session_state["results"]:
    st.divider()
st.download_button(
    "Скачать историю чата (zip)",
    data=_history_zip_bytes(),
    file_name="history.zip",
    mime="application/zip",
    disabled=(len(st.session_state["results"]) == 0),
)
