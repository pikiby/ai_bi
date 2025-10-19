# app.py
# Полноценное приложение Streamlit: динамическая подгрузка промптов, роутер режимов,
# SQL (ClickHouse), RAG, безопасный Plotly, история результатов и экспорт.

import os
import re
import json
import io
import zipfile
from datetime import datetime

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
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
# Обязательное подтверждение плана перед SQL/PIVOT (можно отключить флагом)
AUTO_PLAN_REQUIRED = os.getenv("AUTO_PLAN_REQUIRED", "1") == "1"

# --- Флаг сохранений (только на операции сохранения) ---
# По требованию: флаг влияет ТОЛЬКО на отображение кнопок "Сохранить ..." и INSERT.
# Чтение/запуск/переименование/удаление доступны всегда.
SAVED_QUERIES_ENABLED = os.getenv("SAVED_QUERIES_ENABLED", "1") == "1"

# --- Общий каталог пользователя (нулевой UUID) ---
COMMON_USER_UUID = "00000000-0000-0000-0000-000000000000"

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
# модель LLM по умолчанию (для режимов, где выбирается модель из состояния)
if "model" not in st.session_state:
    st.session_state["model"] = OPENAI_MODEL

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

# служебные поля для диагностики маршрутизации
if "last_mode" not in st.session_state:
    st.session_state["last_mode"] = None
if "mode_source" not in st.session_state:
    st.session_state["mode_source"] = "router"
if "mode_history" not in st.session_state:
    st.session_state["mode_history"] = []
if "last_router_hint" not in st.session_state:
    st.session_state["last_router_hint"] = None

# --- Широкий режим интерфейса (переключатель рядом со строкой ввода) ---
if "wide_mode" not in st.session_state:
    st.session_state["wide_mode"] = False

def _apply_wide_mode_css():
    """Применяет CSS для растягивания контейнера на всю ширину при активном флаге wide_mode.
    Нельзя менять layout через set_page_config на лету, поэтому используем CSS."""
    if st.session_state.get("wide_mode"):
        st.markdown(
            """
            <style>
            .block-container {max-width: 100%; padding-left: 1.5rem; padding-right: 1.5rem;}
            </style>
            """,
            unsafe_allow_html=True,
        )

_apply_wide_mode_css()

# ======================== Saved Queries: helpers ========================

def _save_current_result(kind: str, item: dict):
    """Сохранение текущего результата как элемента каталога сохранённых запросов.
    Сохраняем сырые строки: sql_code, table_code/plotly_code.
    kind ∈ {"table", "chart"}.
    """
    if not SAVED_QUERIES_ENABLED:
        return False, "Сохранение отключено"
    try:
        import uuid
        ch = ClickHouse_client()
        meta = item.get("meta") or {}
        # Берём исходный SQL от модели, иначе использованный SQL
        sql_code = (meta.get("sql_original") or meta.get("sql") or "").strip()
        # Для графиков часто SQL лежит в last_sql_meta, т.к. meta у графика может не содержать SQL
        if not sql_code and kind == "chart":
            last_meta = st.session_state.get("last_sql_meta") or {}
            sql_code = (last_meta.get("sql_original") or last_meta.get("sql") or "").strip()
        table_code = (meta.get("table_code") or "").strip() if kind == "table" else ""
        plotly_code = (meta.get("plotly_code") or "").strip() if kind == "chart" else ""
        if not sql_code:
            return False, "Нет SQL-кода для сохранения"
        title_suggest = None
        try:
            if kind == "table":
                df_pl = item.get("df_pl")
                pdf = df_pl.to_pandas() if df_pl is not None else None
                title_suggest = _get_title(meta, pdf, "sql")
            else:
                title_suggest = (meta.get("title") or "Результаты запроса").strip()
        except Exception:
            title_suggest = "Мой запрос"
        # Уникальные ключи виджетов, чтобы не конфликтовали между несколькими результатами
        key_base = f"{kind}_{item.get('ts','')}_{item.get('msg_idx','')}"
        with st.popover("Сохранить " + ("таблицу" if kind == "table" else "график")):
            t = st.text_input("Название", value=title_suggest or "Мой запрос", key=f"sq_name_{key_base}")
            if st.button("Сохранить", use_container_width=True, key=f"sq_btn_{key_base}"):
                try:
                    ch.insert_saved_query(
                        user_uuid=COMMON_USER_UUID,
                        item_uuid=str(uuid.uuid4()),
                        title=t.strip() or "Без названия",
                        db="",
                        sql_code=sql_code,
                        table_code=table_code,
                        plotly_code=plotly_code,
                        pivot_code=(meta.get("pivot_code") or ""),
                    )
                    st.success("Сохранено")
                    st.session_state.pop("_saved_queries_cache", None)
                    # Обновим сайдбар немедленно
                    st.rerun()
                    return True, "OK"
                except Exception as e:
                    st.error(f"Ошибка сохранения: {e}")
                    return False, str(e)
        return False, None
    except Exception as e:
        return False, str(e)


def _render_saved_queries_sidebar():
    """Сайдбар со списком сохранённых запросов: поиск по названию,
    запуск по клику, подменю "..." с переименованием и удалением."""
    st.sidebar.markdown("**Сохранённые запросы**")
    search = st.sidebar.text_input("Поиск по названию", key="sq_search", placeholder="Начните вводить...")
    ch = ClickHouse_client()
    rows = ch.list_saved_queries(COMMON_USER_UUID, search_text=search or None)
    for row in rows:
        item_uuid = row.get("item_uuid")
        title = row.get("title") or "Без названия"
        col_btn, col_more = st.sidebar.columns([10, 2])
        with col_btn:
            if st.button(title, key=f"sq_run_{item_uuid}", use_container_width=True):
                # Триггерим запуск вне сайдбара, чтобы рендер шёл в основной чат
                st.session_state["__sq_run__"] = item_uuid
        with col_more:
            with st.popover("…"):
                new_title = st.text_input("Переименовать", value=title, key=f"sq_rename_{item_uuid}")
                if st.button("Сохранить имя", key=f"sq_rename_btn_{item_uuid}"):
                    try:
                        ch.rename_saved_query(COMMON_USER_UUID, item_uuid, new_title)
                        st.success("Переименовано")
                        st.session_state.pop("_saved_queries_cache", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                if st.button("Удалить…", key=f"sq_del_open_{item_uuid}"):
                    st.session_state["_sq_confirm_delete"] = {"uuid": item_uuid, "title": title}
    _show_delete_modal_if_needed()


def _show_delete_modal_if_needed():
    data = st.session_state.get("_sq_confirm_delete")
    if not data:
        return
    item_uuid = data.get("uuid")
    title = data.get("title")
    @st.dialog(f"Удалить «{title}»?")
    def _confirm_delete_dialog():
        st.write("Действие нельзя отменить. Удалить элемент?")
        col_ok, col_cancel = st.columns(2)
        with col_ok:
            if st.button("Удалить", type="primary"):
                try:
                    ch = ClickHouse_client()
                    ch.soft_delete_saved_query(COMMON_USER_UUID, item_uuid)
                    st.success("Удалено")
                    st.session_state.pop("_saved_queries_cache", None)
                    st.session_state.pop("_sq_confirm_delete", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        with col_cancel:
            if st.button("Отмена"):
                st.session_state.pop("_sq_confirm_delete", None)
    _confirm_delete_dialog()


def _run_saved_item(item_uuid: str):
    """Выполнить сохранённый элемент: всегда сначала SQL → затем table_code/plotly_code.
    Результат добавляется в чат теми же функциями, что и обычно."""
    try:
        ch = ClickHouse_client()
        rec = ch.get_saved_query(COMMON_USER_UUID, item_uuid)
        if not rec:
            st.error("Элемент не найден или удалён")
            return
        sql = (rec.get("sql_code") or "").strip()
        if not sql:
            st.error("У элемента отсутствует SQL-код")
            return
        # Публикуем системное ассистентское сообщение, к которому привяжем результаты
        title = rec.get("title") or "Сохранённый запрос"
        assist_text = f"Запуск сохранённого запроса: {title}"
        st.session_state["messages"].append({"role": "assistant", "content": assist_text})
        st.session_state["last_assistant_idx"] = len(st.session_state["messages"]) - 1

        with st.spinner("Выполняю сохранённый запрос…"):
            df_pl = ch.query_run(sql)
        st.session_state["last_df"] = df_pl
        meta = {"sql": sql, "sql_original": sql, "title": rec.get("title") or ""}
        pivot_code = (rec.get("pivot_code") or "").strip()
        table_code = (rec.get("table_code") or "").strip()
        plotly_code = (rec.get("plotly_code") or "").strip()
        # Применим сводное преобразование перед стилями/графиком, если есть
        if pivot_code:
            try:
                df_polars = df_pl
                df = df_polars.to_pandas() if isinstance(df_polars, pl.DataFrame) else df_polars
                def col(*names):
                    for nm in names:
                        if nm in df.columns:
                            return nm
                    raise KeyError(f"Нет ни одной из колонок: {names}")
                def has_col(name):
                    return name in df.columns
                COLS = list(df.columns)
                safe_globals = {"__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list},
                                "pd": pd, "df": df, "col": col, "has_col": has_col, "COLS": COLS}
                local_vars = {}
                exec(pivot_code, safe_globals, local_vars)
                new_df = local_vars.get("df")
                if isinstance(new_df, pd.DataFrame):
                    df_pl = pl.from_pandas(new_df)
                    st.session_state["last_df"] = df_pl
                    meta["pivot_code"] = pivot_code
                else:
                    st.info("pivot_code должен присвоить df новый pandas.DataFrame.")
            except Exception as e:
                st.error(f"Ошибка выполнения pivot_code: {e}")
        if table_code:
            try:
                df_polars = df_pl
                df = df_polars.to_pandas() if isinstance(df_polars, pl.DataFrame) else df_polars
                def col(*names):
                    for nm in names:
                        if nm in df.columns:
                            return df[nm]
                    raise KeyError(f"Нет ни одной из колонок: {names}")
                def has_col(name):
                    return name in df.columns
                COLS = list(df.columns)
                # Инкрементальные правки: передаём предыдущий styled_df если есть
                _prev_styler = None
                try:
                    for _it_prev in reversed(st.session_state.get("results", [])):
                        if _it_prev.get("kind") == "table":
                            _prev_styler = (_it_prev.get("meta") or {}).get("_styler_obj")
                            if _prev_styler is not None:
                                break
                except Exception:
                    _prev_styler = None

                safe_builtins = {"__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list}}
                local_vars = {"pd": pd, "df": df, "col": col, "has_col": has_col, "COLS": COLS, "styled_df": _prev_styler}
                exec(table_code, safe_builtins, local_vars)
                styled_df_obj = local_vars.get("styled_df")
                if styled_df_obj is not None and hasattr(styled_df_obj, "to_html"):
                    meta["table_code"] = table_code
                    meta["_styler_obj"] = styled_df_obj
                    try:
                        meta["rendered_html"] = styled_df_obj.to_html(escape=False, table_id="styled-table")
                    except Exception:
                        pass
                _push_result("table", df_pl=df_pl, meta=meta)
                return
            except Exception as e:
                st.error(f"Ошибка выполнения table_code: {e}")
        if plotly_code and st.session_state.get("last_df") is not None:
            try:
                pdf = df_pl.to_pandas()
                def col(*names):
                    for nm in names:
                        if nm in pdf.columns:
                            return pdf[nm]
                    raise KeyError(f"Нет ни одной из колонок: {names}")
                def has_col(name):
                    return name in pdf.columns
                COLS = list(pdf.columns)
                code_clean = re.sub(r"(?m)^\s*(?:from\s+\S+\s+import\s+.*|import\s+.*)\s*$", "", plotly_code)
                safe_globals = {
                    "__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list},
                    "pd": pd, "px": px, "go": go, "df": pdf, "col": col, "has_col": has_col, "COLS": COLS,
                }
                local_vars = {}
                exec(code_clean, safe_globals, local_vars)
                fig = local_vars.get("fig")
                if isinstance(fig, go.Figure):
                    meta["plotly_code"] = plotly_code
                    _push_result("chart", fig=fig, meta=meta)
                    return
                else:
                    st.error("Код графика должен создать переменную fig (plotly.graph_objects.Figure)")
            except Exception as e:
                st.error(f"Ошибка выполнения plotly_code: {e}")
        _push_result("table", df_pl=df_pl, meta=meta)
    except Exception as e:
        st.error(f"Ошибка запуска сохранённого запроса: {e}")

# ----------------------- Вспомогательные функции -----------------------

# КРИТИЧЕСКИ ВАЖНАЯ ФУНКЦИЯ: Горячая перезагрузка системных промптов
# ЦЕЛЬ: Позволяет изменять LLM-поведение без перезапуска приложения
# МЕСТА ИСПОЛЬЗОВАНИЯ: При каждом запросе пользователя (строка 1035), проверка предупреждений (строка 1008)
# ВАЖНОСТЬ: Обеспечивает гибкость разработки, надежность через fallback-значения, отладку через предупреждения
def _reload_prompts():
    try:
        importlib.reload(prompts)
    except ImportError as e:
        st.warning(f"Не удалось перезагрузить промпты: {e}. Используем кэшированные версии.")
    except Exception as e:
        st.warning(f"Ошибка при перезагрузке промптов: {e}. Используем кэшированные версии.")
    
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
        "sql_plan": _get(
            "RULES_SQL_PLAN",
            "Режим SQL_PLAN. Верни ровно один блок ```sql_plan``` по шаблону."
        ),
        "rag": _get(
            "RULES_RAG",
            "Режим RAG. Сначала верни блок ```rag <краткий_запрос>```, без пояснений."
        ),
        "plotly": _get(
            "RULES_PLOTLY",
            "Режим PLOTLY. Верни ровно один блок ```plotly``` с кодом, создающим переменную fig."
        ),
        "table": _get(
            "RULES_TABLE",
            "Режим TABLE. Верни ровно один блок ```table_code``` с кодом, создающим переменную styled_df (pandas Styler)."
        ),
        "pivot": _get(
            "RULES_PIVOT",
            "Режим PIVOT. Верни ровно один блок ```pivot_code```; внутри перезапиши df на результат df.pivot(...)."
        ),
        "pivot_plan": _get(
            "RULES_PIVOT_PLAN",
            "Режим PIVOT_PLAN. Верни ровно один блок ```pivot_plan``` по шаблону."
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
    # Убираем служебные блоки (table_style обрабатывается отдельно в строке 2172)
    for tag in ("title", "explain", "sql", "rag", "python", "plotly", "table", "pivot", "pivot_code"):
        text = re.sub(
            rf"```{tag}\s*.*?```",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ------------- Clarify helpers (human-facing) -------------
def _parse_plan_kv(plan_text: str) -> dict:
    """Парсит простой ключ: значение из плана LLM в словарь."""
    out = {}
    if not plan_text:
        return out
    for line in plan_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out

def _show_human_sql_clarify(plan_text: str, user_text: str):
    d = _parse_plan_kv(plan_text)
    src = d.get("source")
    date_logic = d.get("date_logic")
    metrics = d.get("metrics")
    group_by = d.get("group_by")
    ask = d.get("ask", "")
    question = ask or "Уточните метрику и трактовку: нужен топ в каждом периоде или общий?"
    with st.chat_message("assistant"):
        parts = []
        if src:
            parts.append(f"Источник: {src}")
        if date_logic:
            parts.append(f"Периодизация: {date_logic}")
        if metrics:
            parts.append(f"Метрики: {metrics}")
        if group_by:
            parts.append(f"Срез: {group_by}")
        if parts:
            st.markdown("; ".join(parts))
        st.markdown(question)
        st.caption("Ответьте одной фразой — продолжу построение запроса.")

def _show_human_pivot_clarify(plan_text: str):
    d = _parse_plan_kv(plan_text)
    idx = d.get("index")
    cols = d.get("columns")
    vals = d.get("values")
    date_fmt = d.get("date_format") or "D.M.Y"
    ask = d.get("ask", "")
    question = ask or "Уточните: строки (index), столбцы (columns), значения (values) и формат даты (по умолчанию D.M.Y)."
    with st.chat_message("assistant"):
        parts = []
        if idx:
            parts.append(f"Строки: {idx}")
        if cols:
            parts.append(f"Столбцы: {cols}")
        if vals:
            parts.append(f"Значения: {vals}")
        if date_fmt:
            parts.append(f"Формат даты: {date_fmt}")
        if parts:
            st.markdown("; ".join(parts))
        st.markdown(question)
        st.caption("Ответьте одной фразой — продолжу.")


def _last_result_hint() -> str | None:
    results = st.session_state.get("results", [])
    has_df = st.session_state.get("last_df") is not None
    last_mode = st.session_state.get("last_mode") or "unknown"

    if not results:
        df_note = "готов" if has_df else "отсутствует"
        return (
            "Контекст: результатов ещё нет. df=" + df_note + ". "
            "Если пользователь просит данные или график — сначала sql для получения таблицы." 
            "Если запрос про структуру таблиц — режим rag."
        )

    last = results[-1]
    kind = (last.get("kind") or "unknown").lower()
    df_note = "есть" if has_df else "нет"

    base = [f"Контекст: последний_результат={kind}", f"последний_режим={last_mode}", f"df={df_note}"]
    if kind == "chart":
        base.append("Если пользователь говорит про цвета, тип графика, круг/диаграмму/plot — выбирай plotly. Не выбирай table." )
        base.append("Не генерируй новый SQL без прямого запроса на новые данные.")
    elif kind == "table":
        base.append("Если просят график или визуализацию — сначала sql (если df нет), затем plotly.")
        base.append("Если просят стили таблицы — режим table.")
    else:
        base.append("Следуй явным ключевым словам пользователя; если требуется расчёт данных — sql.")

    return "; ".join(base)



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
    # st.info(f"🔍 DEBUG: _render_result вызван с kind='{kind}'")
    
    if kind == "table":
        # st.info("🔍 DEBUG: Вызываю _render_table")
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
    """
    НОВАЯ ЛОГИКА: Использует Streamlit + Pandas Styler вместо HTML+CSS
    """
    df_pl = item.get("df_pl")
    if not isinstance(df_pl, pl.DataFrame):
        return
    
    pdf = df_pl.to_pandas()
    n = _table_number_for(item)
    meta = item.get("meta") or {}
    
    title = _get_title(meta, pdf, "sql")
    st.markdown(f"**Таблица {n}:** {title}")
    
    # НОВАЯ ЛОГИКА: Используем Streamlit + Pandas Styler
    # st.info("🔍 DEBUG: Вызываю _render_table_content_styler")
    _render_table_content_styler(pdf, meta)
    _render_table_caption(meta, pdf)
    _render_sql_block(meta)
    _render_table_code(meta)
    _render_table_style_block_styler(meta)
    _render_download_buttons(pdf, item, "table")
    # Кнопка сохранения таблицы (флаг влияет только на сохранение)
    _save_current_result("table", item)


# Отрисовка графиков: координирует полную отрисовку Plotly-графиков с интерактивностью и экспортом
# АЛГОРИТМ: Валидация данных → Подготовка → Заголовок → График → Подпись → SQL → Код Plotly → Скачивание
# ИСПОЛЬЗУЕТ: _get_title(), _render_chart_caption(), _render_sql_block(), _render_plotly_code(), _render_download_buttons()
# ОСОБЕННОСТИ: Интерактивные графики с PNG-экспортом, fallback на контекст SQL, двойная документация (SQL + Plotly)
def _render_chart(item: dict):
    fig = item.get("fig")
    if not isinstance(fig, go.Figure):
        return
    
    meta = item.get("meta") or {}
    
    # Убираем лишний заголовок графика
    # title = _get_title(meta, fallback_source="context")
    # st.markdown(f"### {title}")
    
    # Уникальный ключ для графика, чтобы исключить конфликт идентификаторов при повторном рендере
    _chart_key = f"plotly_{item.get('ts','')}_{item.get('msg_idx','')}_{id(fig)}"
    st.plotly_chart(
        fig,
        theme=None,
        use_container_width=True,
        key=_chart_key,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 2}
        },
    )
    
    _render_chart_caption(meta)
    _render_sql_block(meta)
    _render_plotly_code(meta)
    _render_download_buttons(fig, item, "chart")
    # Кнопка сохранения графика (флаг влияет только на сохранение)
    _save_current_result("chart", item)


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



# Отрисовка содержимого таблицы с учетом стилей
def _render_table_content_styler(pdf: pd.DataFrame, meta: dict):
    """Единый пайплайн: DataFrame → Styler/HTML → Streamlit.

    Приоритет:
    1) Если ассистент прислал готовый HTML в meta["rendered_html"], рендерим его сразу.
    2) Иначе применяем базовые стили через Pandas Styler и рендерим.
    """
    _save_table_dataframe(pdf, meta)

    def _enforce_table_width(fragment: str) -> str:
        try:
            idx = fragment.find('<table')
            if idx == -1:
                return fragment
            end_tag = fragment.find('>', idx)
            if end_tag == -1:
                return fragment
            tag_open = fragment[idx:end_tag]
            if 'style=' in tag_open:
                import re
                def _add_width(match):
                    styles = match.group(1)
                    if 'width' in styles:
                        return match.group(0)
                    styles = styles.rstrip()
                    if styles and not styles.endswith(';'):
                        styles += ';'
                    styles += ' width:100%;'
                    return f'style="{styles}"'
                tag_new = re.sub(r'style="([^"]*)"', _add_width, tag_open, count=1)
                return fragment[:idx] + tag_new + fragment[end_tag:]
            else:
                return fragment[:end_tag] + ' style="width:100%;"' + fragment[end_tag:]
        except Exception:
            return fragment

    ready_html = (meta.get("rendered_html") or "").strip()
    if ready_html:
        css_part, table_part = "", ready_html
        end_style = ready_html.find("</style>")
        if end_style != -1:
            end = end_style + len("</style>")
            css_part = ready_html[:end]
            table_part = ready_html[end:]
        table_part = _enforce_table_width(table_part)
        mask_open = "<div style=\"max-height:520px; overflow:auto; border-radius:10px;\">"
        st.markdown(css_part + mask_open + table_part + "</div>", unsafe_allow_html=True)
        return

    defaults = {
        "header_fill_color": "#f4f4f4",
        "header_font_color": "black",
        "cells_fill_color": "transparent",
        # Наследуем цвет текста от темы (чтобы тёмная/светлая работали автоматически)
        "font_color": "inherit",
    }
    user_cfg = meta.get("styler_config") or {}
    cfg = {**defaults, **user_cfg}

    styler = pdf.style
    
    # Применяем стили для HTML (через селекторы CSS)
    styler = styler.set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", cfg.get("header_fill_color")),
            ("color", cfg.get("header_font_color")),
            ("font-weight", "bold"),
        ]},
        {"selector": "tbody td", "props": [
            ("background-color", cfg.get("cells_fill_color")),
            ("color", cfg.get("font_color")),
        ]},
    ])
    
    # Создаём отдельный styler для Excel с применением стилей к ячейкам
    # (set_table_styles НЕ работает для Excel, нужен apply/applymap)
    styler_for_excel = pdf.style
    
    # Применяем стили к ячейкам данных (для Excel)
    cells_bg = cfg.get("cells_fill_color", "transparent")
    if cells_bg and cells_bg != "transparent":
        styler_for_excel = styler_for_excel.applymap(
            lambda x: f'background-color: {cells_bg}'
        )
    
    # Применяем стили к заголовкам через set_table_styles (работает для Excel)
    header_bg = cfg.get("header_fill_color", "#f4f4f4")
    header_color = cfg.get("header_font_color", "black")
    styler_for_excel = styler_for_excel.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", header_bg),
            ("color", header_color),
            ("font-weight", "bold"),
        ]},
    ], overwrite=False)
    
    # Сохраняем styler для Excel-экспорта
    meta["_styler_obj"] = styler_for_excel

    html = styler.to_html(escape=False, table_id="styled-table")

    # Встроенный CSS Styler может оказаться внутри контейнера и ломать разметку.
    # Вынесем <style> отдельно: css_part (если есть) + сама таблица.
    css_part, table_part = "", html
    end_style = html.find("</style>")
    if end_style != -1:
        end = end_style + len("</style>")
        css_part = html[:end]
        table_part = html[end:]

    table_part = _enforce_table_width(table_part)

    mask_open = "<div style=\"max-height:520px; overflow:auto; border-radius:10px;\">"
    st.markdown(css_part + mask_open + table_part + "</div>", unsafe_allow_html=True)


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
    
    # Блок с SQL (сверху)
    with st.expander("Показать SQL", expanded=False):
        if used_sql:
            st.markdown("**Использованный SQL**")
            st.code(used_sql, language="sql")
            if orig_sql and orig_sql != used_sql:
                st.markdown("**Исходный SQL от модели**")
                st.code(orig_sql, language="sql")
        elif orig_sql:
            st.code(orig_sql, language="sql")


# Отрисовка блока стилей таблицы
def _render_table_style_block_styler(meta: dict):
    """НОВАЯ ЛОГИКА: Отрисовка блока со стилями Pandas Styler"""
    styler_config = meta.get("styler_config", {})
    
    if not styler_config:
        return
    
    with st.expander("Показать стили таблицы (Pandas Styler)", expanded=False):
        st.markdown("**Стили таблицы (Pandas Styler)**")
        
        # Проверяем валидность JSON и показываем ошибки
        try:
            import json
            json_str = json.dumps(styler_config, ensure_ascii=False, indent=2)
            st.json(styler_config)
            
            # Проверяем cell_rules и row_rules на корректность
            cell_rules = styler_config.get("cell_rules", [])
            row_rules = styler_config.get("row_rules", [])
            special_rules = styler_config.get("special_rules", [])
            column_rules = styler_config.get("column_rules", [])
            row_alternating_color = styler_config.get("row_alternating_color")
            striped_rows = styler_config.get("striped_rows")
            cells_fill_color = styler_config.get("cells_fill_color")
            
            if cell_rules or row_rules or special_rules or column_rules or row_alternating_color or striped_rows or isinstance(cells_fill_color, list):
                st.markdown("**Проверка правил форматирования:**")
                
                # Предупреждение о неправильных ключах
                if column_rules:
                    st.warning("⚠️ Обнаружен устаревший ключ 'column_rules'. Используйте 'cell_rules' вместо 'column_rules'.")
                    st.info("Пример правильного формата: {\"cell_rules\": [{\"value\": \"max\", \"color\": \"red\", \"column\": \"Общая выручка\"}]}")
                
                if row_alternating_color:
                    st.warning("⚠️ Обнаружен устаревший ключ 'row_alternating_color'. Используйте 'striped': true для чередующихся строк.")
                    st.info("Пример правильного формата: {\"striped\": true}")
                
                if striped_rows:
                    st.warning("⚠️ Обнаружен устаревший ключ 'striped_rows'. Используйте 'striped': true для чередующихся строк.")
                    st.info("Пример правильного формата: {\"striped\": true}")
                
                if isinstance(cells_fill_color, list):
                    st.warning("⚠️ Обнаружен неправильный формат 'cells_fill_color' как массив. Используйте строку для цвета ячеек.")
                    st.info("Пример правильного формата: {\"cells_fill_color\": \"transparent\"}")
                
                # Проверяем cell_rules
                for i, rule in enumerate(cell_rules):
                    if not isinstance(rule, dict):
                        st.error(f"cell_rules {i+1}: должно быть словарем")
                        continue
                    
                    # Проверяем обязательные поля
                    if not rule.get("value") and not rule.get("rule"):
                        st.error(f"cell_rules {i+1}: отсутствует 'value' или 'rule'")
                    if not rule.get("color"):
                        st.error(f"cell_rules {i+1}: отсутствует 'color'")
                    
                    # Предупреждения о неправильных ключах
                    if "column_id" in rule and "column" not in rule:
                        st.warning(f"cell_rules {i+1}: используйте 'column' вместо 'column_id'")
                    if "rule" in rule and "value" not in rule:
                        st.warning(f"cell_rules {i+1}: используйте 'value' вместо 'rule'")
                
                # Проверяем row_rules
                for i, rule in enumerate(row_rules):
                    if not isinstance(rule, dict):
                        st.error(f"row_rules {i+1}: должно быть словарем")
                        continue
                    
                    # Проверяем обязательные поля
                    if not rule.get("value") and not rule.get("rule"):
                        st.error(f"row_rules {i+1}: отсутствует 'value' или 'rule'")
                    if not rule.get("color"):
                        st.error(f"row_rules {i+1}: отсутствует 'color'")
                    if not rule.get("column"):
                        st.warning(f"row_rules {i+1}: рекомендуется указать 'column' для поиска")
                    
                    st.info(f"row_rules {i+1}: будет выделена вся строка с '{rule.get('value')}' в колонке '{rule.get('column')}'")
                
                # Проверяем special_rules
                for i, rule in enumerate(special_rules):
                    if not isinstance(rule, dict):
                        st.error(f"special_rules {i+1}: должно быть словарем")
                        continue
                    
                    rule_type = rule.get("type")
                    color = rule.get("color", "red")
                    
                    if rule_type == "first_n_rows":
                        count = rule.get("count", 1)
                        st.info(f"special_rules {i+1}: будут выделены первые {count} строк цветом {color}")
                    elif rule_type == "last_n_rows":
                        count = rule.get("count", 1)
                        st.info(f"special_rules {i+1}: будут выделены последние {count} строк цветом {color}")
                    elif rule_type == "specific_row":
                        row_index = rule.get("row_index", 0)
                        st.info(f"special_rules {i+1}: будет выделена строка {row_index + 1} (индекс {row_index}) цветом {color}")
                    elif rule_type == "first_n_cols":
                        count = rule.get("count", 1)
                        st.info(f"special_rules {i+1}: будут выделены первые {count} столбцов цветом {color}")
                    elif rule_type == "last_n_cols":
                        count = rule.get("count", 1)
                        st.info(f"special_rules {i+1}: будут выделены последние {count} столбцов цветом {color}")
                    elif rule_type == "specific_col":
                        column = rule.get("column", "")
                        st.info(f"special_rules {i+1}: будет выделен столбец '{column}' цветом {color}")
                    else:
                        st.warning(f"special_rules {i+1}: неизвестный тип '{rule_type}'")
                        
        except Exception as e:
            st.error(f"Ошибка в JSON стилей: {e}")
            st.json(styler_config)


# Отрисовка кода Plotly
def _render_plotly_code(meta: dict):
    """Отрисовка свернутого блока с кодом Plotly"""
    plotly_src = (meta.get("plotly_code") or "").strip()
    if not plotly_src:
        return
    
    with st.expander("Показать код Plotly", expanded=False):
        st.code(plotly_src, language="python")

def _render_table_code(meta: dict):
    """Отрисовка свернутого блока с кодом TABLE (table_code)."""
    table_src = (meta.get("table_code") or "").strip()
    if not table_src:
        return
    with st.expander("Показать код TABLE (table_code)", expanded=False):
        st.code(table_src, language="python")


# Отрисовка кнопок скачивания
def _render_download_buttons(data, item: dict, data_type: str):
    """Отрисовка кнопок скачивания для таблиц и графиков"""
    # Уникальный суффикс для ключей виджетов: timestamp + id(item) избегает коллизий при двух результатах в одну секунду
    ts = (item.get("ts") or data_type).replace(":", "-")
    key_sfx = f"{ts}_{id(item)}"
    
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
                key=f"dl_csv_{key_sfx}",
                use_container_width=True,
            )
        with col_xlsx:
            # Пытаемся получить styler для экспорта стилей в Excel
            meta = item.get("meta", {})
            styler_obj = meta.get("_styler_obj")
            
            st.download_button(
                "Скачать XLSX",
                data=_df_to_xlsx_bytes(data, "Result", styler=styler_obj),
                file_name=f"table_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{key_sfx}",
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
                key=f"dl_html_{key_sfx}",
                use_container_width=True,
            )


# ======================== КОНЕЦ РЕФАКТОРИНГА ========================

def _df_to_csv_bytes(pdf: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pdf.to_csv(buf, index=False)
    return buf.getvalue()


def _normalize_color_for_excel(color_str: str) -> str:
    """Преобразует CSS-цвета (rgba, rgb, hex) в hex формат для Excel.
    
    Excel не понимает rgba/rgb, нужен только hex формат (#RRGGBB).
    """
    if not color_str or color_str == "transparent" or color_str == "inherit":
        return None
    
    # Если уже hex - возвращаем как есть
    if color_str.startswith("#"):
        return color_str
    
    # Преобразуем rgba(r,g,b,a) -> #RRGGBB
    import re
    rgba_match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)', color_str)
    if rgba_match:
        r, g, b = map(int, rgba_match.groups())
        return f'#{r:02x}{g:02x}{b:02x}'
    
    return color_str  # Возвращаем как есть, если не распознали

def _normalize_styler_for_excel(styler):
    """Нормализует Styler для корректного экспорта в Excel.
    
    Проблема: openpyxl может некорректно обрабатывать rgba() цвета.
    Решение: конвертируем все цвета в hex формат.
    """
    # Получаем стили, применённые к ячейкам
    ctx = styler._compute()  # Вычисляем все стили
    
    # Проходим по всем стилям и нормализуем цвета
    new_styles = []
    for style_data in ctx.ctx:
        normalized = []
        for row_styles in style_data:
            row_normalized = []
            for cell_style in row_styles:
                if cell_style:
                    # Преобразуем rgba/rgb в hex
                    new_style = cell_style
                    for color_prop in ['background-color', 'color']:
                        pattern = f'{color_prop}:\\s*([^;]+)'
                        match = re.search(pattern, new_style)
                        if match:
                            old_color = match.group(1).strip()
                            new_color = _normalize_color_for_excel(old_color)
                            if new_color:
                                new_style = new_style.replace(
                                    f'{color_prop}: {old_color}',
                                    f'{color_prop}: {new_color}'
                                )
                    row_normalized.append(new_style)
                else:
                    row_normalized.append('')
            normalized.append(row_normalized)
        new_styles.append(normalized)
    
    return styler

def _convert_styler_colors_to_hex(styler):
    """Конвертирует rgba/rgb цвета в hex для совместимости с Excel."""
    import re
    
    # Вычисляем все стили
    ctx = styler._compute()
    
    if not (hasattr(ctx, 'ctx') and ctx.ctx):
        return styler
    
    # ctx.ctx это defaultdict: {(row_idx, col_idx): [('prop', 'value'), ...]}
    # Проверяем есть ли rgba/rgb
    has_rgba = False
    for (row_idx, col_idx), styles in ctx.ctx.items():
        for prop, value in styles:
            if isinstance(value, str) and ('rgba(' in value or 'rgb(' in value):
                has_rgba = True
                break
        if has_rgba:
            break
    
    if not has_rgba:
        return styler
    
    # Функция конвертации
    def rgba_to_hex(match):
        try:
            parts = match.group(1).split(',')
            r = int(parts[0].strip())
            g = int(parts[1].strip())
            b = int(parts[2].strip())
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            return hex_color
        except:
            return '#FFFFFF'
    
    # Создаем новый styler и применяем сконвертированные стили
    df = styler.data
    new_styler = df.style
    
    # Применяем стили поячейно
    for (row_idx, col_idx), styles in ctx.ctx.items():
        converted_styles = []
        for prop, value in styles:
            if isinstance(value, str) and ('rgba(' in value or 'rgb(' in value):
                new_value = re.sub(r'rgba?\(([^)]+)\)', rgba_to_hex, value)
                converted_styles.append(f'{prop}: {new_value}')
            else:
                converted_styles.append(f'{prop}: {value}')
        
        if converted_styles:
            css_string = '; '.join(converted_styles)
            row_label = df.index[row_idx]
            col_name = df.columns[col_idx]
            
            new_styler = new_styler.apply(
                lambda x, ci=col_idx, s=css_string: 
                    [s if i == ci else '' for i in range(len(x))],
                subset=pd.IndexSlice[row_label:row_label, :],
                axis=1
            )
    
    # Копируем table_styles
    if hasattr(styler, 'table_styles') and styler.table_styles:
        new_styler.table_styles = styler.table_styles
    
    return new_styler

def _df_to_xlsx_bytes(pdf: pd.DataFrame, sheet_name: str = "Sheet1", styler=None) -> bytes:
    """Экспорт DataFrame в Excel с опциональной поддержкой стилей через Styler.
    
    Args:
        pdf: pandas DataFrame для экспорта
        sheet_name: имя листа Excel
        styler: опциональный pandas.Styler со стилями (если None, экспортируется без стилей)
    """
    buf = io.BytesIO()
    
    if styler is not None and hasattr(styler, 'to_excel'):
        # Экспорт со стилями через Styler.to_excel()
        try:
            # Конвертируем rgba/rgb в hex для Excel
            styler = _convert_styler_colors_to_hex(styler)
            
            # Экспорт с использованием openpyxl
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                styler.to_excel(writer, sheet_name=sheet_name, index=False)
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            # Fallback на обычный экспорт если что-то пошло не так
            import traceback
            traceback.print_exc()
            buf = io.BytesIO()  # Сброс буфера
    
    # Обычный экспорт без стилей
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pdf.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# ======================== КОМПАКТНАЯ СИСТЕМА: AI-генерация кода таблиц ========================

# СТАНДАРТНЫЕ СТИЛИ ТАБЛИЦЫ (базовый шаблон для AI)
STANDARD_TABLE_STYLES = {
    "header_fill_color": "rgba(240, 240, 240, 0.8)",  # полупрозрачный серый
    "cells_fill_color": "transparent", 
    "align": "left",
    "font_color": None,  # автоматический контрастный цвет
    "header_font_color": None  # автоматический контрастный цвет
}

def _save_table_dataframe(pdf: pd.DataFrame, meta: dict) -> str:
    """Сохраняет DataFrame для последующей генерации кода."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    table_key = f"table_{timestamp}"
    
    st.session_state[f"table_data_{table_key}"] = {
        "df": pdf,
        "meta": meta,
        "timestamp": timestamp
    }
    
    return table_key


def _generate_table_code_styler(table_key: str, user_request: str) -> str:
    """
    НОВАЯ ЛОГИКА: AI генерирует ТОЛЬКО styler_config для Pandas Styler
    """
    # Получаем данные
    table_data = st.session_state.get(f"table_data_{table_key}")
    if not table_data:
        return "❌ Данные таблицы не найдены"
    
    df = table_data["df"]
    
    # ШАБЛОН ТОЛЬКО ДЛЯ styler_config
    template = f"""
styler_config = {{
    "header_fill_color": "#f4f4f4",
    "header_font_color": "black", 
    "cells_fill_color": "white",
    "font_color": "black",
    "striped": False,
    "cell_rules": [],
    "row_rules": []
}}

# AI анализирует запрос: {user_request}
# AI добавляет нужные изменения к styler_config на основе запроса
"""
    
    return template


def _build_css_styles(style_meta: dict, unique_id: str = "adaptive-table") -> str:
    """
    Создает CSS стили на основе метаданных стиля.
    Использует только поддерживаемые Streamlit CSS свойства.
    Включает стилизованную прокрутку для таблиц.
    Поддерживает адаптацию к темной теме.
    Автоматически подбирает контрастные цвета текста.
    
    Args:
        style_meta: Словарь с метаданными стилей
        unique_id: Уникальный ID для изоляции CSS этой таблицы
    """
    header_bg = style_meta.get("header_fill_color", "rgb(240, 240, 240)")
    # Принудительно делаем заголовки непрозрачными
    if header_bg and "rgba" in header_bg:
        # Заменяем rgba на rgb, убирая альфа-канал
        header_bg = header_bg.replace("rgba", "rgb").rsplit(",", 1)[0] + ")"
    
    cell_bg = style_meta.get("cells_fill_color", "transparent")
    
    # Поддержка cells_fill_color как массива (неправильный формат)
    if isinstance(cell_bg, list) and len(cell_bg) >= 2:
        # Если это массив, активируем striped и используем первый цвет
        cell_bg = cell_bg[0] if cell_bg[0] else "transparent"
        # Также активируем striped для чередования
        if "striped" not in style_meta:
            style_meta["striped"] = True
    text_align = style_meta.get("align", "left")
    font_color = style_meta.get("font_color", None)
    header_font_color = style_meta.get("header_font_color", None)
    
    # Автоматический подбор контрастных цветов текста
    def get_contrast_color(bg_color):
        if not bg_color or bg_color == "transparent":
            return None
        # Простая эвристика: если фон светлый - темный текст, если темный - светлый
        if isinstance(bg_color, str):
            if bg_color.startswith('rgba'):
                # Извлекаем RGB значения из rgba
                import re
                match = re.search(r'rgba\((\d+),\s*(\d+),\s*(\d+)', bg_color)
                if match:
                    r, g, b = map(int, match.groups())
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    return "#000000" if brightness > 128 else "#ffffff"
            elif bg_color.startswith('#'):
                # Простая проверка для hex цветов
                hex_color = bg_color.lstrip('#')
                if len(hex_color) == 6:
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    return "#000000" if brightness > 128 else "#ffffff"
        return None
    
    # Автоматический подбор цветов текста
    auto_header_text = get_contrast_color(header_bg)
    auto_cell_text = get_contrast_color(cell_bg)
    
    # Используем автоматические цвета если не заданы явно
    final_header_font_color = header_font_color or auto_header_text or "#333333"
    final_font_color = font_color or auto_cell_text or "inherit"
    
    # Базовые стили таблицы с прокруткой и адаптацией к темной теме
    css = f"""
    .adaptive-table-container {{
        width: 100%;
        margin: 10px 0;
        overflow: auto;
        max-height: 500px;
        border: 1px solid rgba(221, 221, 221, 0.6);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: var(--background-color, #ffffff);
        color: var(--text-color, #000000);
    }}
    
    /* Адаптация к темной теме Streamlit */
    @media (prefers-color-scheme: dark) {{
        .adaptive-table-container {{
            background-color: #1e1e1e;
            color: #ffffff;
            border-color: rgba(221, 221, 221, 0.6);
        }}
        
        .adaptive-table th {{
            background-color: {header_bg};
            color: {final_header_font_color};
            border-color: rgba(221, 221, 221, 0.6);
            font-size: 0.75em;
        }}
        
        .adaptive-table th:first-child {{
            text-align: left;
        }}
        
        .adaptive-table th:not(:first-child) {{
            text-align: right;
        }}
        
        .adaptive-table td {{
            background-color: {cell_bg};
            color: {final_font_color};
            border-color: rgba(221, 221, 221, 0.6);
        }}
        
        .adaptive-table td:first-child {{
            text-align: left;
        }}
        
        .adaptive-table td:not(:first-child) {{
            text-align: right;
        }}
        
        .adaptive-table tr:nth-child(even) {{
            background-color: transparent;
        }}
        
        /* Подсветка для темной темы */
        .adaptive-table .highlight-max {{
            background-color: rgba(255, 0, 0, 0.4) !important;
            color: #ffffff !important;
            font-weight: bold;
        }}
        
        .adaptive-table .highlight-min {{
            background-color: rgba(0, 255, 0, 0.4) !important;
            color: #000000 !important;
            font-weight: bold;
        }}
        
        /* Условное форматирование для темной темы - ПОСЛЕ striped для приоритета! */
        .adaptive-table td.cell-blue,
        .adaptive-table.striped tbody tr td.cell-blue {{
            background-color: rgba(0, 0, 255, 0.4) !important;
            font-weight: bold;
        }}
        
        .adaptive-table td.cell-red,
        .adaptive-table.striped tbody tr td.cell-red {{
            background-color: rgba(255, 0, 0, 0.4) !important;
            font-weight: bold;
        }}
        
        .adaptive-table td.cell-green,
        .adaptive-table.striped tbody tr td.cell-green {{
            background-color: rgba(0, 255, 0, 0.4) !important;
            font-weight: bold;
        }}
        
        .adaptive-table td.cell-yellow,
        .adaptive-table.striped tbody tr td.cell-yellow {{
            background-color: rgba(255, 255, 0, 0.4) !important;
            font-weight: bold;
        }}
        
        .adaptive-table td.cell-orange,
        .adaptive-table.striped tbody tr td.cell-orange {{
            background-color: rgba(255, 165, 0, 0.4) !important;
            font-weight: bold;
        }}
        
        .adaptive-table td.cell-purple,
        .adaptive-table.striped tbody tr td.cell-purple {{
            background-color: rgba(128, 0, 128, 0.4) !important;
            font-weight: bold;
        }}
        
        /* Специальные стили для текста в темной теме */
        .adaptive-table .text-white {{
            color: #ffffff !important;
        }}
        
        .adaptive-table .text-black {{
            color: #000000 !important;
        }}
        
        .adaptive-table .text-red {{
            color: #ff0000 !important;
        }}
        
        .adaptive-table .text-blue {{
            color: #0000ff !important;
        }}
        
        .adaptive-table .text-green {{
            color: #008000 !important;
        }}
        
        .adaptive-table .text-yellow {{
            color: #ffff00 !important;
        }}
        
        .adaptive-table .text-orange {{
            color: #ffa500 !important;
        }}
        
        .adaptive-table .text-purple {{
            color: #800080 !important;
        }}
        
        .adaptive-table tr:hover {{
            background-color: rgba(58, 58, 58, 0.7);
        }}
        
        .adaptive-table tr:hover td {{
            color: #000000 !important;
        }}
        
        .adaptive-table tr:hover th {{
            color: #000000 !important;
        }}
        
        /* Чередующиеся строки для темной темы (идет ПЕРЕД условным форматированием) */
        .adaptive-table.striped tbody tr:nth-child(even) {{
            background-color: rgba(173, 216, 230, 0.2);
        }}
        
        .adaptive-table.striped tbody tr:nth-child(even) td {{
            background-color: rgba(173, 216, 230, 0.2);
        }}
        
        .adaptive-table.striped tbody tr:nth-child(odd) {{
            background-color: transparent;
        }}
    }}
    
    .adaptive-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0;
        font-family: Arial, sans-serif;
        min-width: 100%;
    }}
    
    .adaptive-table th {{
        background-color: {header_bg};
        color: {final_header_font_color};
        padding: 8px;
        text-align: {text_align};
        border: 1px solid rgba(221, 221, 221, 0.6);
        font-weight: bold;
        font-size: 0.75em;
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    
    .adaptive-table th:first-child {{
        text-align: left;
    }}
    
    .adaptive-table th:not(:first-child) {{
        text-align: right;
    }}
    
    .adaptive-table td {{
        padding: 6px 8px;
        border: 1px solid rgba(221, 221, 221, 0.6);
        text-align: {text_align};
        background-color: {cell_bg};
        font-size: 13px;
        color: {final_font_color};
    }}
    
    .adaptive-table td:first-child {{
        text-align: left;
    }}
    
    .adaptive-table td:not(:first-child) {{
        text-align: right;
    }}
    
    .adaptive-table tr:nth-child(even) {{
        background-color: transparent;
    }}
    
    /* Подсветка максимальных значений */
    .adaptive-table .highlight-max {{
        background-color: rgba(255, 0, 0, 0.3) !important;
        color: #000000 !important;
        font-weight: bold;
    }}
    
    .adaptive-table .highlight-min {{
        background-color: rgba(0, 255, 0, 0.3) !important;
        color: #000000 !important;
        font-weight: bold;
    }}
    
    /* Условное форматирование ячеек - ПОСЛЕ striped для приоритета! */
    .adaptive-table td.cell-blue,
    .adaptive-table.striped tbody tr td.cell-blue {{
        background-color: rgba(0, 0, 255, 0.3) !important;
        font-weight: bold;
    }}
    
    .adaptive-table td.cell-red,
    .adaptive-table.striped tbody tr td.cell-red {{
        background-color: rgba(255, 0, 0, 0.3) !important;
        font-weight: bold;
    }}
    
    .adaptive-table td.cell-green,
    .adaptive-table.striped tbody tr td.cell-green {{
        background-color: rgba(0, 255, 0, 0.3) !important;
        font-weight: bold;
    }}
    
    .adaptive-table td.cell-yellow,
    .adaptive-table.striped tbody tr td.cell-yellow {{
        background-color: rgba(255, 255, 0, 0.3) !important;
        font-weight: bold;
    }}
    
    .adaptive-table td.cell-orange,
    .adaptive-table.striped tbody tr td.cell-orange {{
        background-color: rgba(255, 165, 0, 0.3) !important;
        font-weight: bold;
    }}
    
    .adaptive-table td.cell-purple,
    .adaptive-table.striped tbody tr td.cell-purple {{
        background-color: rgba(128, 0, 128, 0.3) !important;
        font-weight: bold;
    }}
    
    /* Специальные стили для текста */
    .adaptive-table .text-white {{
        color: #ffffff !important;
    }}
    
    .adaptive-table .text-black {{
        color: #000000 !important;
    }}
    
    .adaptive-table .text-red {{
        color: #ff0000 !important;
    }}
    
    .adaptive-table .text-blue {{
        color: #0000ff !important;
    }}
    
    .adaptive-table .text-green {{
        color: #008000 !important;
    }}
    
    .adaptive-table .text-yellow {{
        color: #ffff00 !important;
    }}
    
    .adaptive-table .text-orange {{
        color: #ffa500 !important;
    }}
    
    .adaptive-table .text-purple {{
        color: #800080 !important;
    }}
    
    .adaptive-table tr:hover {{
        background-color: rgba(240, 248, 255, 0.7);
    }}
    
    .adaptive-table tr:hover td {{
        color: #000000 !important;
    }}
    
    .adaptive-table tr:hover th {{
        color: #000000 !important;
    }}
    
    /* Поддержка чередующихся строк (идет ПЕРЕД условным форматированием) */
    .adaptive-table.striped tbody tr:nth-child(even) {{
        background-color: rgba(173, 216, 230, 0.3);
    }}
    
    .adaptive-table.striped tbody tr:nth-child(even) td {{
        background-color: rgba(173, 216, 230, 0.3);
    }}
    
    .adaptive-table.striped tbody tr:nth-child(odd) {{
        background-color: transparent;
    }}
    
    /* Стилизованная прокрутка */
    .adaptive-table-container::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    .adaptive-table-container::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 4px;
    }}
    
    .adaptive-table-container::-webkit-scrollbar-thumb {{
        background: #888;
        border-radius: 4px;
    }}
    
    .adaptive-table-container::-webkit-scrollbar-thumb:hover {{
        background: #555;
    }}
    
    /* Для Firefox */
    .adaptive-table-container {{
        scrollbar-width: thin;
        scrollbar-color: #888 #f1f1f1;
    }}
    """
    
    # КРИТИЧНО: Заменяем глобальные классы на уникальные для изоляции CSS
    css = css.replace(".adaptive-table-container", f".{unique_id}-container")
    css = css.replace(".adaptive-table", f".{unique_id}")
    
    return css


def _is_style_error(style_dict: dict) -> tuple[bool, list[str]]:
    """
    Проверяет стиль на наличие неправильных ключей и форматов.
    Возвращает (есть_ли_ошибки, список_ошибок)
    """
    errors = []
    
    # Проверяем на неправильные ключи
    invalid_keys = []
    if "column_rules" in style_dict:
        invalid_keys.append("column_rules")
    if "max_value_color" in style_dict:
        invalid_keys.append("max_value_color")
    if "row_alternating_color" in style_dict:
        invalid_keys.append("row_alternating_color")
    if "striped_rows" in style_dict:
        invalid_keys.append("striped_rows")
    
    if invalid_keys:
        errors.append(f"Неправильные ключи: {', '.join(invalid_keys)}")
    
    # Проверяем cells_fill_color как массив
    if isinstance(style_dict.get("cells_fill_color"), list):
        errors.append("cells_fill_color не должен быть массивом")
    
    # Проверяем JSON синтаксис в cell_rules и row_rules
    for rule_type in ["cell_rules", "row_rules"]:
        rules = style_dict.get(rule_type, [])
        if not isinstance(rules, list):
            errors.append(f"{rule_type} должен быть массивом")
            continue
            
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"{rule_type}[{i}] должен быть словарем")
                continue
                
            # Проверяем обязательные поля
            if not rule.get("value") and not rule.get("rule"):
                errors.append(f"{rule_type}[{i}] отсутствует 'value'")
            if not rule.get("color"):
                errors.append(f"{rule_type}[{i}] отсутствует 'color'")
    
    return len(errors) > 0, errors


def normalize_table_style_with_auto_fix(style_dict: dict, llm_client=None, model_name: str = "gpt-4o-mini") -> dict:
    """
    Нормализует стиль таблицы с автоматическим исправлением ошибок.
    Аналогично run_sql_with_auto_schema, но для стилей таблиц.
    
    Args:
        style_dict: Исходный словарь стилей
        llm_client: Клиент LLM для перегенерации (опционально)
        model_name: Имя модели для перегенерации
        
    Returns:
        Нормализованный словарь стилей
    """
    import re
    
    # 0. Хелпер: расширение col_rules → special_rules (канон для движка)
    def _extend_with_col_rules(s: dict) -> dict:
        style = dict(s)
        col_rules = style.get("col_rules")
        if not col_rules or not isinstance(col_rules, list):
            return style
        special = list(style.get("special_rules", []))

        # Унифицированная сборка правил на основе effect
        def _mk_effect_rules(target_kind: str, targets, effect: dict):
            bg = (effect or {}).get("bg")
            fg = (effect or {}).get("fg", "white")
            transparent = bool((effect or {}).get("transparent"))

            out = []
            if transparent:
                # прозрачный: используем col_transparent для индексов и
                # specific_col с флагом transparent для имён
                if target_kind == "by_index":
                    out.append({"type": "col_transparent", "columns": targets})
                elif target_kind == "by_name":
                    for name in targets:
                        out.append({"type": "specific_col", "column": name, "transparent": True})
                elif target_kind == "nth":
                    for n in targets:
                        out.append({"type": "nth_col", "n": int(n), "transparent": True})
                elif target_kind == "relative":
                    left = int(targets.get("left", 0) or 0)
                    right = int(targets.get("right", 0) or 0)
                    if left > 0:
                        out.append({"type": "first_n_cols", "count": left, "transparent": True})
                    if right > 0:
                        out.append({"type": "last_n_cols", "count": right, "transparent": True})
                return out

            # цвет фона
            color = bg or "#00AAFF"
            if target_kind == "by_index":
                out.append({"type": "columns", "columns": targets, "color": color})
            elif target_kind == "by_name":
                for name in targets:
                    out.append({"type": "specific_col", "column": name, "color": color})
            elif target_kind == "nth":
                for n in targets:
                    out.append({"type": "nth_col", "n": int(n), "color": color})
            elif target_kind == "relative":
                left = int(targets.get("left", 0) or 0)
                right = int(targets.get("right", 0) or 0)
                if left > 0:
                    out.append({"type": "first_n_cols", "count": left, "color": color})
                if right > 0:
                    out.append({"type": "last_n_cols", "count": right, "color": color})
            return out

        for r in col_rules:
            if not isinstance(r, dict):
                continue
            effect = r.get("effect") or {}
            if r.get("by_name"):
                names = [c for c in r.get("by_name", []) if isinstance(c, str)]
                special += _mk_effect_rules("by_name", names, effect)
            if r.get("by_index"):
                idxs = [int(i) for i in r.get("by_index", []) if isinstance(i, int)]
                special += _mk_effect_rules("by_index", idxs, effect)
            if r.get("nth"):
                nths = [int(i) for i in r.get("nth", []) if isinstance(i, int)]
                special += _mk_effect_rules("nth", nths, effect)
            if r.get("relative"):
                rel = r.get("relative")
                if isinstance(rel, dict):
                    special += _mk_effect_rules("relative", rel, effect)

        style["special_rules"] = special
        return style

    # 1. Проверяем на ошибки
    has_errors, errors = _is_style_error(style_dict)
    
    if not has_errors:
        # Нет ошибок — вернём стиль + расширения col_rules→special_rules
        return _extend_with_col_rules(style_dict)
    
    # 2. Автоматическое исправление простых ошибок
    normalized_style = {}
    
    # Копируем правильные ключи
    for key in ["header_fill_color", "cells_fill_color", "cell_rules", "row_rules", "striped"]:
        if key in style_dict:
            normalized_style[key] = style_dict[key]
    
    # Исправляем cells_fill_color если это массив
    if isinstance(normalized_style.get("cells_fill_color"), list):
        normalized_style["cells_fill_color"] = "transparent"
        normalized_style["striped"] = True
    
    # Конвертируем column_rules в cell_rules
    if "column_rules" in style_dict:
        if "cell_rules" not in normalized_style:
            normalized_style["cell_rules"] = []
        for rule in style_dict["column_rules"]:
            if isinstance(rule, dict):
                new_rule = {}
                if "column" in rule:
                    new_rule["column"] = rule["column"]
                if "max_value_color" in rule:
                    new_rule["value"] = "max"
                    new_rule["color"] = rule["max_value_color"]
                elif "min_value_color" in rule:
                    new_rule["value"] = "min"
                    new_rule["color"] = rule["min_value_color"]
                if new_rule:
                    normalized_style["cell_rules"].append(new_rule)
    
    # Конвертируем row_alternating_color в striped
    if "row_alternating_color" in style_dict:
        normalized_style["striped"] = True
    
    # Конвертируем striped_rows в striped
    if "striped_rows" in style_dict:
        normalized_style["striped"] = True
    
    # 3. Если есть LLM клиент, пытаемся перегенерировать сложные случаи
    if llm_client and len(errors) > 2:  # Только для сложных случаев
        try:
            fix_hint = (
                "Обнаружены ошибки в JSON стилей таблицы. "
                "Исправь согласно правилам:\n"
                "- ТОЛЬКО ключи: header_fill_color, cells_fill_color, cell_rules, row_rules, striped\n"
                "- cells_fill_color ТОЛЬКО строка (НЕ массив)\n"
                "- row_rules для строк, cell_rules для ячеек\n"
                "- striped: true для чередования\n"
                "Верни только исправленный JSON."
            )
            
            # Формируем сообщения для LLM
            messages = [
                {"role": "system", "content": "Ты эксперт по JSON стилей таблиц. Исправляй ошибки в JSON."},
                {"role": "system", "content": fix_hint},
                {"role": "user", "content": f"Исходный JSON (исправь):\n{style_dict}"}
            ]
            
            response = llm_client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                temperature=0
            )
            
            fixed_content = response.choices[0].message.content
            
            # Пытаемся извлечь JSON из ответа
            import json
            try:
                # Ищем JSON в ответе
                json_match = re.search(r'\{.*\}', fixed_content, re.DOTALL)
                if json_match:
                    fixed_style = json.loads(json_match.group())
                    # Проверяем, что исправленный стиль лучше
                    fixed_has_errors, _ = _is_style_error(fixed_style)
                    if not fixed_has_errors:
                        return fixed_style
            except Exception:
                pass  # Если не удалось распарсить, используем автоматическое исправление
                
        except Exception:
            pass  # Если LLM недоступен, используем автоматическое исправление
    
    # Добавляем совместимость col_rules -> special_rules даже при автофиксе
    normalized_style = _extend_with_col_rules(normalized_style)
    return normalized_style



def _history_zip_bytes() -> bytes:
    """Собрать ZIP с историей результатов (таблицы: csv+xlsx+sql, графики: html)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(st.session_state["results"], start=1):
            base = f"{idx:03d}_{item['kind']}_{item['ts'].replace(':','-')}"
            if item["kind"] == "table" and isinstance(item.get("df_pl"), pl.DataFrame):
                pdf = item["df_pl"].to_pandas()
                zf.writestr(f"{base}.csv", _df_to_csv_bytes(pdf))
                
                # Экспорт XLSX со стилями (если есть)
                meta = item.get("meta", {})
                styler_obj = meta.get("_styler_obj")
                zf.writestr(f"{base}.xlsx", _df_to_xlsx_bytes(pdf, "Result", styler=styler_obj))
                
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
    # Защита от неправильной агрегации *_cum (кумулятивные поля нельзя суммировать/усреднять)
    try:
        if re.search(r"\b(sum|avg|count|min|max)\s*\([^)]*?_cum[^)]*\)", sql_text, flags=re.IGNORECASE):
            return False, "Нельзя агрегировать поля *_cum; для конца месяца бери снимок последнего дня без SUM."
    except Exception:
        pass
    # Запрет SQL PIVOT (должно быть клиентское pandas-преобразование)
    try:
        if re.search(r"\bPIVOT\b", sql_text, flags=re.IGNORECASE):
            return False, "Запрещён синтаксис PIVOT в SQL — сводная делается на клиенте (pandas)."
    except Exception:
        pass
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
        ql = query.lower()
        keywords = []
        if 'монетизация' in ql:
            keywords.append('монетизация партнеры')
        if 'партнер' in ql or 'партнёр' in ql:
            keywords.append('партнеры статус')
        if 'блокировка' in ql:
            keywords.append('блокировка монетизация')
        # Новые ключи для оплат/выручки/топ по городам
        if any(k in ql for k in ['оплат', 'платеж', 'платёж', 'выручк', 'revenue', 'android_pl', 'ios_pl']):
            keywords += [
                't_ai_global_report описание поля оплаты выручка',
                'таблица t_ai_global_report метрики оплаты выручка',
            ]
        if any(k in ql for k in ['город', 'cities', 'по городам', 'top', 'топ']):
            keywords.append('t_ai_global_report города метрики')
        
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

# Сайдбар: список сохранённых запросов с поиском и действиями
_render_saved_queries_sidebar()

# Если из сайдбара пришёл триггер запуска сохранённого запроса — выполняем его в основном контейнере
_pending_run = st.session_state.pop("__sq_run__", None)
if _pending_run:
    _run_saved_item(_pending_run)

# Рендер существующей истории чата
if st.session_state["messages"]:
    for i, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            # Не показываем пустые сообщения (защита от случайных пустых записей)
            if m["content"]:
                # Сначала заменим table_code/table_style на читабельные фразы, потом удалим служебные блоки
                if m["role"] == "assistant":
                    txt = m["content"]
                    txt = re.sub(r"```table_code[\s\S]*?```", "_Создаю таблицу с новыми стилями..._", txt, flags=re.IGNORECASE)
                    txt = re.sub(r"```table_style[\s\S]*?```", "_Создаю таблицу с новыми стилями..._", txt, flags=re.IGNORECASE)
                    cleaned_content = _strip_llm_blocks(txt).strip()
                    if cleaned_content:
                        st.markdown(cleaned_content)
                else:
                    st.markdown(m["content"])
            if m["role"] == "assistant":
                # >>> все результаты, привязанные к этому ответу
                for item in st.session_state["results"]:
                    if item.get("msg_idx") == i:
                        _render_result(item)


# Переключатель ширины рядом со строкой ввода
try:
    col_wide, _ = st.columns([2, 10], gap="small")
except TypeError:
    col_wide, _ = st.columns([2, 10])
with col_wide:
    st.toggle("Широкий экран", key="wide_mode")

# Поле ввода внизу
user_input = st.chat_input("Введите запрос…")

# ----------------------- Обработка нового запроса -----------------------

if user_input:
    # 0) Сохраняем и показываем реплику пользователя
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompts_map, _ = _reload_prompts()  # >>> Горячая подгрузка актуальных блоков

    # Если ждём подтверждение плана — трактуем это сообщение как ответ и принудительно выбираем режим
    pre_mode, mode_notice = (None, None)
    awaiting = st.session_state.pop("awaiting_plan", None)
    if awaiting:
        st.session_state["plan_confirmation"] = user_input
        st.session_state["plan_locked"] = awaiting.get("plan", "")
        pre_mode = awaiting.get("kind")
        mode_notice = "Использую подтверждённый план"

    # Если pre_mode выставлен (подтверждение плана) — используем его
    mode_source = "router"

    if pre_mode:
        mode = pre_mode
        mode_source = "prehook"
    else:
        # 1) Маршрутизация: ждём ровно ```mode ...``` где в тексте sql|rag|plotly
        hint = _last_result_hint()
        st.session_state["last_router_hint"] = hint
        router_msgs = (
            [{"role": "system", "content": _tables_index_hint()}]
            + ([{"role": "system", "content": hint}] if hint else [])
            + [{"role": "system", "content": prompts_map["router"]}]
            + st.session_state["messages"]
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

        if mode not in {"sql", "rag", "plotly", "catalog", "table"}:
            mode = "sql"  # >>> на случай 'pivot' или другого не реализованного режима
        mode_notice = None

    st.session_state["last_mode"] = mode
    st.session_state["mode_source"] = mode_source
    try:
        st.session_state["mode_history"].append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "user": user_input,
            "mode": mode,
            "source": mode_source,
            "hint": st.session_state.get("last_router_hint"),
        })
    except Exception:
        pass

    # Временный тех.блок: показываем выбранный режим (для отладки маршрутизации)
    try:
        if os.getenv("SHOW_MODE_DEBUG", "1") == "1":
            with st.chat_message("assistant"):
                st.caption(f"[debug] mode: {mode} • source: {mode_source}")
    except Exception:
        pass

    final_reply = ""

    if mode == "catalog":
        # Детерминированный каталог из КУРАТОРСКИХ файлов без LLM
        text = user_input.lower()
        want_tables = any(w in text for w in ["таблиц", "таблица", "tables", "table"])
        want_dash = any(w in text for w in ["дашборд", "дашборды", "dashboard", "dashboards", "datalens"])
        if not (want_tables or want_dash):
            want_tables = True
            want_dash = True

        def _clean_catalog_content(content: str) -> str:
            """Очищает содержимое каталога от лишних элементов"""
            if not content:
                return "—"
            
            # Убираем YAML front matter (строки между ---)
            content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
            
            # Убираем заголовки типа "# Таблицы (каталог)"
            content = re.sub(r'^#\s+.*?каталог.*?\n', '', content, flags=re.IGNORECASE)
            
            # Убираем вводные фразы
            content = re.sub(r'^Ниже перечислены все таблицы.*?\n', '', content, flags=re.DOTALL)
            
            # Убираем раздел "Расширенные возможности анализа" и всё после него
            content = re.sub(r'\n## Расширенные возможности анализа.*$', '', content, flags=re.DOTALL)
            
            # Убираем лишние пустые строки
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            return content.strip()
        
        out = []
        if want_dash:
            dashboards_md = _read_text_file_quiet(CATALOG_DASHBOARDS_FILE)
            clean_dash = _clean_catalog_content(dashboards_md)
            out.append("**Дашборды**\n" + clean_dash)
        if want_tables:
            tables_md = _read_text_file_quiet(CATALOG_TABLES_FILE)
            clean_tables = _clean_catalog_content(tables_md)
            out.append("**Таблицы**\n" + clean_tables)

        final_reply = "\n\n".join(out) if out else "Каталог пуст."
        st.session_state["messages"].append({"role": "assistant", "content": final_reply})
        with st.chat_message("assistant"):
            if mode_notice and mode_source == "prehook":
                st.caption(mode_notice)
            # Простой вывод без HTML-разметки
            st.markdown("**Вот что нашел:**")
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
        hint_exec = _last_result_hint()
        exec_msgs = (
            [{"role": "system", "content": _tables_index_hint()}]
            + ([{"role": "system", "content": hint_exec}] if hint_exec else [])
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
        # Шаг 0: короткий план (sql_plan) для уточнений — всегда показываем и ждём подтверждения при AUTO_PLAN_REQUIRED
        hint_exec = _last_result_hint()
        plan_msgs = (
            ([{"role": "system", "content": hint_exec}] if hint_exec else [])
            + [{"role": "system", "content": prompts_map["sql_plan"]}]
            + st.session_state["messages"]
        )
        try:
            plan_reply = client.chat.completions.create(
                model=OPENAI_MODEL, messages=plan_msgs, temperature=0
            ).choices[0].message.content
        except Exception:
            plan_reply = ""
        m_plan = re.search(r"```sql_plan\s*([\s\S]*?)```", plan_reply, re.IGNORECASE)
        if m_plan:
            plan_text = m_plan.group(1).strip()
            # Всегда требуем подтверждение при AUTO_PLAN_REQUIRED или явной двусмысленности
            needs_confirm = AUTO_PLAN_REQUIRED or bool(re.search(r"\b(по\s+месяц|по\s+дням|по\s+год|на\s+конец\s+месяц|итог\s+месяц|топ)\b", user_input, flags=re.IGNORECASE))
            if needs_confirm:
                st.session_state["awaiting_plan"] = {"kind": "sql", "plan": plan_text}
                _show_human_sql_clarify(plan_text, user_input)
                st.stop()

        # Шаг 1: генерация SQL (и при необходимости — plotly сразу после него)
        exec_msgs = (
            [{"role": "system", "content": _tables_index_hint()}]
            + ([{"role": "system", "content": hint_exec}] if hint_exec else [])
            + [
                {"role": "system", "content": prompts_map["sql"]},
                {"role": "system", "content": "Если пользователь просит визуализацию (график/диаграмму), верни сразу после блока ```sql``` ещё и блок ```plotly```."},
                {"role": "system", "content": prompts_map["plotly"]},
            ]
            + st.session_state["messages"]
        )
        try:
            # Сформируем итоговый список сообщений отдельно (проще для синтаксиса и читаемости)
            _prefix = []
            _locked = st.session_state.pop("plan_locked", "")
            if _locked:
                _prefix += [
                    {"role": "system", "content": "Используй подтверждённый sql_plan, если он задан."},
                    {"role": "system", "content": _locked},
                ]
            _confirm = st.session_state.pop("plan_confirmation", "")
            if _confirm:
                _prefix += [{"role": "system", "content": "Уточнение пользователя: " + _confirm}]
            _messages_payload = _prefix + exec_msgs

            final_reply = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=_messages_payload,
                temperature=0.2,
            ).choices[0].message.content
        except Exception as e:
            final_reply = "Не удалось получить ответ в режиме SQL."
            st.error(f"Ошибка на шаге ответа (SQL): {e}")



    elif mode == "pivot":
        # Режим PIVOT: формирование сводной таблицы (только код преобразования df)
        cols_hint_msg = []
        try:
            if st.session_state.get("last_df") is not None:
                _pdf = st.session_state["last_df"].to_pandas()
                cols_hint_text = "Доступные столбцы и типы:\n" + "\n".join(
                    [f"- {c}: {str(_pdf[c].dtype)}" for c in _pdf.columns]
                )
                cols_hint_msg = [{"role": "system", "content": cols_hint_text}]
        except Exception:
            cols_hint_msg = []

        # Шаг 0: план PIVOT — обязателен к подтверждению
        hint_exec = _last_result_hint()
        p_msgs = (
            ([{"role": "system", "content": hint_exec}] if hint_exec else [])
            + [{"role": "system", "content": prompts_map["pivot_plan"]}]
            + cols_hint_msg
            + st.session_state["messages"]
        )
        try:
            plan_reply = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=p_msgs,
                temperature=0,
            ).choices[0].message.content
        except Exception:
            plan_reply = ""
        m_pplan = re.search(r"```pivot_plan\s*([\s\S]*?)```", plan_reply, re.IGNORECASE)
        if m_pplan:
            ptext = m_pplan.group(1).strip()
            st.session_state["awaiting_plan"] = {"kind": "pivot", "plan": ptext}
            _show_human_pivot_clarify(ptext)
            st.stop()

        # Если сюда дошли по подтверждённому плану — генерируем pivot_code с учётом plan_locked/подтверждения
        # Сформируем payload для генерации pivot_code
        exec_msgs = ([{"role": "system", "content": prompts_map["pivot"]}] + cols_hint_msg + st.session_state["messages"])
        _prefix = []
        _locked = st.session_state.pop("plan_locked", "")
        if _locked:
            _prefix += [
                {"role": "system", "content": "Используй подтверждённый pivot_plan:"},
                {"role": "system", "content": _locked},
            ]
        _confirm = st.session_state.pop("plan_confirmation", "")
        if _confirm:
            _prefix += [{"role": "system", "content": "Уточнение пользователя: " + _confirm}]
        _messages_payload = _prefix + exec_msgs
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=_messages_payload,
                temperature=0.2,
            )
            final_reply = response.choices[0].message.content
        except Exception as e:
            final_reply = "Не удалось получить код PIVOT."
            st.error(f"Ошибка на шаге ответа (PIVOT): {e}")

    elif mode == "table":
        # Режим TABLE: генерация стилей для таблиц
        # Передаём модели список доступных колонок и их типы (аналогично ветке plotly)
        cols_hint_msg = []
        try:
            if st.session_state.get("last_df") is not None:
                _pdf = st.session_state["last_df"].to_pandas()
                cols_hint_text = "Доступные столбцы и типы:\n" + "\n".join(
                    [f"- {c}: {str(_pdf[c].dtype)}" for c in _pdf.columns]
                )
                cols_hint_msg = [{"role": "system", "content": cols_hint_text}]
        except Exception:
            cols_hint_msg = []

        hint_exec = _last_result_hint()
        exec_msgs = (
            ([{"role": "system", "content": hint_exec}] if hint_exec else [])
            + [{"role": "system", "content": prompts_map["table"]}]
            + cols_hint_msg
            + st.session_state["messages"]
        )

        try:
            model_name = st.session_state.get("model", OPENAI_MODEL)
            response = client.chat.completions.create(
                model=model_name,
                messages=exec_msgs,
                temperature=0.1,
            )
            final_reply = response.choices[0].message.content
        except Exception as e:
            st.error(f"Ошибка TABLE: {e}")
            st.exception(e)  # Показываем полный traceback
            final_reply = ""

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

        hint_exec = _last_result_hint()
        exec_msgs = (
            ([{"role": "system", "content": hint_exec}] if hint_exec else [])
            + [{"role": "system", "content": prompts_map["plotly"]}]
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
        # Не показываем служебные блоки title/explain/sql/table_style — они рендерятся отдельно
        # Сначала заменим table_code/table_style на читабельные фразы, затем удалим служебные блоки
        _txt = re.sub(r"```table_code[\s\S]*?```", "_Создаю таблицу с новыми стилями..._", final_reply, flags=re.IGNORECASE)
        _txt = re.sub(r"```table_style[\s\S]*?```", "_Создаю таблицу с новыми стилями..._", _txt, flags=re.IGNORECASE)
        cleaned = _strip_llm_blocks(_txt).strip()
        if cleaned:
            st.markdown(cleaned)
        created_chart = False
        created_table = False

        # 4) Если ассистент вернул SQL — выполняем ClickHouse и сохраняем таблицу
        m_sql = re.search(r"```sql\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)

        # Fallback-SQL: иногда модель пишет заглушку, но не возвращает блок ```sql```.
        # Если выбран режим sql/rag, таблицы ещё нет или нужна новая таблица, попробуем один строгий доген SQL.
        if not m_sql and mode in {"sql", "rag"}:
            try:
                strict_msgs = (
                    [{"role": "system", "content": _tables_index_hint()}]
                    + [{"role": "system", "content": "Сгенерируй СТРОГО один блок ```sql``` для запроса пользователя. Без пояснений, без текста, только код."}]
                    + [{"role": "system", "content": prompts_map["sql"]}]
                    + st.session_state["messages"]
                )
                # Если есть кэш RAG — добавим как системный контекст
                if st.session_state.get("last_rag_ctx"):
                    strict_msgs.append({"role": "system", "content": "Контекст базы знаний:\n" + st.session_state["last_rag_ctx"]})
                strict_reply = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=strict_msgs,
                    temperature=0,
                ).choices[0].message.content
                m_sql = re.search(r"```sql\s*(.*?)```", strict_reply, re.DOTALL | re.IGNORECASE)
                if m_sql:
                    # расширим исходный final_reply, чтобы ниже сохранялся sql_meta/title/explain, если появятся
                    final_reply = strict_reply
            except Exception:
                pass
        # Если у нас уже есть данные и пользователь просит график, игнорируем лишний SQL
        # (во избежание повторных запросов к БД и «потери» контекста графика)
        try:
            last_user_text_guard = next((m["content"] for m in reversed(st.session_state.get("messages", [])) if m.get("role") == "user"), "")
        except Exception:
            last_user_text_guard = ""
        wants_chart_now = bool(re.search(r"\b(график|диаграмм|диаграмма|chart|plot)\b", last_user_text_guard, flags=re.IGNORECASE))
        if m_sql and st.session_state.get("last_df") is not None and wants_chart_now:
            # Пропускаем выполнение найденного SQL — сразу перейдём к построению графика ниже/фолбэку
            m_sql = None
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
                # SQL автоматически корректируется, но не показываем это пользователю
                if isinstance(df_any, pl.DataFrame):
                    df_pl = df_any
                else:
                    # на всякий случай: если драйвер вернул pandas
                    df_pl = pl.from_pandas(df_any) if isinstance(df_any, pd.DataFrame) else None

                st.session_state["last_df"] = df_pl
                if df_pl is not None:
                    # НОВАЯ ЛОГИКА (аналог графиков): генерируем готовый HTML сразу
                    meta_table = dict(meta_extra)
                    pdf = df_pl.to_pandas()
                    
                    # Применяем next_table_style если есть, затем очищаем его
                    styler_config = meta_table.get("styler_config") or {}
                    if st.session_state.get("next_table_style"):
                        styler_config = st.session_state["next_table_style"]
                        meta_table["styler_config"] = styler_config
                        st.session_state["next_table_style"] = None  # Очищаем после применения
                    
                    # НОВАЯ СИСТЕМА: сохраняем styler_config для Pandas Styler
                    # HTML будет генерироваться в _render_table_content_styler()
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
                    st.markdown(f"**{title}**")
                if explain:
                    st.caption(explain)
                with st.expander("Показать SQL", expanded=False):
                    orig_sql = (meta_extra.get("sql_original") or "").strip()
                    if orig_sql and orig_sql != st.session_state.get("last_sql_meta", {}).get("sql", ""):
                        st.markdown("**Исходный SQL от модели**")
                        st.code(orig_sql, language="sql")

        # Убрали обработку блока table — режим упразднён

        # 4.5) PIVOT: если ассистент вернул блок ```pivot_code``` — применяем сводное преобразование к df
        m_pivot = re.search(r"```pivot_code\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        if m_pivot and st.session_state.get("last_df") is not None:
            try:
                pivot_code = m_pivot.group(1).strip()
                df_polars = st.session_state["last_df"]
                df = df_polars.to_pandas() if isinstance(df_polars, pl.DataFrame) else df_polars
                def col(*names):
                    for nm in names:
                        if nm in df.columns:
                            return nm
                    raise KeyError(f"Нет ни одной из колонок: {names}")
                def has_col(name):
                    return name in df.columns
                COLS = list(df.columns)
                safe_globals = {
                    "__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list},
                    "pd": pd, "df": df, "col": col, "has_col": has_col, "COLS": COLS,
                }
                local_vars = {}
                exec(pivot_code, safe_globals, local_vars)
                # Ожидаем, что переменная df перезаписана на сводную
                new_df = local_vars.get("df")
                if isinstance(new_df, pd.DataFrame):
                    st.session_state["last_df"] = pl.from_pandas(new_df)
                    # Сохраним pivot_code в meta последней таблицы, чтобы при сохранении он попал в ClickHouse
                    try:
                        if st.session_state.get("results"):
                            st.session_state["results"][-1].setdefault("meta", {})["pivot_code"] = pivot_code
                    except Exception:
                        pass
                else:
                    st.info("Код сводной должен присвоить переменной df новый pandas.DataFrame.")
            except Exception as e:
                st.error(f"Ошибка выполнения pivot_code: {e}")

        # 5) Если ассистент вернул Plotly-код — исполняем его в песочнице и сохраняем график
        m_plotly = re.search(r"```plotly\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        m_python = re.search(r"```python\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        plotly_code = (m_plotly.group(1) if m_plotly else (m_python.group(1) if m_python else "")).strip()
        saw_table_plotly = False  # признак, что модель прислала go.Table внутри plotly-кода

        # Убрана старая логика применения стилей к таблицам из plotly кода

        # 6) Если ассистент вернул Table-код — исполняем его в песочнице и применяем стили
        m_table = re.search(r"```table_code\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        if m_table:
            table_code = m_table.group(1).strip()
            if table_code and st.session_state.get("last_df") is not None:
                # Проверяем, что данные действительно есть
                if st.session_state["last_df"] is None:
                    st.info("Нет данных для изменения таблицы: выполните SQL, чтобы получить df.")
                else:
                    try:
                        # Песочница для выполнения table_code
                        # ВАЖНО: Конвертируем Polars в pandas для совместимости с iloc и другими методами
                        df_polars = st.session_state["last_df"]
                        df = df_polars.to_pandas() if isinstance(df_polars, pl.DataFrame) else df_polars
                        
                        def col(*names):
                            """Вернёт первое подходящее имя колонки из перечисленных."""
                            for n in names:
                                if isinstance(n, str) and n in df.columns:
                                    return n
                            raise KeyError(f"Нет ни одной из колонок {names}. Доступны: {list(df.columns)}")
                        
                        def has_col(name: str) -> bool:
                            return isinstance(name, str) and name in df.columns
                        COLS = list(df.columns)
                        
                        # Поддержка инкрементальных правок стилей: передаём предыдущий styled_df (если был)
                        _prev_styler = None
                        try:
                            for _it_prev in reversed(st.session_state.get("results", [])):
                                if _it_prev.get("kind") == "table":
                                    _prev_styler = (_it_prev.get("meta") or {}).get("_styler_obj")
                                    if _prev_styler is not None:
                                        break
                        except Exception:
                            _prev_styler = None

                        safe_builtins = {
                            "__builtins__": {
                                "len": len, 
                                "range": range, 
                                "min": min, 
                                "max": max, 
                                "dict": dict, 
                                "list": list,
                                "str": str,
                                "int": int,
                                "float": float,
                                "bool": bool
                            },
                            "df": df,  # pandas DataFrame для совместимости
                            "st": st,
                            "pd": pd,
                            "col": col,
                            "has_col": has_col,
                            "COLS": COLS,
                            "true": True,  # Поддержка JSON-стиля
                            "false": False,
                            "null": None,
                            "styled_df": _prev_styler,  # база для инкрементальных изменений
                        }
                        local_vars = {}
                        exec(table_code, safe_builtins, local_vars)
                        
                        # 1) Готовый HTML от ассистента
                        styled_html = local_vars.get("styled_html")
                        if isinstance(styled_html, str) and styled_html.strip():
                            applied = False
                            for it in reversed(st.session_state.get("results", [])):
                                if it.get("kind") == "table" and isinstance(it.get("df_pl"), pl.DataFrame):
                                    import copy
                                    old_meta = it.get("meta") or {}
                                    old_df = it.get("df_pl")
                                    new_meta = copy.deepcopy(old_meta)
                                    new_meta["rendered_html"] = styled_html
                                    new_meta["table_code"] = table_code
                                    _push_result("table", df_pl=old_df, meta=new_meta)
                                    _render_result(st.session_state["results"][-1])
                                    applied = True
                                    created_table = True
                                    break
                        # 2) Styler от ассистента → HTML + сохранение styler для Excel
                        if not created_table:
                            styled_df_obj = local_vars.get("styled_df")
                            try:
                                if styled_df_obj is not None and hasattr(styled_df_obj, "to_html"):
                                    html_out = styled_df_obj.to_html(escape=False, table_id="styled-table")
                                    applied = False
                                    for it in reversed(st.session_state.get("results", [])):
                                        if it.get("kind") == "table" and isinstance(it.get("df_pl"), pl.DataFrame):
                                            import copy
                                            old_meta = it.get("meta") or {}
                                            old_df = it.get("df_pl")
                                            new_meta = copy.deepcopy(old_meta)
                                            new_meta["rendered_html"] = html_out
                                            new_meta["table_code"] = table_code
                                            # ВАЖНО: сохраняем styler для Excel-экспорта
                                            new_meta["_styler_obj"] = styled_df_obj
                                            _push_result("table", df_pl=old_df, meta=new_meta)
                                            _render_result(st.session_state["results"][-1])
                                            applied = True
                                            created_table = True
                                            break
                            except Exception:
                                pass
                        # 3) Старый путь: styler_config
                        if not created_table:
                            styler_config = local_vars.get("styler_config")
                            if isinstance(styler_config, dict):
                                applied = False
                                for it in reversed(st.session_state.get("results", [])):
                                    if it.get("kind") == "table" and isinstance(it.get("df_pl"), pl.DataFrame):
                                        import copy
                                        old_meta = it.get("meta") or {}
                                        old_df = it.get("df_pl")
                                        new_meta = copy.deepcopy(old_meta)
                                        new_meta["styler_config"] = styler_config
                                        new_meta["table_code"] = table_code
                                        _push_result("table", df_pl=old_df, meta=new_meta)
                                        _render_result(st.session_state["results"][-1])
                                        applied = True
                                        created_table = True
                                        break
                                if not applied:
                                    st.session_state["next_table_style"] = styler_config
                    except Exception as e:
                        st.error(f"Ошибка выполнения table_code: {e}")
                        # Для отладки: прикрепляем исходный table_code к последней таблице
                        try:
                            for _it in reversed(st.session_state.get("results", [])):
                                if _it.get("kind") == "table":
                                    _meta = _it.get("meta") or {}
                                    _meta["table_code"] = table_code
                                    _meta["table_code_error"] = str(e)
                                    _it["meta"] = _meta
                                    break
                        except Exception:
                            pass

        # Новый лёгкий формат для стилизации: блок ```table_style```
        m_tstyle = re.search(r"```table_style\s*([\s\S]*?)```", final_reply, re.IGNORECASE)
        if m_tstyle:
            try:
                import ast
                block = m_tstyle.group(1).strip()
                # Извлекаем словарь из блока (формат: table_style = {...})
                dict_match = re.search(r"\{[\s\S]*\}", block)
                if dict_match:
                    # Безопасный парсинг Python-литерала
                    table_style = ast.literal_eval(dict_match.group(0))
                    
                    if isinstance(table_style, dict):
                        # УСТАРЕВШИЙ КОД: этот блок больше не используется
                        # Используется новая система через table_code и Styler
                        # Сохраняем для новых таблиц (legacy support)
                        st.session_state["next_table_style"] = table_style
            except Exception as e:
                # Тихая обработка ошибок парсинга
                pass

        if plotly_code:
            if st.session_state["last_df"] is None:
                st.info("Нет данных для графика: выполните SQL, чтобы получить df.")
            elif m_table or m_tstyle:
                # Если есть table_code или table_style, не создаем график
                st.info("Обнаружен режим TABLE - пропускаем создание графика")
                pass
            else:
                code = plotly_code  # берём уже извлечённый текст из ```plotly или ```python

                # Если это go.Table — это табличный код, а не график. Авто‑переход в режим TABLE.
                if re.search(r"go\.Table\(", code):
                    saw_table_plotly = True
                    try:
                        # Преобразуем текущий df в табличный результат без участия plotly-кода
                        if st.session_state.get("last_df") is not None:
                            _df_pl = st.session_state["last_df"]
                            # Простой baseline: "зебра" как мягкий аналог раскраски строк
                            meta_tbl = {"sql": st.session_state.get("last_sql_meta", {}).get("sql"),
                                        "sql_original": st.session_state.get("last_sql_meta", {}).get("sql_original"),
                                        "title": st.session_state.get("last_sql_meta", {}).get("title", "")}
                            # Пометим специальный стиль на следующий рендер
                            st.session_state["next_table_style"] = {"striped": True}
                            _push_result("table", df_pl=_df_pl, meta=meta_tbl)
                            _render_result(st.session_state["results"][-1])
                            created_table = True
                            # Не выполняем plotly-код
                            code = ""
                            st.info("Получен go.Table → переключаюсь в режим TABLE и показываю таблицу с базовой раскраской (зебра).")
                        else:
                            st.info("Получен go.Table, но данных df нет — пропускаю.")
                            code = ""
                    except Exception:
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
                    # кода для графика нет (вероятно, был go.Table) — не рисуем здесь
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
                            created_chart = True

                        else:
                            st.error("Ожидается, что код в ```plotly``` создаст переменную fig (plotly.graph_objects.Figure).")
                    except Exception as e:
                        # Если ошибка связана с выбором колонок (наш helper col(...) кинул KeyError),
                        # попробуем один автоматический ретрай: напомним модели доступные колонки.
                        err_text = str(e)
                        needs_retry = isinstance(e, KeyError) or "Нет ни одной из колонок" in err_text

                        # Ретраим не только при отсутствии колонок, но и при типичных ошибках обработки строк
                        strip_related = (
                            "NoneType" in err_text or "'NoneType' object has no attribute 'strip'" in err_text or " strip(" in err_text
                        )
                        if needs_retry or strip_related:
                            try:
                                _pdf = st.session_state["last_df"].to_pandas()
                                _cols_list = list(_pdf.columns)
                                retry_hint = (
                                    "Ошибка при построении графика: " + err_text
                                    + "\nДоступные колонки: " + ", ".join(map(str, _cols_list))
                                    + "\nСгенерируй НОВЫЙ код для переменной fig, используя ТОЛЬКО эти имена через col(...). "
                                    + "Не используй .strip() или ручную обработку строк значений; работай напрямую с колонками."
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
                                            created_chart = True
                                        else:
                                            st.error("Повтор: код не создал переменную fig (plotly.graph_objects.Figure).")
                                else:
                                    st.error("Повтор: ассистент не вернул блок ```plotly```.")
                            except Exception as e2:
                                st.error(f"Повтор также не удался: {e2}")
                        else:
                            st.error(f"Ошибка выполнения кода графика: {e}")

        # --- Фолбэк: пользователь просит график, df уже есть, но модель не вернула plotly-код ---
        # Сценарий: «сделай график» после получения таблицы. Если кода нет — попросим у модели
        # с чёткой инструкцией и перечнем колонок. Выполняем один раз, тихо.
        if (
            not created_chart
            and not created_table
            and st.session_state.get("last_df") is not None
            and not (m_table or m_tstyle)
        ):
            # Найдём последнее пользовательское сообщение
            last_user_text = ""
            for _m in reversed(st.session_state.get("messages", [])):
                if _m.get("role") == "user":
                    last_user_text = _m.get("content", "")
                    break
            # Триггерим фолбэк если пользователь явно просил график ИЛИ
            # если модель прислала табличный plotly-код (go.Table), который мы пропустили
            if re.search(r"\b(график|диаграмм|диаграмма|chart|plot)\b", last_user_text, flags=re.IGNORECASE) or saw_table_plotly:
                try:
                    _pdf_fb = st.session_state["last_df"].to_pandas()
                    _cols_fb = ", ".join(map(str, list(_pdf_fb.columns)))
                    _retry_hint = (
                        "Построй НОВЫЙ график по уже существующим данным df. "
                        "Доступные колонки: " + _cols_fb + ". "
                        "Верни ровно один блок ```plotly``` с переменной fig."
                    )
                    _retry_msgs = (
                        [{"role": "system", "content": prompts_map["plotly"]},
                         {"role": "system", "content": _retry_hint}]
                        + st.session_state["messages"]
                    )
                    _retry_reply = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=_retry_msgs,
                        temperature=0,
                    ).choices[0].message.content

                    _m_plotly_fb = re.search(r"```plotly\s*(.*?)```", _retry_reply, re.DOTALL | re.IGNORECASE)
                    if _m_plotly_fb:
                        _code_retry = _m_plotly_fb.group(1).strip()
                        _code_retry_clean = re.sub(r"(?m)^\s*(?:from\s+\S+\s+import\s+.*|import\s+.*)\s*$", "", _code_retry)
                        _scan2 = re.sub(r"'''[\s\S]*?'''", "", _code_retry_clean)
                        _scan2 = re.sub(r'"""[\s\S]*?"""', "", _scan2)
                        _scan2 = re.sub(r"(?m)#.*$", "", _scan2)
                        _BANNED2 = re.compile(
                            r"(?:\bopen\b|\bexec\b|\beval\b|subprocess|socket|"
                            r"os\.[A-Za-z_]+|sys\.[A-Za-z_]+|Path\(|write\(|remove\(|unlink\(|requests|httpx)",
                            re.IGNORECASE,
                        )
                        if not _BANNED2.search(_scan2):
                            # Поддержим col(...) и базовое окружение
                            def _col(*names):
                                for n in names:
                                    if isinstance(n, str) and n in _pdf_fb.columns:
                                        return n
                                raise KeyError(f"Нет ни одной из колонок {names}. Доступны: {list(_pdf_fb.columns)}")
                            _safe_globals = {
                                "__builtins__": {"len": len, "range": range, "min": min, "max": max, "dict": dict, "list": list},
                                "pd": pd, "px": px, "go": go, "df": _pdf_fb, "col": _col, "COLS": list(_pdf_fb.columns)
                            }
                            _locals = {}
                            exec(_code_retry_clean, _safe_globals, _locals)
                            _fig = _locals.get("fig")
                            if isinstance(_fig, go.Figure):
                                _push_result("chart", fig=_fig, meta={"plotly_code": _code_retry})
                                _render_result(st.session_state["results"][-1])
                                created_chart = True
                except Exception:
                    pass

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
