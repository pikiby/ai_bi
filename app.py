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

# --- Авто-индексация при старте (однократно на процесс) ---
# Включается флагом окружения: KB_AUTO_INGEST_ON_START=1
@st.cache_resource  # гарантирует запуск ровно один раз на процесс Streamlit
def _auto_ingest_once():
    from ingest import run_ingest  # ленивый импорт, чтобы не тянуть зависимости зря
    return run_ingest()

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

# история чата
if "messages" not in st.session_state:
    st.session_state["messages"] = []  

# история результатов (таблицы/графики)
if "results" not in st.session_state:
    st.session_state["results"] = []   

#  последний df (polars), для построения графиков
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

# кэш последнего непустого RAG-контекста (используем как фолбэк)
if "last_rag_ctx" not in st.session_state:
    st.session_state["last_rag_ctx"] = ""

# кэш схемы БД для быстрого доступа
if "db_schema_cache" not in st.session_state:
    st.session_state["db_schema_cache"] = {}

# ----------------------- Вспомогательные функции -----------------------

def _reload_prompts():
    """
    Перечитать prompts.py на лету, вернуть словарь с системными блоками.
    Если какого-то блока нет — подставим дефолт и пометим предупреждение.
    """
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
            "Ты — маршрутизатор. Верни ровно один блок ```mode\nsql\n``` где sql|rag|table|plotly|catalog."
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
        "table": _get(
            "RULES_TABLE",
            "Режим TABLE. Верни ровно один блок ```table``` с кодом, создающим df2 из df."
        ),
    }
    return p_map, warn


def _push_result(kind: str, df_pl: pl.DataFrame | None = None,
                 fig: go.Figure | None = None, meta: dict | None = None):
    """
    Сохраняем элемент в «Историю результатов».
    kind: "table" | "chart"
    meta: например {"sql": "..."}.
    """
    st.session_state["results"].append({
        "kind": kind,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "df_pl": df_pl,     # polars.DataFrame (для таблицы/экспорта)
        "fig": fig,         # plotly.graph_objects.Figure (для графика/экспорта)
        "meta": meta or {},
        "msg_idx": st.session_state.get("last_assistant_idx"),
    })

# Порядковый номер текущей таблицы среди уже отрисованных результатов.
def _table_number_for(item: dict) -> int:
    n = 0
    for it in st.session_state.get("results", []):
        if it.get("kind") == "table":
            n += 1
        if it is item:
            break
    return n


# Достаём из SQL краткие сведения для подписи: таблицы, поля, период, лимит.
# Всё максимально компактно и устойчиво к разным диалектам.
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

def _tables_index_hint() -> str:
    """
    Компактная «память» для модели: список ранее построенных таблиц с номерами,
    заголовками (если уже задавались) и краткой выжимкой из SQL (источники и период).
    Эту строку добавляем как system-сообщение перед роутером/SQL.
    """
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

def _strip_llm_blocks(text: str) -> str:
    """
    Удаляет служебные блоки из ответа модели:
    ```title```, ```explain```, ```sql```, ```rag```, а также код для графиков
    из ```python```/```plotly``` — чтобы не дублировать их в чате.
    """
    if not text:
        return text
    for tag in ("title", "explain", "sql", "rag", "python", "plotly", "table"):
        text = re.sub(
            rf"```{tag}\s*.*?```",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _render_result(item: dict):
    """
    Отрисовка одного элемента истории результатов.
    """
    kind = item.get("kind")
    # Блок для отрисовывания таблиц
    if kind == "table":
        df_pl = item.get("df_pl")
        if isinstance(df_pl, pl.DataFrame):
            pdf = df_pl.to_pandas()

            # --- Заголовок из meta.title, fallback: старая эвристика ---
            n = _table_number_for(item)
            meta = item.get("meta") or {}
            title = (meta.get("title") or "").strip()
            explain = (meta.get("explain") or "").strip()
            sql = (meta.get("sql") or "").strip()

            if not title:
                # Fallback: если модель не дала title — формируем «разумный» заголовок из LIMIT и колонок
                info = _extract_sql_info(sql, pdf)
                if info.get("limit"):
                    lead = None
                    for c in info["columns"]:
                        if re.search(r"(city|город|category|катег|product|товар|region|регион|name|назв)", c, flags=re.IGNORECASE):
                            lead = c; break
                    title = f'Топ {info["limit"]}' + (f" по «{lead}»" if lead else "")
                else:
                    title = "Результаты запроса"

            st.markdown(f"### Таблица {n}: {title}")

            # --- Сама таблица (редактируемая) ---
            edit_key = f"ed_{item.get('ts','')}"
            st.data_editor(pdf, use_container_width=True, key=edit_key, num_rows="dynamic")

            # --- Подпись под таблицей: prefer explain от модели; fallback — выжимка из SQL ---
            if explain:
                st.caption(explain)
            else:
                info = _extract_sql_info(sql, pdf)
                src = ", ".join(info.get("tables") or []) or "источник не указан"
                period = info.get("period") or "период не указан"
                st.caption(f"Источник: {src}. Период: {period}.")
            
            # --- Свернутый блок с SQL запроса (по кнопке) ---
            used_sql = (meta.get("sql") or "").strip()
            orig_sql = (meta.get("sql_original") or "").strip()
            with st.expander("Показать SQL", expanded=False):
                if used_sql:
                    st.markdown("**Использованный SQL**")
                    st.code(used_sql, language="sql")
                    if orig_sql and orig_sql != used_sql:
                        st.markdown("**Исходный SQL от модели**")
                        st.code(orig_sql, language="sql")
                elif orig_sql:
                    st.code(orig_sql, language="sql")
                else:
                    st.caption("SQL недоступен для этой таблицы.")

            # --- Свернутый блок с исходным кодом Plotly (по аналогии с графиками) ---
            # Если кода ещё нет (таблица из SQL) — сгенерируем минимальный go.Table
            if not (meta.get("plotly_code") or "").strip():
                try:
                    meta["plotly_code"] = _default_plotly_table_code(pdf)
                except Exception:
                    meta["plotly_code"] = ""
            plotly_src_tbl = (meta.get("plotly_code") or "").strip()
            with st.expander("Показать код Plotly", expanded=False):
                if plotly_src_tbl:
                    st.code(plotly_src_tbl, language="python")
                else:
                    st.caption("Код недоступен для этой таблицы.")

            # --- Кнопки скачивания ИМЕННО этой таблицы ---
            ts = (item.get("ts") or "table").replace(":", "-")
            try:
                col_csv, col_xlsx, _ = st.columns([4, 4, 2], gap="small")
            except TypeError:
                col_csv, col_xlsx, _ = st.columns([4, 4, 2])

            with col_csv:
                st.download_button(
                    "Скачать CSV",
                    data=_df_to_csv_bytes(pdf),
                    file_name=f"table_{ts}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{ts}",
                    use_container_width=True,
                )
            with col_xlsx:
                st.download_button(
                    "Скачать XLSX",
                    data=_df_to_xlsx_bytes(pdf, "Result"),
                    file_name=f"table_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xlsx_{ts}",
                    use_container_width=True,
                )
    
    # Блок для отрисовывания чартов
    elif kind == "chart":
        fig = item.get("fig")
        if isinstance(fig, go.Figure):
            st.markdown("**График**")

            # Включаем клиентскую кнопку PNG в тулбаре (иконка «камера»).
            # Это работает без нихрена лишнего: Plotly в браузере сам сформирует PNG.
            st.plotly_chart(
                fig,
                theme=None,  # важно: не мешаем Streamlit-тему, чтобы цвета совпадали с экспортом
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "toImageButtonOptions": {"format": "png", "scale": 2}
                },
            )

            # --- Кнопка скачивания ИМЕННО этого графика (HTML), стиль как у таблиц ---
            ts = (item.get("ts") or "chart").replace(":", "-")
            html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")

            try:
                col_html, _ = st.columns([4, 8], gap="small")  # левая широкая кнопка + спейсер
            except TypeError:
                col_html, _ = st.columns([4, 8])

            # --- Свернутый блок с исходным кодом Plotly (по кнопке) ---
            meta = item.get("meta") or {}
            plotly_src = (meta.get("plotly_code") or "").strip()
            with st.expander("Показать код Plotly", expanded=False):
                if plotly_src:
                    st.code(plotly_src, language="python")
                else:
                    st.caption("Код недоступен для этого графика.")

            with col_html:
                st.download_button(
                    "Скачать график",
                    data=html_bytes,
                    file_name=f"chart_{ts}.html",
                    mime="text/html",
                    key=f"dl_html_{ts}",
                    use_container_width=True,
                )


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
    """Создаём Plotly-таблицу, чтобы можно было править данные прямо в UI."""

    # Plotly ожидает список колонок; сразу приводим к list для совместимости с numpy/pandas типами.
    column_values = [pdf[col].tolist() for col in pdf.columns] if not pdf.empty else [[] for _ in pdf.columns]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[str(col) for col in pdf.columns],
                    fill_color="#111827",  # тёмный фон шапки
                    font=dict(color="#f9fafb", size=13),  # светлый текст + небольшое увеличение размера
                    align="left",
                ),
                cells=dict(
                    values=column_values,
                    fill_color="#1f2933",  # тёмный фон строк, близкий к шапке
                    font=dict(color="#f9fafb"),  # светлый текст для контраста
                    align="left",
                ),
            )
        ]
    )

    # Небольшие отступы и ограничение по высоте, чтобы таблица выглядела аккуратно.
    fig.update_layout(margin=dict(l=0, r=0, t=12, b=0), height=min(560, 80 + 24 * len(pdf)))

    return fig

def _default_plotly_table_code(df: pd.DataFrame) -> str:
    """Сгенерировать минимальный код Plotly-таблицы для текущего df (редактируемый пользователем)."""
    cols = [str(c) for c in df.columns]
    # Значения берём из df по именам колонок
    code = (
        "values = [df[c].tolist() for c in df.columns]\n"
        "fig = go.Figure(data=[go.Table(\n"
        "    header=dict(values=[str(c) for c in df.columns]),\n"
        "    cells=dict(values=values)\n"
        ")])\n"
        "fig.update_layout(margin=dict(l=0, r=0, t=12, b=0))\n"
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
    precise_hint = "\n".join(lines)

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
         {"role": "system", "content": prompts_map["sql"]}] +
        base_messages +
        [{"role": "system", "content": guard_instr}]
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

    if mode not in {"sql", "rag", "table", "plotly", "catalog"}:
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

    elif mode == "table":
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
            [{"role": "system", "content": prompts_map["table"]}]
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
            final_reply = "Не удалось получить код преобразования таблицы."
            st.error(f"Ошибка на шаге ответа (Table): {e}")

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

        # 4) Если ассистент вернул SQL — выполняем ClickHouse и сохраняем таблицу
        m_sql = re.search(r"```sql\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        if m_sql:
            sql = m_sql.group(1).strip()
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
                # Обновим meta: показываем фактически выполненный SQL, сохранив исходный
                meta_extra["sql"] = used_sql
                if used_sql.strip() != sql.strip():
                    st.info("SQL был автоматически скорректирован по схеме (отсутствующие поля/алиасы исправлены).")
                if isinstance(df_any, pl.DataFrame):
                    df_pl = df_any
                else:
                    # на всякий случай: если драйвер вернул pandas
                    df_pl = pl.from_pandas(df_any) if isinstance(df_any, pd.DataFrame) else None

                st.session_state["last_df"] = df_pl
                if df_pl is not None:
                    _push_result("table", df_pl=df_pl, meta=meta_extra)
                    _render_result(st.session_state["results"][-1])
                else:
                    st.error("Драйвер вернул неожиданный формат данных.")
            except Exception as e:
                st.error(f"Ошибка выполнения SQL: {e}")

        # 5a) Если ассистент вернул Table-код — исполняем его в песочнице и сохраняем новую таблицу
        m_table = re.search(r"```table\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        table_code = (m_table.group(1).strip() if m_table else "")
        if table_code:
            if st.session_state["last_df"] is None:
                st.info("Нет данных: сначала выполните SQL, чтобы получить df.")
            else:
                # Базовая защита
                BANNED_RE = re.compile(
                    r"(?:\bimport\b|\bopen\b|\bexec\b|\beval\b|subprocess|socket|"
                    r"os\.[A-Za-z_]+|sys\.[A-Za-z_]+|Path\(|write\(|remove\(|unlink\(|requests|httpx)",
                    re.IGNORECASE,
                )
                # Чистим безопасные импорты (не требуются) и комментарии
                code_clean = table_code
                code_clean = re.sub(r"(?m)^\s*import\s+pandas\s+as\s+pd\s*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*import\s+numpy\s+as\s+np\s*$", "", code_clean)
                code_clean = re.sub(r"'''[\s\S]*?'''", "", code_clean)
                code_clean = re.sub(r'"""[\s\S]*?"""', "", code_clean)
                code_scan = re.sub(r"(?m)#.*$", "", code_clean)

                m_banned = BANNED_RE.search(code_scan)
                if m_banned:
                    st.error(f"Код преобразования отклонён (запрещено: '{m_banned.group(0)}').")
                else:
                    try:
                        pdf = st.session_state["last_df"].to_pandas()

                        def col(*names):
                            for n in names:
                                if isinstance(n, str) and n in pdf.columns:
                                    return n
                            raise KeyError(f"Нет ни одной из колонок {names}. Доступны: {list(pdf.columns)}")

                        def has_col(name: str) -> bool:
                            return isinstance(name, str) and name in pdf.columns

                        COLS = list(pdf.columns)

                        safe_globals = {
                            "__builtins__": {"len": len, "range": range, "min": min, "max": max},
                            "pd": pd,
                            "df": pdf,       # данные для преобразования (только чтение оригинала)
                            "col": col,
                            "has_col": has_col,
                            "COLS": COLS,
                        }
                        local_vars = {}
                        exec(code_clean, safe_globals, local_vars)

                        df2 = local_vars.get("df2")
                        # Допускаем inplace-подход: если df2 не создан, но локальный df изменён — используем его
                        if not isinstance(df2, pd.DataFrame):
                            cand = local_vars.get("df")
                            if isinstance(cand, pd.DataFrame):
                                df2 = cand
                        if isinstance(df2, pd.DataFrame):
                            df_pl2 = pl.from_pandas(df2)
                            st.session_state["last_df"] = df_pl2
                            m_title = re.search(r"```title\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
                            m_explain = re.search(r"```explain\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
                            meta = {
                                "title": (m_title.group(1).strip() if m_title else None),
                                "explain": (m_explain.group(1).strip() if m_explain else None),
                                "table_code": table_code,
                            }
                            _push_result("table", df_pl=df_pl2, meta=meta)
                            _render_result(st.session_state["results"][-1])
                        else:
                            st.error("Ожидается, что код в ```table``` создаст переменную df2 (pandas.DataFrame).")
                    except Exception as e:
                        # Авто-ретрай: напоминаем доступные колонки и просим новый код
                        err_text = str(e)
                        needs_retry = isinstance(e, KeyError) or "Нет ни одной из колонок" in err_text
                        if needs_retry:
                            try:
                                _pdf = st.session_state["last_df"].to_pandas()
                                _cols_list = list(_pdf.columns)
                                retry_hint = (
                                    "Ошибка выбора колонок при преобразовании таблицы: "
                                    + err_text
                                    + "\nДоступные колонки: "
                                    + ", ".join(map(str, _cols_list))
                                    + "\nСгенерируй НОВЫЙ код для переменной df2, используя ТОЛЬКО эти имена через col(...)."
                                )
                                retry_msgs = (
                                    [{"role": "system", "content": prompts_map["table"]},
                                     {"role": "system", "content": retry_hint}]
                                    + st.session_state["messages"]
                                )
                                retry_reply = client.chat.completions.create(
                                    model=OPENAI_MODEL,
                                    messages=retry_msgs,
                                    temperature=0,
                                ).choices[0].message.content

                                m_table_retry = re.search(r"```table\s*(.*?)```", retry_reply, re.DOTALL | re.IGNORECASE)
                                if m_table_retry:
                                    code_retry = m_table_retry.group(1).strip()
                                    # Повтор: чистим безопасные импорты
                                    code_retry_clean = re.sub(r"(?m)^\s*import\s+pandas\s+as\s+pd\s*$", "", code_retry)
                                    code_retry_clean = re.sub(r"(?m)^\s*import\s+numpy\s+as\s+np\s*$", "", code_retry_clean)
                                    code_scan2 = re.sub(r"'''[\s\S]*?'''", "", code_retry_clean)
                                    code_scan2 = re.sub(r'"""[\s\S]*?"""', "", code_scan2)
                                    code_scan2 = re.sub(r"(?m)#.*$", "", code_scan2)
                                    m_banned2 = BANNED_RE.search(code_scan2)
                                    if m_banned2:
                                        st.error(f"Код преобразования (повтор) отклонён (запрещено: '{m_banned2.group(0)}').")
                                    else:
                                        safe_globals_retry = {
                                            "__builtins__": {"len": len, "range": range, "min": min, "max": max},
                                            "pd": pd,
                                            "df": pdf,
                                            "col": col,
                                            "has_col": has_col,
                                            "COLS": COLS,
                                        }
                                        local_vars = {}
                                        exec(code_retry_clean, safe_globals_retry, local_vars)
                                        df2 = local_vars.get("df2")
                                        if not isinstance(df2, pd.DataFrame):
                                            cand = local_vars.get("df")
                                            if isinstance(cand, pd.DataFrame):
                                                df2 = cand
                                        if isinstance(df2, pd.DataFrame):
                                            df_pl2 = pl.from_pandas(df2)
                                            st.session_state["last_df"] = df_pl2
                                            _push_result("table", df_pl=df_pl2, meta={"table_code": code_retry})
                                            _render_result(st.session_state["results"][-1])
                                        else:
                                            st.error("Повтор: код не создал переменную df2 (pandas.DataFrame).")
                                else:
                                    st.error("Повтор: ассистент не вернул блок ```table```. ")
                            except Exception as e2:
                                st.error(f"Повтор также не удался: {e2}")
                        else:
                            st.error(f"Ошибка выполнения кода преобразования: {e}")

        # 5) Если ассистент вернул Plotly-код — исполняем его в песочнице и сохраняем график
        m_plotly = re.search(r"```plotly\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        m_python = re.search(r"```python\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
        plotly_code = (m_plotly.group(1) if m_plotly else (m_python.group(1) if m_python else "")).strip()
        if plotly_code:
            if st.session_state["last_df"] is None:
                st.info("Нет данных для графика: выполните SQL, чтобы получить df.")
            else:
                code = plotly_code  # берём уже извлечённый текст из ```plotly или ```python

                # Базовая защита: не допускаем опасные конструкции
                BANNED_RE = re.compile(
                    r"(?:\bimport\b|\bopen\b|\bexec\b|\beval\b|subprocess|socket|"
                    r"os\.[A-Za-z_]+|sys\.[A-Za-z_]+|Path\(|write\(|remove\(|unlink\(|requests|httpx)",
                    re.IGNORECASE,
                )
                # >>> удалим безопасные импорты px/go/pd/np и комментарии/тройные строки
                code_clean = code
                code_clean = re.sub(r"(?m)^\s*import\s+plotly\.express\s+as\s+px\s*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*import\s+plotly\.graph_objects\s+as\s+go\s*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*from\s+plotly\s+import\s+graph_objects\s+as\s+go\s*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*from\s+plotly\.graph_objects\s+import\s+.*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*from\s+plotly\.express\s+import\s+.*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*import\s+pandas\s+as\s+pd\s*$", "", code_clean)
                code_clean = re.sub(r"(?m)^\s*import\s+numpy\s+as\s+np\s*$", "", code_clean)
                code_scan = code_clean
                # многострочные ''' ... ''' и """ ... """
                code_scan = re.sub(r"'''[\s\S]*?'''", "", code_scan)
                code_scan = re.sub(r'"""[\s\S]*?"""', "", code_scan)
                # однострочные комментарии: # ...
                code_scan = re.sub(r"(?m)#.*$", "", code_scan)

                if BANNED_RE.search(code_scan):
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
                            "__builtins__": {"len": len, "range": range, "min": min, "max": max},
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

                                    # Повторная базовая проверка безопасности + удалим безопасные импорты
                                    code_retry_clean = code_retry
                                    code_retry_clean = re.sub(r"(?m)^\s*import\s+plotly\.express\s+as\s+px\s*$", "", code_retry_clean)
                                    code_retry_clean = re.sub(r"(?m)^\s*import\s+plotly\.graph_objects\s+as\s+go\s*$", "", code_retry_clean)
                                    code_retry_clean = re.sub(r"(?m)^\s*from\s+plotly\s+import\s+graph_objects\s+as\s+go\s*$", "", code_retry_clean)
                                    code_retry_clean = re.sub(r"(?m)^\s*from\s+plotly\.graph_objects\s+import\s+.*$", "", code_retry_clean)
                                    code_retry_clean = re.sub(r"(?m)^\s*from\s+plotly\.express\s+import\s+.*$", "", code_retry_clean)
                                    code_retry_clean = re.sub(r"(?m)^\s*import\s+pandas\s+as\s+pd\s*$", "", code_retry_clean)
                                    code_retry_clean = re.sub(r"(?m)^\s*import\s+numpy\s+as\s+np\s*$", "", code_retry_clean)
                                    code_scan2 = re.sub(r"'''[\s\S]*?'''", "", code_retry_clean)
                                    code_scan2 = re.sub(r'"""[\s\S]*?"""', "", code_scan2)
                                    code_scan2 = re.sub(r"(?m)#.*$", "", code_scan2)
                                    if BANNED_RE.search(code_scan2):
                                        st.error("Код графика (повтор) отклонён (запрещённые конструкции).")
                                    else:
                                        # Выполняем повторный код в том же «песочном» окружении
                                        # Собираем такое же безопасное окружение, как в первом запуске
                                        safe_globals_retry = {
                                            "__builtins__": {"len": len, "range": range, "min": min, "max": max},
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

