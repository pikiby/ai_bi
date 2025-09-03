# app.py
# Полноценное приложение Streamlit: динамическая подгрузка промптов, роутер режимов,
# SQL (ClickHouse), RAG, безопасный Plotly, история результатов и экспорт.

import os
import re
import io
import zipfile
import subprocess
from datetime import datetime
import sys

import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from openai import OpenAI
from clickhouse_client import ClickHouse_client
import retriever

# >>> Горячая подгрузка prompts.py при каждом обращении
import importlib
import prompts

# ----------------------- Базовые настройки страницы -----------------------

# Из уважения к предпочтениям — без emoji в иконке
st.set_page_config(page_title="AI SQL Assistant")

# ----------------------- Константы окружения -----------------------

CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")   # было, вероятно: "./chroma"
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")  # было, вероятно: "kb_default"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# ----------------------- Клиент OpenAI -----------------------

# >>> клиент создаём один раз
client = OpenAI()

# ----------------------- Состояние приложения -----------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # история чата

if "results" not in st.session_state:
    st.session_state["results"] = []   # история результатов (таблицы/графики)

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None # последний df (polars), для построения графиков

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
            "Ты — маршрутизатор. Верни ровно один блок ```mode\nsql\n``` где sql|rag|plotly."
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
        "meta": meta or {}
    })


def _render_result(item: dict):
    """
    Отрисовка одного элемента истории результатов.
    """
    kind = item.get("kind")
    if kind == "table":
        df_pl = item.get("df_pl")
        if isinstance(df_pl, pl.DataFrame):
            st.markdown("**Таблица**")
            pdf = df_pl.to_pandas()
            st.dataframe(pdf)

            # --- Кнопки скачивания ИМЕННО этой таблицы ---
            # Делаем стабильные уникальные ключи для кнопок и читаемые имена файлов.
            ts = (item.get("ts") or "table").replace(":", "-")
            col_a, col_b = st.columns(2)

            with col_a:
                st.download_button(
                    "Скачать CSV",
                    data=_df_to_csv_bytes(pdf),
                    file_name=f"table_{ts}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{ts}",
                )
            with col_b:
                st.download_button(
                    "Скачать XLSX",
                    data=_df_to_xlsx_bytes(pdf, "Result"),
                    file_name=f"table_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xlsx_{ts}",
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


# ----------------------- Сайдбар -----------------------

with st.sidebar:
    st.header("Данные / экспорт")

    # Кнопка пересоздания/обновления индекса базы знаний
    with st.expander("Переиндексация базы знаний"):
        if st.button("Запустить ingest.py"):
            with st.status("Индексируем…", expanded=True) as status:
                try:
                    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
                    proc = subprocess.run(
                        [sys.executable, "ingest.py"],
                        capture_output=True, text=True, check=False
                    )
                    st.code(proc.stdout or "(пусто)")
                    if proc.returncode == 0:
                        status.update(label="Готово", state="complete")
                        st.success("Индекс обновлён.")
                        st.rerun()  # сразу обновим интерфейс
                    else:
                        status.update(label="Ошибка", state="error")
                        st.error(proc.stderr or "Неизвестная ошибка")
                except Exception as e:
                    st.error(f"Не удалось запустить ingest.py: {e}")

    st.divider()

    # Кнопка скачивания архива

    st.divider()
    if st.button("Очистить историю результатов"):
        st.session_state["results"].clear()
        st.session_state["last_df"] = None
        st.rerun()

# ----------------------- Основной layout -----------------------

st.title("AI SQL Assistant")

# Предупреждения о пропущенных блоках промптов (если есть)
_prompts_map, _prompts_warn = _reload_prompts()
if _prompts_warn:
    st.warning("В `prompts.py` отсутствуют: " + ", ".join(_prompts_warn))

# Рендер существующей истории чата
if st.session_state["messages"]:
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# Рендер истории результатов (если есть)
if st.session_state["results"]:
    st.markdown("### История результатов")
    for item in st.session_state["results"]:
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
    if mode not in {"sql", "rag", "plotly"}:
        mode = "sql"  # >>> на случай 'pivot' или другого не реализованного режима

    final_reply = ""

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
                hits = retriever.retrieve(
                    rag_query, k=5,
                    chroma_path=CHROMA_PATH,
                    collection_name=COLLECTION_NAME,
                )
            except Exception as e:
                st.warning(f"Не удалось получить контекст из базы знаний: {e}")

        context = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])

        # 2b) Финальный ответ/SQL с учётом контекста
        exec_msgs = (
            [{"role": "system", "content": prompts_map["sql"]}] +  # <<< КЛЮЧЕВАЯ ЗАМЕНА
            st.session_state["messages"] +
            [{"role": "system", "content": f"Контекст базы знаний:\n{context}\nОтвечай кратко и строго по этому контексту. Если нужных таблиц нет — скажи об этом и не пиши SQL."}]
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
        exec_msgs = [{"role": "system", "content": prompts_map["sql"]}] + st.session_state["messages"]
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
    with st.chat_message("assistant"):
        st.markdown(final_reply)

    # 4) Если ассистент вернул SQL — выполняем ClickHouse и сохраняем таблицу
    m_sql = re.search(r"```sql\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
    if m_sql:
        sql = m_sql.group(1).strip()
        try:
            ch = ClickHouse_client()
            df_any = ch.query_run(sql)  # ожидается polars.DataFrame
            if isinstance(df_any, pl.DataFrame):
                df_pl = df_any
            else:
                # на всякий случай: если драйвер вернул pandas
                df_pl = pl.from_pandas(df_any) if isinstance(df_any, pd.DataFrame) else None

            st.session_state["last_df"] = df_pl
            if df_pl is not None:
                _push_result("table", df_pl=df_pl, meta={"sql": sql})
                _render_result(st.session_state["results"][-1])
            else:
                st.error("Драйвер вернул неожиданный формат данных.")
        except Exception as e:
            st.error(f"Ошибка выполнения SQL: {e}")

    # 5) Если ассистент вернул Plotly-код — исполняем его в песочнице и сохраняем график
    m_plotly = re.search(r"```plotly\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
    if m_plotly:
        if st.session_state["last_df"] is None:
            st.info("Нет данных для графика: выполните SQL, чтобы получить df.")
        else:
            code = m_plotly.group(1).strip()

            # Базовая защита: не допускаем опасные конструкции
            BANNED_RE = re.compile(
                r"(?:\bimport\b|\bopen\b|\bexec\b|\beval\b|__|subprocess|socket|"
                r"os\.[A-Za-z_]+|sys\.[A-Za-z_]+|Path\(|write\(|remove\(|unlink\(|requests|httpx)",
                re.IGNORECASE,
            )
            # >>> Перед проверкой уберём комментарии и тройные строки
            code_scan = code
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
                    exec(code, safe_globals, local_vars)

                    fig = local_vars.get("fig")
                    if isinstance(fig, go.Figure):
                        _push_result("chart", fig=fig, meta={})
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

                                # Повторная базовая проверка безопасности
                                code_scan2 = re.sub(r"'''[\s\S]*?'''", "", code_retry)
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
                                    exec(code_retry, safe_globals_retry, local_vars)
                                    fig = local_vars.get("fig")

                                    if isinstance(fig, go.Figure):
                                        _push_result("chart", fig=fig, meta={})
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

