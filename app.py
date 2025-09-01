# app.py
# =========================
# ЕДИНЫЙ чат: авто-роутинг между RAG (docs/ через Chroma) и SQL (ClickHouse),
# поддержка визуализации (Plotly) и "липкого" графика между ререндерами.
# =========================

import os
import sys
import re
import io
import glob
import subprocess
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
from openai import OpenAI
from clickhouse_client import ClickHouse_client

from retriever import retrieve
from sql_assistant import run_sql_assistant  # генерация/исполнение безопасного SELECT

st.set_page_config(page_title="Chat + RAG + SQL (Auto)", page_icon="💬", layout="centered")

# ---------- Глобальные настройки Chroma ----------
CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")

def _validate_collection_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,63}", n):
        raise ValueError(f"Некорректное имя коллекции: {name!r}. Разрешены 3–63 символа: [a-z0-9_].")
    return n

COLLECTION_NAME = _validate_collection_name(COLLECTION_NAME)
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ---------- OpenAI ключ ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Переменная окружения OPENAI_API_KEY не задана.")
    st.stop()
client = OpenAI(api_key=api_key)

# ---------- Session State ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sql", None)
st.session_state.setdefault("viz_active", False)  # показывать график при ререндерах
st.session_state.setdefault("viz_text", "")       # последний запрос, по которому строился график

# ---------- Служебные хелперы ----------
def build_history_for_llm(max_turns: int = 6):
    """Вернуть последние max_turns ходов диалога для LLM (без таблиц/графиков)."""
    msgs = []
    for m in st.session_state.messages[-max_turns:]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs

def _extract_last_sql_from_history() -> str | None:
    # ищем последний блок ```sql ... ``` в ответах ассистента
    for m in reversed(st.session_state.get("messages", [])):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        match = re.search(r"```sql\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def get_last_sql() -> str | None:
    """
    Единая точка: достаём последний SQL.
    1) Берём из session_state (если ранее сохранился);
    2) Иначе — вытаскиваем из истории ассистента (```sql ... ```).
    """
    return st.session_state.get("last_sql") or _extract_last_sql_from_history()

_FORBIDDEN_DML = re.compile(
    r"(?is)\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|REPLACE|OPTIMIZE|ATTACH|DETACH|SYSTEM|KILL|RENAME|GRANT|REVOKE|CREATE)\b"
)

def _safe_fetch_df_from_sql(sql: str, limit: int = 500):
    """
    Выполнить ПОСЛЕДНИЙ SELECT безопасно:
    - допускаем только WITH...SELECT или SELECT в начале;
    - блокируем DML/DDL;
    - если нет LIMIT — добавляем в конец.
    Возвращает polars.DataFrame или None.
    """
    if not sql:
        return None
    s = sql.strip()
    if not re.match(r"(?is)^\s*(with\b.*?select\b|select\b)", s):
        return None
    if _FORBIDDEN_DML.search(s):
        return None
    if re.search(r"(?is)\blimit\s+\d+\b", s) is None:
        s = s.rstrip().rstrip(";") + f"\nLIMIT {limit}"
    try:
        ch = ClickHouse_client()
        return ch.query_run(s)
    except Exception:
        return None

def get_last_df():
    """
    Единая точка для данных последнего запроса.
    Сейчас — просто обёртка над кешем.
    Если захотите — сюда легко добавить sql_runs/эвикцию и т.п.
    """
    return st.session_state.get("last_sql_df")

# Явное намерение «график из последних данных»
_CHART_FROM_LAST_TRIGGERS = [
    "из последнего запроса",
    "из последнегозапроса",  # без пробела — частый кейс
    "по последнему запросу",
    "по предыдущему запросу",
    "по прошлому запросу",
    "из того запроса",
    "как в прошлый раз график",
]
def is_chart_from_last_data(text: str) -> bool:
    t = (text or "").lower()
    t_nospace = re.sub(r"\s+", "", t)
    return any(p in t or p in t_nospace for p in _CHART_FROM_LAST_TRIGGERS)

# Единая «context-first» проверка
def context_first_orchestrate(user_input: str):
    """
    Возвращает dict с ранним решением, если контекста достаточно:
      {"action": "chart_from_last", "reason": "..."}           — строить график из last_sql_df
      {"action": "sql_edit", "prev_sql": "...", "reason": "..."} — правка прошлого SQL
      None                                                     — нет раннего решения, идём в общий роутер
    """
    # 1) График из последних данных
    if is_chart_intent(user_input) and is_chart_from_last_data(user_input):
        if get_last_sql():
            return {"action": "chart_from_last", "reason": "context:last_sql_in_history"}
        # нет last_sql_df — раннее решение не принимаем

    # 2) Правка прошлého SQL
    from_here_prev = st.session_state.get("last_sql")
    if from_here_prev is None:
        from_here_prev = _extract_last_sql_from_history()
        if from_here_prev:
            st.session_state["last_sql"] = from_here_prev

    if from_here_prev and is_followup_sql_edit(user_input):
        return {"action": "sql_edit", "prev_sql": from_here_prev, "reason": "context:followup_with_prev_sql"}

    return None

def is_followup_sql_edit(text: str) -> bool:
    t = (text or "").lower()
    # короткие бытовые формулировки
    triggers = [
        "добавь столб", "добавь колон", "добавь поле",
        "переимен", "замени", "удали столб", "удали колон", "удали поле",
        "добавь фильтр", "убери фильтр", "добавь услов", "убери услов",
        "сортир", "order by", "группир", "group by", "агрег",
        "сделай долю", "процент", "кумулятив", "кумулятивный", "running total",
        "тот же запрос", "как раньше", "как в прошлый раз", "тот же, но", "добавь в запрос",
    ]
    return any(p in t for p in triggers)

# --- Эвристики для SQL / RAG ---
SQL_HINTS = [
    r"\bselect\b", r"\bjoin\b", r"\bwhere\b", r"\border by\b", r"\bgroup by\b",
    r"\bcount\b", r"\bsum\b", r"\bavg\b", r"\bmin\b", r"\bmax\b",
    r"\bагрег", r"\bсумм", r"\bпосчит", r"\bсколько\b", r"\bтренд\b",
]
RAG_HINTS = [
    r"\bчто такое\b", r"\bобъясн", r"\bописан", r"\bдокументац", r"\bсхем",
    r"\bddl\b", r"\bschema\b", r"\bтип пол(я|ей)\b", r"\bописание таблиц",
]

def _score(patterns, text):
    return sum(1 for p in patterns if re.search(p, text))

def heuristic_route(question: str):
    q = (question or "").lower()
    score_sql = _score(SQL_HINTS, q)
    score_rag = _score(RAG_HINTS, q)
    if score_sql - score_rag >= 2:
        return "sql", f"heuristic:{score_sql}"
    if score_rag - score_sql >= 1:
        return "rag", f"heuristic:{score_rag}"
    return "unknown", f"heuristic:{score_sql}-{score_rag}"

def llm_route(question: str, model: str = "gpt-4o-mini"):
    """Фолбэк-классификация через LLM: возвращает ('sql'|'rag', 'llm')."""
    sys_txt = (
        "Классифицируй запрос:\n"
        "- SQL — если нужно посчитать/выбрать/агрегировать данные из ClickHouse.\n"
        "- RAG — если вопрос про документацию/схему/описания из docs/.\n"
        "Ответь строго одним словом: SQL или RAG."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_txt},
                  {"role": "user", "content": question}],
        temperature=0.0,
    )
    label = (resp.choices[0].message.content or "").strip().upper()
    if label == "SQL":
        return "sql", "llm"
    if label == "RAG":
        return "rag", "llm"
    return "rag", "llm:default"

def route_question(question: str, model: str = "gpt-4o-mini", use_llm_fallback: bool = True):
    mode, reason = heuristic_route(question)
    if mode != "unknown":
        return mode, reason
    if use_llm_fallback:
        return llm_route(question, model=model)
    return "rag", "default"

# --- Намерение построения графика ---
_CHART_HINTS = [
    "график", "диаграмм", "построй", "визуализ", "plot", "chart",
    "линейный график", "столбчат", "bar", "line", "scatter", "hist",
    "долю", "распределение", "динамик", "тренд"
]
def is_chart_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _CHART_HINTS)

# --- Хелперы визуализации ---
def _to_pandas(df):
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError("Ожидается Polars или Pandas DataFrame")

def _guess_roles(pdf: pd.DataFrame):
    """Эвристика выбора X/Y/категории."""
    cols = list(pdf.columns)
    if not cols:
        return {"x": None, "y_candidates": [], "cat": None}
    # явные даты
    dt_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(pdf[c])]
    if not dt_cols:
        # попробуем привести строковые колонки к дате
        for c in cols:
            if pdf[c].dtype == object:
                try:
                    pd.to_datetime(pdf[c], errors="raise")
                    dt_cols.append(c)
                    break
                except Exception:
                    pass
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(pdf[c])]
    str_cols = [c for c in cols if pdf[c].dtype == object]
    x = dt_cols[0] if dt_cols else (str_cols[0] if str_cols else None)
    y_candidates = [c for c in num_cols if c != x]
    cat = str_cols[0] if (dt_cols and str_cols) else None
    return {"x": x, "y_candidates": y_candidates, "cat": cat}

def _safe_df_for_chart_from_last_sql(click, sql: str):
    # Только SELECT
    if not re.search(r"(?is)^\s*(with\b.*?select\b|select\b)", sql):
        return None
    # Гарантируем LIMIT (не переписывая подзапросы)
    sql_limited = sql if re.search(r"(?is)\blimit\s+\d+\b", sql) else sql + "\nLIMIT 500"
    try:
        return click.query_df(sql_limited)
    except Exception:
        return None


def render_auto_chart(df, user_text: str, *, key_prefix: str = "viz"):
    """Построить график по df и тексту запроса. С уникальными ключами для виджетов."""
    pdf = _to_pandas(df).copy()
    # приведём очевидные названия дат к datetime
    for c in pdf.columns:
        if any(k in c.lower() for k in ["date", "time", "dt", "timestamp", "дата", "время"]):
            try:
                pdf[c] = pd.to_datetime(pdf[c], errors="ignore")
            except Exception:
                pass

    roles = _guess_roles(pdf)
    x, y_cands, cat = roles["x"], roles["y_candidates"], roles["cat"]
    if x is None and not y_cands:
        st.info("Недостаточно данных для графика (нет оси X или числовых столбцов).")
        return

    # Уникальные ключи для виджетов (из колонок)
    key_base = f"{key_prefix}_{hash(tuple(pdf.columns)) % 10**8}"

    chart_type = st.radio(
        "Тип графика",
        options=["auto", "line", "bar", "scatter", "hist"],
        index=0,
        horizontal=True,
        help="Выберите тип вручную, если авто-выбор не подходит.",
        key=f"{key_base}_type"
    )

    # Автовыбор
    if chart_type == "auto":
        if x is not None and np.issubdtype(pdf[x].dtype, np.datetime64):
            chart_type = "line" if y_cands else "hist"
        elif x is not None and pdf[x].dtype == object and y_cands:
            chart_type = "bar"
        elif len(y_cands) >= 2:
            chart_type = "scatter"
        elif y_cands:
            chart_type = "hist"
        else:
            chart_type = "bar"

    y = y_cands[0] if y_cands else None

    st.markdown("### Визуализация")
    st.caption(f"Тип: {chart_type}; X={x or '—'}; Y={y or '—'}; Category={cat or '—'}")

    if chart_type == "line":
        if x is None or not y:
            st.info("Для line-графика нужна ось X и числовая Y.")
            return
        fig = px.line(pdf, x=x, y=y, color=cat, markers=True, title=None)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_plot")

    elif chart_type == "bar":
        if x and y:
            fig = px.bar(pdf, x=x, y=y, color=cat, title=None)
        elif y:
            fig = px.bar(pdf, x=pdf.index, y=y, title=None)
        else:
            st.info("Нечего отображать на bar-графике.")
            return
        st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_plot")

    elif chart_type == "scatter":
        if len(y_cands) >= 2:
            y2 = y_cands[1]
            fig = px.scatter(pdf, x=y, y=y2, color=cat, hover_data=pdf.columns, title=None)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_plot")
        else:
            st.info("Для scatter нужны как минимум две числовые колонки.")
            return

    elif chart_type == "hist":
        target = y or x
        if target is None:
            st.info("Не удалось выбрать поле для гистограммы.")
            return
        fig = px.histogram(pdf, x=target, color=cat, nbins=30, title=None)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_plot")

    else:
        st.info("Неизвестный тип графика.")

# ---------- Сайдбар ----------
with st.sidebar:
    st.header("Настройки")
    model = st.selectbox(
        "Модель",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )
    override = st.selectbox("Режим (отладка)", ["Auto", "RAG", "SQL"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    system_prompt = st.text_area(
        "System prompt",
        value="Ты — полезный ассистент. Используй предоставленный контекст. Если контекста недостаточно — скажи об этом.",
        height=120,
    )

    st.divider()
    st.subheader("Ингест базы знаний")
    st.caption(f"Коллекция: {COLLECTION_NAME!r} · Путь к индексу: {CHROMA_PATH!r}")
    if st.button("Переиндексировать docs/"):
        with st.status("Индексируем документы…", expanded=True) as status:
            env = os.environ.copy()
            env["KB_COLLECTION_NAME"] = COLLECTION_NAME
            env["KB_CHROMA_PATH"] = CHROMA_PATH
            proc = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True,
                env=env,
            )
            st.code(proc.stdout or "(нет stdout)")
            if proc.returncode == 0:
                status.update(label="Готово", state="complete")
            else:
                st.error(proc.stderr)

    with st.sidebar.expander("Диагностика RAG", expanded=False):
        st.write("Working dir:", os.getcwd())
        st.write("CHROMA_PATH:", CHROMA_PATH)
        st.write("COLLECTION_NAME:", COLLECTION_NAME)
        st.write("docs/ существует?", os.path.isdir("docs"))
        st.write("Файлов .md:", len(glob.glob("docs/**/*.md", recursive=True)))
        st.write("Файлов .pdf:", len(glob.glob("docs/**/*.pdf", recursive=True)))
        try:
            import chromadb
            chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            col = chroma.get_or_create_collection(COLLECTION_NAME)
            cnt = col.count()
            st.write("Docs в коллекции:", cnt)
            if cnt > 0:
                peek = col.peek(limit=min(3, cnt))
                metas = peek.get("metadatas", [])
                docs = peek.get("documents", [])
                for i in range(len(docs)):
                    src = (metas[i] or {}).get("source")
                    path = (metas[i] or {}).get("path")
                    st.write(f"{i+1}) source={src} path={path}")
                    st.code((docs[i] or "")[:300])
        except Exception as e:
            st.error(f"Chroma peek error: {e}")

    st.divider()
    if st.button("Очистить историю", key="clear_history"):
        st.session_state["messages"] = []
        st.session_state["last_sql"] = None
        st.session_state["viz_active"] = False
        st.session_state["viz_text"] = ""
        st.rerun()

# ---------- Заголовок ----------
st.title("Единый чат: документы (RAG) + данные (SQL) — авто-роутинг")
st.caption("Пишите запросы как есть. Бот сам решит: искать в docs/ или выполнить SQL к ClickHouse. Если просите график — таблица скрывается.")

# ---------- Рендер истории ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Основной ввод ----------
user_input = st.chat_input("Введите вопрос…")
if not user_input:
# Sticky визуализация (когда нет нового сообщения)
    if st.session_state.get("viz_active"):
        last_sql = get_last_sql()
        if last_sql:
            df_sticky = _safe_fetch_df_from_sql(last_sql, limit=500)
            if df_sticky is not None:
                try:
                    render_auto_chart(df_sticky, st.session_state.get("viz_text",""), key_prefix="main_viz")
                except Exception as e:
                    st.warning(f"Не удалось построить график: {e}")
    st.stop()

# новое сообщение — сброс "липкости" (по желанию)
st.session_state["viz_active"] = False
st.session_state["viz_text"] = ""

# 1) фиксируем пользовательский ход
st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

if is_followup_sql_edit(user_input) and not st.session_state.get("last_sql"):
    recovered = _extract_last_sql_from_history()
    if recovered:
        st.session_state["last_sql"] = recovered

chart_requested = is_chart_intent(user_input)

# --- CONTEXT-FIRST ОРКЕСТРАЦИЯ ---
early = context_first_orchestrate(user_input)
if early:
    if early["action"] == "chart_from_last":
        # строим график строго из последних данных, не идём в роутинг/LLM/SQL
        st.session_state["viz_active"] = True
        st.session_state["viz_text"] = user_input
        with st.chat_message("assistant"):
            render_auto_chart(st.session_state["last_sql_df"], user_input, key_prefix="main_viz")
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "**Визуализация по последним данным (без нового SQL).**\n\n"
                "Базовый SELECT:\n"
                f"```sql\n{st.session_state.get('last_sql') or '-- недоступно --'}\n```"
            )
        })
        st.stop()

    if early["action"] == "sql_edit":
        # фиксируем принудительный SQL-путь и prev_sql для правки
        force_sql = True
        mode, decided_by = "sql", early.get("reason", "context")
        prev_sql = early["prev_sql"]
    else:
        prev_sql = None
else:
    prev_sql = None

# Если пользователь явно просит график ИМЕННО из последних данных — строим сейчас и выходим
if chart_requested and is_chart_from_last_data(user_input):
    last_sql = get_last_sql()
    if not last_sql:
        with st.chat_message("assistant"):
            st.info("Не нашёл последний SQL в истории. Сначала выполните запрос.")
        st.session_state.messages.append({"role":"assistant","content":"Нет последнего SQL для графика."})
        st.stop()

    df_last = _safe_fetch_df_from_sql(last_sql, limit=500)
    if df_last is None:
        with st.chat_message("assistant"):
            st.info("Не удалось безопасно получить данные по последнему SQL. Запустите запрос ещё раз.")
        st.session_state.messages.append({"role":"assistant","content":"Не удалось получить данные для графика."})
        st.stop()

    st.session_state["viz_active"] = True
    st.session_state["viz_text"] = user_input
    with st.chat_message("assistant"):
        render_auto_chart(df_last, user_input, key_prefix="main_viz")

    st.session_state.messages.append({
        "role":"assistant",
        "content": "**Визуализация по последнему SQL (без нового генерирования).**\n\n"
                   f"```sql\n{last_sql}\n```"
    })
    st.stop()

# chart_requested уже посчитан выше
# Блок CONTEXT-FIRST мог установить force_sql/mode/prev_sql
if "force_sql" not in locals():
    force_sql = False

if not force_sql:
    if is_followup_sql_edit(user_input) and get_last_sql():
        mode, decided_by = "sql", "followup-edit"
        force_sql = True
    else:
        mode, decided_by = route_question(user_input, model=model, use_llm_fallback=True)

if override != "Auto" and not force_sql:
    mode = "rag" if override == "RAG" else "sql"
    decided_by = f"override:{override}"

st.caption(f"Маршрутизация: {mode} ({decided_by})")

# 3) Основной роутинг
if mode == "sql":
    # --- SQL путь ---
    try:
        database = "db1"
        allowed_tables = ["total_active_users", "total_active_users_rep_mobile_total"]

        # prev_sql мог прийти из context_first_orchestrate; если нет — восстановим здесь
        if "prev_sql" not in locals() or prev_sql is None:
            prev_sql = st.session_state.get("last_sql") or _extract_last_sql_from_history() if is_followup_sql_edit(user_input) else None
        

        if prev_sql:
            question_for_sql = (
                "Измени предыдущий SELECT согласно инструкции пользователя. "
                "Не меняй логику WHERE/JOIN/агрегатов; только внеси правки к списку столбцов, "
                "группировкам, сортировкам и/или условиям. "
                f"Инструкция: {user_input}\n\n"
                f"ПРЕДЫДУЩИЙ SQL:\n{prev_sql}"
            )
        else:
            question_for_sql = user_input



        sql, df = run_sql_assistant(
            question=question_for_sql,
            database=database,
            allowed_tables=allowed_tables,
            model=model,
            # Передаём RAG-подсказки внутрь SQL-ассистента (если он это поддерживает)
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            previous_sql=prev_sql
        )

        # живой вывод
        with st.chat_message("assistant"):
            st.markdown("**Сформированный SQL:**")
            st.code(sql, language="sql")

            if chart_requested:
                # только график (без таблицы / CSV)
                st.session_state["viz_active"] = True
                st.session_state["viz_text"] = user_input
                render_auto_chart(df, user_input, key_prefix="main_viz")
            else:
                st.markdown("**Результат:**")
                st.dataframe(df.to_pandas(), use_container_width=True)
                csv_bytes = io.BytesIO()
                df.to_pandas().to_csv(csv_bytes, index=False)
                st.download_button("Скачать результат (CSV)", csv_bytes.getvalue(),
                                file_name="result.csv", mime="text/csv")


        # --- Сохраняем в историю: без превью-таблицы, если просили график ---
        if chart_requested:
            history_block = (
                "**Сформированный SQL:**\n"
                f"```sql\n{sql}\n```\n\n"
                "_Построена визуализация по результатам; табличное превью скрыто._"
            )
        else:
            try:
                preview_pd: pd.DataFrame = df.head(50).to_pandas()
                try:
                    preview_md = preview_pd.to_markdown(index=False)  # требует tabulate
                except Exception:
                    preview_md = "```\n" + preview_pd.to_csv(index=False) + "\n```"
            except Exception:
                preview_md = "_не удалось сформировать превью_"

            history_block = (
                "**Сформированный SQL:**\n"
                f"```sql\n{sql}\n```\n\n"
                f"**Превью результата (первые {min(50, len(df))} строк):**\n\n"
                f"{preview_md}"
            )

        st.session_state.messages.append({"role": "assistant", "content": history_block})

        # сохраняем состояние
        st.session_state.last_sql = sql

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Ошибка при формировании/выполнении SQL: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Ошибка: {e}"})

else:
    # --- RAG путь ---
    try:
        ctx_docs = retrieve(
            user_input,
            k=5,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
        st.caption(f"RAG: найдено чанков = {len(ctx_docs)}")
        if not ctx_docs:
            st.info("Контекст не найден. Проверьте индексацию и наличие .md/.pdf в docs/.")
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Ошибка ретрива: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Не удалось получить контекст: {e}"})
        ctx_docs = []

    context = "\n\n".join([f"[{i+1}] {d['source']}: {d['text'][:300]}..." for i, d in enumerate(ctx_docs)]) or "—"

    history_msgs = build_history_for_llm(max_turns=6)
    llm_messages = (
        [{"role": "system", "content": system_prompt}]
        + history_msgs
        + [{
            "role": "user",
            "content": (
                f"QUESTION:\n{user_input}\n\n"
                f"CONTEXT:\n{context}\n\n"
                "Правила: отвечай только по CONTEXT. Если данных нет — так и скажи."
            )
        }]
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_text = ""
        stream = client.chat.completions.create(
            model=model,
            messages=llm_messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            stream_text += delta
            placeholder.markdown(stream_text)

    st.session_state.messages.append({"role": "assistant", "content": stream_text})

    if ctx_docs:
        with st.expander("Источники"):
            for i, d in enumerate(ctx_docs, 1):
                st.write(f"[{i}] {d['source']} — {d['path']}  (score={d['score']:.4f})")

# --- Построить график по последним данным (если просили, а режим был не SQL) ---
if chart_requested and mode != "sql":
    last_sql = get_last_sql()
    if last_sql:
        df_last = _safe_fetch_df_from_sql(last_sql, limit=500)
        if df_last is not None:
            st.session_state["viz_active"] = True
            st.session_state["viz_text"] = user_input
            try:
                render_auto_chart(df_last, user_input, key_prefix="main_viz")
            except Exception as e:
                st.warning(f"Не удалось построить график: {e}")

