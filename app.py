# app.py
# =========================
# ЕДИНЫЙ режим: авто-роутинг между RAG (документы) и SQL (ClickHouse)
# =========================

import os
import sys
import re
import subprocess
import streamlit as st
import pandas as pd  # для превью/скачивания результатов SQL
from openai import OpenAI
from retriever import retrieve
from sql_assistant import run_sql_assistant  # генерация безопасного SQL + исполнение
# --- визуализация ---
import plotly.express as px
import numpy as np
import polars as pl
import io

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



# ---------- Служебные хелперы ----------

# ---------- Автоматическая визуализация ----------

# Фразы, по которым считаем, что пользователь просит график
_CHART_HINTS = [
    "график", "диаграмм", "построй", "визуализ", "plot", "chart",
    "линейный график", "столбчат", "bar", "line", "scatter", "hist"
]

def is_chart_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _CHART_HINTS)

def _to_pandas(df):
    """Унифицируем: поддержим и Polars, и Pandas."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError("Ожидается Polars или Pandas DataFrame")

def _guess_roles(pdf):
    """
    Эвристика:
      - если есть явная дата/время → это ось X
      - если есть категориальная (строковая) → X=категория
      - числовые поля → кандидаты на Y
    Возвращает словарь {'x': <col or None>, 'y_candidates': [..], 'cat': <col or None>}
    """
    import pandas as pd
    cols = list(pdf.columns)
    if not cols:
        return {"x": None, "y_candidates": [], "cat": None}

    # типы
    dt_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(pdf[c])]
    # распознаем даты-строками (например '2025-01-01') — попробуем привести
    if not dt_cols:
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
    cat = None
    # если X — дата, попробуем найти категорию для группировки (например city)
    if dt_cols and str_cols:
        cat = str_cols[0]
    return {"x": x, "y_candidates": y_candidates, "cat": cat}

def render_auto_chart(df, user_text: str):
    """
    Строит график на основании df и текста пользователя.
    Возвращает None — всё рисует в Streamlit.
    """
    pdf = _to_pandas(df).copy()

    # Попробуем привести очевидные столбцы-даты к datetime
    for c in pdf.columns:
        if any(k in c.lower() for k in ["date", "time", "dt", "timestamp", "дата", "время"]):
            try:
                pdf[c] = pd.to_datetime(pdf[c], errors="ignore")
            except Exception:
                pass

    roles = _guess_roles(pdf)
    x, y_cands, cat = roles["x"], roles["y_candidates"], roles["cat"]

    if x is None and not y_cands:
        st.info("Недостаточно данных для графика (нет числовых или оси X).")
        return

    # Пользовательский override типа графика (удобно в отладке)
    chart_type = st.radio(
        "Тип графика",
        options=["auto", "line", "bar", "scatter", "hist"],
        index=0,
        horizontal=True,
        help="Выберите тип вручную, если авто-выбор не подходит."
    )

    # Автовыбор
    auto_type = None
    if chart_type == "auto":
        if x is not None and np.issubdtype(pdf[x].dtype, np.datetime64):
            auto_type = "line" if y_cands else "hist"
        elif x is not None and pdf[x].dtype == object and y_cands:
            auto_type = "bar"
        elif len(y_cands) >= 2:
            auto_type = "scatter"
        elif y_cands:
            auto_type = "hist"
        else:
            auto_type = "bar"
        chart_type = auto_type

    # Выбор осей (простая логика)
    y = y_cands[0] if y_cands else None

    st.markdown("### Визуализация")
    st.caption(f"Выбрано: {chart_type}; X={x or '—'}; Y={y or '—'}; Category={cat or '—'}")

    # Построение
    if chart_type == "line":
        if x is None or not y:
            st.info("Для line-графика нужна ось X и числовая Y.")
            return
        fig = px.line(pdf, x=x, y=y, color=cat, markers=True, title=None)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "bar":
        # Если есть X-строка и Y — строим bar; иначе сделаем топ по числовой
        if x and y:
            fig = px.bar(pdf, x=x, y=y, color=cat, title=None)
        elif y:
            fig = px.bar(pdf, x=pdf.index, y=y, title=None)
        else:
            st.info("Нечего отображать на bar-графике.")
            return
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "scatter":
        if len(y_cands) >= 2:
            y2 = y_cands[1]
            fig = px.scatter(pdf, x=y, y=y2, color=cat, hover_data=pdf.columns, title=None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Для scatter нужны как минимум две числовые колонки.")
            return

    elif chart_type == "hist":
        target = y or x
        if target is None:
            st.info("Не удалось выбрать поле для гистограммы.")
            return
        fig = px.histogram(pdf, x=target, color=cat, nbins=30, title=None)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Неизвестный тип графика.")


def build_history_for_llm(max_turns: int = 6):
    """
    Возвращает последние max_turns ходов (user/assistant) в формате OpenAI messages.
    Не добавляем внутрь большие таблицы — в историю мы уже кладём сжатые тексты.
    """
    msgs = []
    for m in st.session_state.messages[-max_turns:]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs

# Простые эвристики для определения намерения “SQL vs RAG”
# --- РОУТЕР: аккуратнее со словами и порогом уверенности ---
SQL_HINTS = [
    r"\bselect\b", r"\bjoin\b", r"\bwhere\b", r"\border by\b", r"\bgroup by\b",
    r"\bcount\b", r"\bsum\b", r"\bavg\b", r"\bmin\b", r"\bmax\b",
    r"\bагрег", r"\bсумм", r"\bпосчит", r"\bсколько\b", r"\bтренд\b",
]
RAG_HINTS = [
    r"\bчто такое\b", r"\bобъясн", r"\bописан", r"\bдокументац", r"\bсхем",
    r"\bddl\b", r"\bschema\b", r"\bтип пол(я|я)\b", r"\bописание таблиц",
]

def _score(patterns, text):
    return sum(1 for p in patterns if re.search(p, text))

def heuristic_route(question: str):
    q = (question or "").lower()
    score_sql = _score(SQL_HINTS, q)
    score_rag = _score(RAG_HINTS, q)

    # Требуем «запас» хотя бы в 2 балла, иначе считаем неуверенным
    if score_sql - score_rag >= 2:
        return "sql", f"heuristic:{score_sql}"
    if score_rag - score_sql >= 1:
        return "rag", f"heuristic:{score_rag}"
    return "unknown", f"heuristic:{score_sql}-{score_rag}"

def llm_route(question: str, model: str = "gpt-4o-mini"):
    """
    Фолбэк-классификация через LLM — просим вернуть строго 'SQL' или 'RAG'.
    """
    sys_txt = (
        "Классифицируй пользовательский запрос на одну из категорий:\n"
        "- SQL: если нужно посчитать метрики/сделать выборку/агрегацию из ClickHouse.\n"
        "- RAG: если вопрос про документацию/значение полей/описание схемы из docs/.\n"
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
    return "rag", "llm:default"  # по умолчанию идём в RAG

# решает, куда отправить пользовательский запрос на основе heuristic_route или llm_route, если heuristic_route не сработал
def route_question(question: str, model: str = "gpt-4o-mini", use_llm_fallback: bool = True):
    mode, reason = heuristic_route(question)
    if mode != "unknown":
        return mode, reason
    if use_llm_fallback:
        return llm_route(question, model=model)
    return "rag", "default"

def is_repeat_sql_command(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "повтори", "тот же", "как раньше", "как в прошлый раз", "с теми же данными"
    ])

# ---------- Session State ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sql", None)
st.session_state.setdefault("last_sql_df", None)

# ---------- Сайдбар ----------
with st.sidebar:
    st.header("Настройки")
    model = st.selectbox(
        "Модель",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
        ],
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
        
    with st.sidebar.expander("Диагностика RAG"):
        import os, glob
        st.write("Working dir:", os.getcwd())
        st.write("CHROMA_PATH:", CHROMA_PATH)
        st.write("COLLECTION_NAME:", COLLECTION_NAME)
        st.write("docs/ существует?", os.path.isdir("docs"))
        st.write("Файлов .md:", len(glob.glob("docs/**/*.md", recursive=True)))
        st.write("Файлов .pdf:", len(glob.glob("docs/**/*.pdf", recursive=True)))

        # Подключимся к Chroma и посмотрим, что там лежит
        try:
            import chromadb
            chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            col = chroma.get_or_create_collection(COLLECTION_NAME)

            cnt = col.count()
            st.write("Docs в коллекции:", cnt)

            if cnt > 0:
                peek = col.peek(limit=min(3, cnt))
                # покажем метаданные и первые символы текста
                metas = peek.get("metadatas", [])
                docs  = peek.get("documents", [])
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
        st.session_state["last_sql_df"] = None
        st.rerun()

# ---------- Заголовок ----------
st.title("Единый чат: документы (RAG) + данные (SQL) — авто-роутинг")
st.caption("Пишите запросы как есть. Бот сам решит: искать в docs/ или выполнить SQL к ClickHouse.")

# ---------- Рендер истории ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Основной ввод ----------
user_input = st.chat_input("Введите вопрос…")
if not user_input:
    st.stop()

# 1) фиксируем пользовательский ход
st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

# 2) авто-роутинг
mode, decided_by = route_question(user_input, model=model, use_llm_fallback=True)
if override != "Auto":
    mode = "rag" if override == "RAG" else "sql"
    decided_by = f"override:{override}"
st.caption(f"Роутер: {mode} ({decided_by})")

# 3) SQL: поддержка “повтори запрос”
if mode == "sql" and is_repeat_sql_command(user_input):
    if st.session_state.last_sql:
        try:
            from clickhouse_client import ClickHouse_client
            sql = st.session_state.last_sql
            df = ClickHouse_client().query_run(sql)

            with st.chat_message("assistant"):
                st.markdown("**Повтор предыдущего SQL:**")
                st.code(sql, language="sql")
                st.dataframe(df.to_pandas(), use_container_width=True)

                # CSV кнопка (не сохраняется в историю — намеренно)
                import io
                csv_bytes = io.BytesIO()
                df.to_pandas().to_csv(csv_bytes, index=False)
                st.download_button("Скачать результат (CSV)", csv_bytes.getvalue(),
                                   file_name="result.csv", mime="text/csv")

            # сохраняем состояние
            st.session_state.last_sql = sql
            st.session_state.last_sql_df = df

            # пишем в историю краткий блок
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Повторили предыдущий SQL. Строк: {df.height}, столбцов: {df.width}."
            })
            st.stop()
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Не удалось повторить SQL: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Ошибка повтора SQL: {e}"})
            st.stop()

# 4) Основной роутинг
if mode == "sql":
    # --- SQL путь ---
    try:
        database = "db1"
        allowed_tables = ["total_active_users", "total_active_users_rep_mobile_total"]  # при необходимости сузить

        sql, df = run_sql_assistant(
            question=user_input,
            database=database,
            allowed_tables=["total_active_users", "total_active_users_rep_mobile_total"],  # при желании сузить
            model=model,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )

        # живой вывод
        with st.chat_message("assistant"):
            st.markdown("**Сформированный SQL:**")
            st.code(sql, language="sql")
            if is_chart_intent(user_input):
            # если запрос явно про график → показываем только график
                render_auto_chart(df, user_input)
            else:
                # если нет — выводим таблицу и даём скачать
                st.markdown("**Результат:**")
                st.dataframe(df.to_pandas(), use_container_width=True)

            # кнопка CSV (живой рендер; не пишем в историю)
            csv_bytes = io.BytesIO()
            df.to_pandas().to_csv(csv_bytes, index=False)
            st.download_button("Скачать результат (CSV)", csv_bytes.getvalue(),
                               file_name="result.csv", mime="text/csv")

        # сериализуем в историю SQL + компактное превью (до 50 строк)
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

        # сохраняем состояние для будущих “повтори/измени”
        st.session_state.last_sql = sql
        st.session_state.last_sql_df = df

        # Если в вопросе просили график — построим
        if is_chart_intent(user_input):
            try:
                render_auto_chart(df, user_input)
            except Exception as e:
                st.warning(f"Не удалось построить график: {e}")

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Ошибка при формировании/выполнении SQL: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Ошибка: {e}"})

else:
    # --- RAG путь ---
    # 1) достаём контекст из Chroma
    try:
        ctx_docs = retrieve(
            user_input,
            k=5,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
        st.caption(f"RAG: найдено чанков = {len(ctx_docs)}")
        if not ctx_docs:
            st.info("Контекст не найден. Проверьте, что выполнили «Переиндексировать docs/» и что в docs/ есть .md/.pdf.")
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Ошибка ретрива: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Не удалось получить контекст: {e}"})
        ctx_docs = []

    context = "\n\n".join([f"[{i+1}] {d['source']}: {d['text'][:300]}..." for i, d in enumerate(ctx_docs)]) or "—"


    # 2) формируем сообщения для LLM: system + история + текущий вопрос с CONTEXT
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

    # 3) потоковый ответ
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

    # 4) фиксируем ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": stream_text})

    # 5) источники (живой рендер; не пишем в историю)
    if ctx_docs:
        with st.expander("Источники"):
            for i, d in enumerate(ctx_docs, 1):
                st.write(f"[{i}] {d['source']} — {d['path']}  (score={d['score']:.4f})")

# --- Построить график на основе последнего набора данных без нового запроса ---
if is_chart_intent(user_input) and st.session_state.get("last_sql_df") is not None and mode != "sql":
    try:
        render_auto_chart(st.session_state["last_sql_df"], user_input)
    except Exception as e:
        st.warning(f"Не удалось построить график из последних данных: {e}")
