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

import re
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
            allowed_tables=allowed_tables,
            model=model,
        )

        # живой вывод
        with st.chat_message("assistant"):
            st.markdown("**Сформированный SQL:**")
            st.code(sql, language="sql")
            st.markdown("**Результат:**")
            st.dataframe(df.to_pandas(), use_container_width=True)

            # кнопка CSV (живой рендер; не пишем в историю)
            import io
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
