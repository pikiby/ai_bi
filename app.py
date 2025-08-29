# app.py
# БЛОК: импортов и базовой проверки ключа
import os
import sys
import re
import subprocess
import streamlit as st
from openai import OpenAI
from retriever import retrieve
from sql_assistant import run_sql_assistant  # <— добавили

st.set_page_config(page_title="Streamline Chat + RAG", page_icon="💬", layout="centered")

# ----- ЕДИНЫЕ настройки для Chroma -----
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

# ----- OpenAI ключ -----
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Переменная окружения OPENAI_API_KEY не задана.")
    st.stop()
client = OpenAI(api_key=api_key)

# БЛОК: сайдбар — настройки модели + кнопка запуска индексации + ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМА
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
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    system_prompt = st.text_area(
        "System prompt",
        value="Ты — полезный ассистент. Отвечай кратко и по делу. Используй только предоставленный контекст.",
        height=120,
    )

    # ← вот он, переключатель
    mode = st.radio("Режим", ["База знаний (RAG)", "Данные (SQL)"], index=0)

    st.caption("Историю чата можно очистить кнопкой ниже.")
    if st.button("Очистить историю", key="clear_history"):
        st.session_state["messages"] = []
        st.rerun()

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

# БЛОК: инициализация истории
st.session_state.setdefault("messages", [])

# БЛОК: заголовки
st.title("Chat → ChatGPT с базой знаний (RAG) и данными (SQL)")
st.caption("Слева выберите режим. Есть кнопка для переиндексации docs/.")

# БЛОК: рендер истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# БЛОК: ввод пользователя
user_input = st.chat_input("Введите вопрос…")
if user_input:
    # 1) сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if mode == "База знаний (RAG)":
        # 2) достать контекст из Chroma
        try:
            ctx_docs = retrieve(
                user_input,
                k=5,
                chroma_path=CHROMA_PATH,
                collection_name=COLLECTION_NAME
            )
        except Exception as e:
            st.error(f"Ошибка ретрива: {e}")
            # НИЧЕГО не чистим, не останавливаем весь рендер
            # Можно дописать «технический» ответ ассистента:
            st.session_state.messages.append({"role": "assistant", "content": f"Не удалось получить контекст: {e}"})
            ctx_docs = []

        context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(ctx_docs)]) or "—"
        
        # 3) собрать сообщения для модели (не путай с историей UI)
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
            "content": (
                f"QUESTION:\n{user_input}\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"Правила: отвечай только по CONTEXT. Если данных нет — так и скажи."
            )}
        ]

        # 4) потоковый ответ
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

        # 5) сохранить ответ
        st.session_state.messages.append({"role": "assistant", "content": stream_text})

        # 6) источники
        if ctx_docs:
            with st.expander("Источники"):
                for i, d in enumerate(ctx_docs, 1):
                    st.write(f"[{i}] {d['source']} — {d['path']}  (score={d['score']:.4f})")

    else:
        # --- РЕЖИМ SQL ---
        try:
            database = "db1"
            allowed_tables = ["total_active_users", "total_active_users_rep_mobile_total"]  # при необходимости

            sql, df = run_sql_assistant(
                question=user_input,
                database=database,
                allowed_tables=allowed_tables,
                model=model,
            )

            with st.chat_message("assistant"):
                st.markdown("**Сформированный SQL:**")
                st.code(sql, language="sql")
                st.markdown("**Результат:**")
                st.dataframe(df.to_pandas(), use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"SQL выполнен. Строк: {df.height}, столбцов: {df.width}."
            })

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Ошибка при формировании/выполнении SQL: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Ошибка: {e}"})
