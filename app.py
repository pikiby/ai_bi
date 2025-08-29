# app.py
# БЛОК: импортов и базовой проверки ключа
import os
import sys
import re
import subprocess
import streamlit as st
from openai import OpenAI
from retriever import retrieve

st.set_page_config(page_title="Streamline Chat + RAG", page_icon="💬", layout="centered")

# ----- ЕДИНЫЕ настройки для Chroma -----
# можно переопределить через переменные окружения на проде
CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")

def _validate_collection_name(name: str) -> str:
    n = (name or "").strip().lower()
    # максимально совместимый паттерн для Chroma (избегаем дефиса на старых версиях)
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

# БЛОК: сайдбар — настройки модели + кнопка запуска индексации
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

    st.caption("Историю чата можно очистить кнопкой ниже.")
    if st.button("Очистить историю"):
        st.session_state.messages = []

    st.divider()
    st.subheader("Ингест базы знаний")
    st.caption(f"Коллекция: {COLLECTION_NAME!r} · Путь к индексу: {CHROMA_PATH!r}")
    if st.button("Переиндексировать docs/"):
        with st.status("Индексируем документы…", expanded=True) as status:
            # Передаём имя коллекции и путь в ingest.py
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
if "messages" not in st.session_state:
    st.session_state.messages = []

# БЛОК: заголовки
st.title("Chat → ChatGPT с базой знаний (RAG)")
st.caption("Слева можно запустить индексацию. Вопросы внизу — ответы с контекстом из docs/.")

# БЛОК: рендер истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# БЛОК: ввод пользователя
user_input = st.chat_input("Вопрос по вашей базе знаний…")
if user_input:
    # 1) показать сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) достать контекст из векторного индекса (Ретривер)
    try:
        ctx_docs = retrieve(
            user_input,
            k=5,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        st.error(f"Ошибка ретрива: {e}")
        st.stop()

    context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(ctx_docs)]) or "—"

    # 3) собрать сообщения для модели
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({
        "role": "user",
        "content": (
            f"QUESTION:\n{user_input}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Правила: отвечай только по CONTEXT. Если данных нет — так и скажи."
        )
    })

    # 4) потоковый ответ ассистента
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_text = ""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            stream_text += delta
            placeholder.markdown(stream_text)

    # 5) сохранить ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": stream_text})

    # 6) показать источники
    if ctx_docs:
        with st.expander("Источники"):
            for i, d in enumerate(ctx_docs, 1):
                st.write(f"[{i}] {d['source']} — {d['path']}  (score={d['score']:.4f})")
