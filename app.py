# app.py
# БЛОК: импортов и базовой проверки ключа
import os
import sys
import subprocess
import streamlit as st
from openai import OpenAI
from retriever import retrieve

st.set_page_config(page_title="Streamline Chat + RAG", page_icon="💬", layout="centered")

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
            "gpt-4o-mini",   # быстрый/дешёвый
            "gpt-4o",        # качественнее
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
    st.caption("Проиндексировать все файлы из папки docs/ в локальный индекс Chroma.")
    if st.button("Переиндексировать docs/"):
        with st.status("Индексируем документы…", expanded=True) as status:
            # ВЫЗОВ ingest.py КАК ОТДЕЛЬНОГО ПРОЦЕССА
            proc = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True
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
    ctx_docs = retrieve(user_input, k=5, chroma_path="data/chroma", collection_name="kb_docs")
    context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(ctx_docs)])

    # 3) собрать сообщения для модели
    messages = [{"role": "system", "content": system_prompt}]
    # Добавляем строгую инструкцию и контекст
    messages.append({
        "role": "user",
        "content": f"QUESTION:\n{user_input}\n\nCONTEXT:\n{context}\n\n"
                   f"Правила: отвечай только по CONTEXT. Если данных нет — так и скажи."
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
