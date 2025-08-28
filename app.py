import os
import streamlit as st
from openai import OpenAI

# ---------- Настройки страницы ----------
st.set_page_config(page_title="Streamline Chat", page_icon="💬", layout="centered")

# ---------- Сайдбар с настройками ----------
with st.sidebar:
    st.header("Настройки")
    model = st.selectbox(
        "Модель",
        options=[
            "gpt-4o-mini",   # быстрый/дешёвый
            "gpt-4o",        # качественнее
            "gpt-4.1-mini",  # альтернатива
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    system_prompt = st.text_area(
        "System prompt",
        value="Ты — полезный ассистент. Отвечай кратко и по делу.",
        height=120,
    )
    st.caption("Историю чата можно очистить кнопкой ниже.")
    if st.button("Очистить историю"):
        st.session_state.messages = []

# ---------- Клиент OpenAI ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Переменная окружения OPENAI_API_KEY не задана.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------- Инициализация истории ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Заголовок ----------
st.title("Chat → ChatGPT")
st.caption("Интерфейс чат-бота на Streamline/Streamlit, подключённый к OpenAI.")

# ---------- Рендер истории ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Ввод пользователя ----------
user_input = st.chat_input("Напишите сообщение...")
if user_input:
    # 1) Показать сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Потоковый ответ ассистента
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_text = ""

        # Собираем сообщения: system + история
        messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]

        # Потоковая генерация
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

    # 3) Сохраняем ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": stream_text})
