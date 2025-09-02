import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from clickhouse_client import ClickHouse_client
import re
import retriever
import sys
import subprocess

# Пути/имена для базы знаний (ChromaDB)
CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")

st.set_page_config(page_title="AI SQL Assistant", page_icon="💬")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY не задан")
    st.stop()
client = OpenAI(api_key=api_key)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_df", None)

SYSTEM_PROMPT = """
Ты — SQL-ассистент для ClickHouse.
Ты всегда видишь полную историю диалога.
Правила:
- Отвечай либо SQL в блоке ```sql ... ```, либо текстом.
- SQL только SELECT/CTE. Никаких DDL/DML.
- Пользователь может ссылаться на «предыдущий запрос», «таблицу-1», «сделай график».
- Под таблицей-1,2,... понимаются результаты предыдущих SQL в истории.
- Если просят график — просто скажи «GRAPH», и я построю его по последней таблице.
"""


# Рендер истории
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


user_input = st.chat_input("Введите запрос...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Отправляем всю историю в LLM
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        temperature=0.2,
    )
    reply = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)

    # --- База знаний (RAG) ---
    with st.sidebar:
        st.header("База знаний (RAG)")
        st.caption(f"Коллекция: {COLLECTION_NAME!r} · Путь: {CHROMA_PATH!r}")

        if st.button("Переиндексировать docs/"):
            with st.status("Индексируем документы…", expanded=True) as status:
                env = os.environ.copy()
                env["KB_COLLECTION_NAME"] = COLLECTION_NAME
                env["KB_CHROMA_PATH"] = CHROMA_PATH
                proc = subprocess.run([sys.executable, "ingest.py"],
                                    capture_output=True, text=True, env=env)
                st.code(proc.stdout or "(нет stdout)")
                if proc.returncode == 0:
                    status.update(label="Готово", state="complete")
                else:
                    st.error(proc.stderr)

     # --- Проверка RAG
    m = re.search(r"```rag\\s*(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if m:
        rag_query = m.group(1).strip()
        hits = retriever.retrieve(rag_query, k=5)
        context = "\n\n".join([h["text"] for h in hits])
        msgs = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + st.session_state.messages
            + [{"role": "system", "content": f"Контекст базы знаний:\\n{context}"}]
        )
        rag_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
        )
        rag_reply = rag_resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": rag_reply})
        with st.chat_message("assistant"):
            st.markdown(rag_reply)


    # Если ответ содержит SQL → выполнить
    
    m = re.search(r"```sql\s*(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
        try:
            ch = ClickHouse_client()
            df = ch.query_run(sql)
            st.session_state["last_df"] = df
            st.dataframe(df.to_pandas(), use_container_width=True)
            csv = io.BytesIO()
            df.to_pandas().to_csv(csv, index=False)
            st.download_button("Скачать CSV", csv.getvalue(), "result.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка SQL: {e}")

    # Если явно сказали построить график → строим по last_df
    if "GRAPH" in reply.upper() and st.session_state["last_df"] is not None:
        pdf = st.session_state["last_df"].to_pandas()
        if not pdf.empty:
            col_x, col_y = pdf.columns[:2]
            fig = px.line(pdf, x=col_x, y=col_y, markers=True)
            st.plotly_chart(fig, use_container_width=True)
