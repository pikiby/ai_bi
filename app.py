import re
import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from clickhouse_client import ClickHouse_client
import retriever
import sys
import subprocess
import importlib, prompts
import plotly.graph_objects as go
import polars as pl

importlib.reload(prompts)             # гарантируем актуальную версию файла prompts.py
SYSTEM_PROMPT = prompts.CHAT_SYSTEM_PROMPT

def _parse_spec_block(spec: str) -> dict:
    params = {}
    for line in spec.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            params[k.strip().lower()] = v.strip()
    return params

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
st.session_state.setdefault("last_pivot", None)



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

# Рендер истории
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


user_input = st.chat_input("Введите запрос...")
# Фиксирует ход в истории и отрисовывает его в UI
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    #Сообщение пользователя добавляется в st.session_state.messages (глобальная история чата), 
    # и выводится пузырём “user” в Streamlit. Эта история будет использована для LLM

    # Отправляем всю историю в LLM
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
    #вызывает gpt-4o и получает черновой ответ
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        temperature=0.2,
    )
    reply = resp.choices[0].message.content

    # --- Сначала решаем: нужен ли RAG ---
    final_reply = None  # ответ, который увидит пользователь и по которому выполняем SQL/GRAPH

    # 1) Проверка RAG (не показываем промежуточный reply)
    m = re.search(r"```rag\s*(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if m:
        rag_query = m.group(1).strip()
        try:
            hits = retriever.retrieve(
                rag_query, k=5,
                chroma_path=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
            )
        except Exception as e:
            hits = []
            st.warning(f"Не удалось получить контекст из базы знаний: {e}")

        # Собираем контекст (при желании можно ограничить длину)
        context = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])

        # Второй вызов LLM уже с контекстом
        msgs = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + st.session_state.messages
            + [{"role": "system", "content": f"Контекст базы знаний:\n{context}\nОтвечай кратко и строго по контексту."}]
        )
        rag_resp = client.chat.completions.create(model="gpt-4o", messages=msgs)
        final_reply = rag_resp.choices[0].message.content
    else:
        # RAG не нужен — используем исходный ответ
        final_reply = reply

    # 2) Теперь показываем РОВНО один ответ и пишем его в историю
    st.session_state.messages.append({"role": "assistant", "content": final_reply})
    with st.chat_message("assistant"):
        st.markdown(final_reply)

    # 3) Дальше всё делаем по финальному ответу: SQL и/или GRAPH
    m_sql = re.search(r"```sql\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
    if m_sql:
        sql = m_sql.group(1).strip()
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

    # if "GRAPH" in final_reply.upper() and st.session_state["last_df"] is not None:
    #     pdf = st.session_state["last_df"].to_pandas()
    #     if not pdf.empty:
    #         col_x, col_y = pdf.columns[:2]
    #         fig = px.line(pdf, x=col_x, y=col_y, markers=True)
    #         st.plotly_chart(fig, use_container_width=True)

    # --- PLOTLY CODE (графики как код) ---
    m_plotly = re.search(r"```plotly\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
    if m_plotly:
        if st.session_state["last_df"] is None:
            st.info("Нет данных для графика: сначала выполните SQL, чтобы получить df.")
        else:
            code = m_plotly.group(1).strip()

            # Базовая песочница: запрещаем опасные конструкции
            banned = re.compile(
                r"\b(import|open|exec|eval|__|subprocess|socket|os\\.|sys\\.|Path\\(|write|remove|unlink|requests|httpx)\b",
                re.IGNORECASE,
            )
            if banned.search(code):
                st.error("Код графика отклонён (запрещённые конструкции).")
            else:
                try:
                    df = st.session_state["last_df"].to_pandas()  # df доступен коду
                    safe_globals = {"pd": pd, "px": px, "go": go, "df": df}
                    safe_locals = {}
                    exec(code, safe_globals, safe_locals)
                    fig = safe_locals.get("fig") or safe_globals.get("fig")
                    if fig is None:
                        st.error("Код не создал переменную fig.")
                    else:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка при построении графика: {e}")
    
    # --- PIVOT (сводная таблица) ---
    m_pivot = re.search(r"```pivot\s*(.*?)```", final_reply, re.DOTALL | re.IGNORECASE)
    if m_pivot and st.session_state["last_df"] is not None:
        try:
            spec = _parse_spec_block(m_pivot.group(1))
            pdf = st.session_state["last_df"].to_pandas()  # last_df у нас polars → приводим к pandas

            index = [s.strip() for s in spec.get("index", "").split(",") if s.strip()]
            columns = [s.strip() for s in spec.get("columns", "").split(",") if s.strip()]
            values = [s.strip() for s in spec.get("values", "").split(",") if s.strip()]

            agg = spec.get("aggfunc", "sum").lower()
            aggfunc = {"sum": "sum", "mean": "mean", "avg": "mean", "count": "count", "max": "max", "min": "min"}.get(agg, "sum")

            fill_raw = spec.get("fill_value", "0")
            try:
                fill_value = int(fill_raw)
            except Exception:
                fill_value = 0

            piv = pd.pivot_table(
                pdf,
                index=index or None,
                columns=columns or None,
                values=values or None,
                aggfunc=aggfunc,
                fill_value=fill_value,
            )
            piv = piv.reset_index()
            # Сохраняем как polars (чтобы остальной код не ломать)
            st.session_state["last_pivot"] = pl.from_pandas(piv)
            st.session_state["last_df"] = st.session_state["last_pivot"]

            st.markdown("**Сводная таблица:**")
            st.dataframe(piv, use_container_width=True)
            buf = io.BytesIO()
            piv.to_csv(buf, index=False)
            st.download_button("Скачать CSV (pivot)", buf.getvalue(), "pivot.csv", "text/csv")
        except Exception as e:
            st.error(f"Не удалось построить сводную таблицу: {e}")

