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
from datetime import datetime

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

# Инициализация ключей состояния.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_df", None)
st.session_state.setdefault("last_pivot", None)
t.session_state.setdefault("results", []) # история результатов (список элементов)
st.session_state.setdefault("run_counter", 0) # счётчик уникальных id для результатов



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
    st.header("Экспорт")
    if not st.session_state["results"]:
        st.caption("Нет результатов для экспорта.")
    else:
        # Собираем ZIP целиком в память
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in st.session_state["results"]:
                base = f"{item['kind']}_{item['id']}"
                # данные (если есть)
                if item.get("df") is not None:
                    pdf = item["df"].to_pandas()
                    zf.writestr(f"{base}.csv", _df_to_csv_bytes(pdf))
                    zf.writestr(f"{base}.xlsx", _df_to_xlsx_bytes(pdf, sheet_name=item['kind'][:31]))
                # график (если есть)
                if item.get("fig") is not None:
                    zf.writestr(f"{base}.html", _fig_to_html_bytes(item["fig"]))
                # SQL (как текст, если есть)
                sql = (item.get("meta") or {}).get("sql")
                if sql:
                    zf.writestr(f"{base}.sql.txt", sql.encode("utf-8"))
        zip_buf.seek(0)

        st.download_button(
            "Скачать все результаты (ZIP)",
            data=zip_buf.getvalue(),
            file_name="results_export.zip",
            mime="application/zip",
            use_container_width=True,
        )

# ---------- Вспомогательные функции сохранения/рендера ----------
# Разместите эти функции выше основной логики чат-цикла, чтобы их было видно везде.
# Они инкапсулируют конвертацию данных в байты, добавление в историю и отрисовку блока.

def _df_to_csv_bytes(pdf: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pdf.to_csv(buf, index=False)
    return buf.getvalue()

def _df_to_xlsx_bytes(pdf: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pdf.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buf.seek(0)
    return buf.getvalue()

def _fig_to_html_bytes(fig) -> bytes:
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    return html.encode("utf-8")

def _push_result(kind: str, *, df_pl: pl.DataFrame | None = None, fig=None, meta: dict | None = None):
    """
    Сохраняем результат в историю.
    kind: "table" | "pivot" | "chart"
    df_pl: polars DataFrame для данных блока
    fig: plotly Figure для графиков
    meta: произвольная мета (sql, note, и т.п.)
    """
    st.session_state["run_counter"] += 1
    item = {
        "id": st.session_state["run_counter"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "meta": meta or {},
    }
    if df_pl is not None:
        item["df"] = df_pl  # храним как polars; конвертируем на скачивание/рендер
    if fig is not None:
        item["fig"] = fig
    st.session_state["results"].append(item)

def _render_result(item):
    """Рисуем один блок истории + кнопки скачивания."""
    kind = item["kind"]
    meta = item.get("meta", {})
    title = {
        "table": "Таблица",
        "pivot": "Сводная таблица",
        "chart": "График",
    }.get(kind, "Результат")

    with st.container(border=True):
        st.caption(f"{title} • #{item['id']} • {item['timestamp']}")

        if "df" in item and item["df"] is not None:
            pdf = item["df"].to_pandas()
            st.dataframe(pdf, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Скачать CSV",
                    _df_to_csv_bytes(pdf),
                    file_name=f"{title.lower()}_{item['id']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "Скачать Excel",
                    _df_to_xlsx_bytes(pdf, sheet_name=f"{title[:25]}"),
                    file_name=f"{title.lower()}_{item['id']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        if "fig" in item and item["fig"] is not None:
            st.plotly_chart(item["fig"], use_container_width=True)
            st.download_button(
                "Скачать график (HTML)",
                _fig_to_html_bytes(item["fig"]),
                file_name=f"chart_{item['id']}.html",
                mime="text/html",
                use_container_width=True,
            )

        # полезно подсветить SQL или заметку, если есть
        if meta.get("sql"):
            with st.expander("Показать SQL"):
                st.code(meta["sql"], language="sql")


# Рендер истории
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Рендер всех накопленных результатов (таблицы/сводные/графики)
if st.session_state["results"]:
    st.markdown("### История результатов")
    for item in st.session_state["results"]:
        _render_result(item)







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
    f m_sql:
    sql = m_sql.group(1).strip()
    try:
        ch = ClickHouse_client()
        df_pl = ch.query_run(sql)
        st.session_state["last_df"] = df_pl

        # Сохраняем в историю
        _push_result("table", df_pl=df_pl, meta={"sql": sql})

        # Рендерим только что добавленный элемент (последний)
        _render_result(st.session_state["results"][-1])

    except Exception as e:
        st.error(f"Ошибка SQL: {e}")

    # if "GRAPH" in final_reply.upper() and st.session_state["last_df"] is not None:
    #     pdf = st.session_state["last_df"].to_pandas()
    #     if not pdf.empty:
    #         col_x, col_y = pdf.columns[:2]
    #         fig = px.line(pdf, x=col_x, y=col_y, markers=True)
    #         st.plotly_chart(fig, use_container_width=True)

    # --- PLOTLY CODE (графики как код) ---
    if m_plotly:
    if st.session_state["last_df"] is None:
        st.info("Нет данных для графика: сначала выполните SQL, чтобы получить df.")
    else:
        code = m_plotly.group(1).strip()
        BANNED_RE = re.compile(
            r"(?:\bimport\b|\bopen\b|\bexec\b|\beval\b|__|subprocess|socket|"
            r"os\.[A-Za-z_]+|sys\.[A-Za-z_]+|Path\(|write\(|remove\(|unlink\(|requests|httpx)",
            re.IGNORECASE,
        )
        if BANNED_RE.search(code):
            st.error("Код графика отклонён (запрещённые конструкции).")
        else:
            try:
                df = st.session_state["last_df"].to_pandas()
                safe_globals = {"pd": pd, "px": px, "go": go, "df": df}
                safe_locals = {}
                exec(code, safe_globals, safe_locals)
                fig = safe_locals.get("fig") or safe_globals.get("fig")
                if fig is None:
                    st.error("Код не создал переменную fig.")
                else:
                    # Сохраняем в историю (данные для графика тоже полезно положить)
                    _push_result("chart", df_pl=st.session_state["last_df"], fig=fig)
                    _render_result(st.session_state["results"][-1])
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
            # ... после формирования piv (pandas) и конвертации в polars:
            st.session_state["last_pivot"] = pl.from_pandas(piv)
            st.session_state["last_df"] = st.session_state["last_pivot"]

            # Сохраняем как отдельный результат истории
            _push_result("pivot", df_pl=st.session_state["last_pivot"])

            # Рисуем элемент истории со своими кнопками скачивания
            _render_result(st.session_state["results"][-1])

            st.markdown("**Сводная таблица:**")
            buf = io.BytesIO()
            piv.to_csv(buf, index=False)
        except Exception as e:
            st.error(f"Не удалось построить сводную таблицу: {e}")

