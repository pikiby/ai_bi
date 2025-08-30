# sql_assistant.py
# ============================================================
# Генерация БЕЗОПАСНОГО SELECT к ClickHouse с поддержкой JOIN,
# с объединением двух источников знаний о схеме:
#   1) фактическая схема из БД (интроспекция),
#   2) выдержки из RAG (docs/), если доступны.
# ============================================================

import os, re
from typing import Dict, List, Tuple, Any, Optional

from openai import OpenAI
from clickhouse_client import ClickHouse_client

# ---------- БЕЗОПАСНОСТЬ ----------
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|RENAME|ATTACH|DETACH|OPTIMIZE|CREATE|KILL|SYSTEM|GRANT|REVOKE)\b",
    re.IGNORECASE
)
ONLY_SELECT = re.compile(r"^\s*(?:--.*\n|\s)*SELECT\b", re.IGNORECASE | re.DOTALL)

# Разрешим только один SELECT-стейтмент
def _single_statement(sql: str) -> bool:
    s = sql.strip()
    # допустим финальную точку с запятой, но не более одного стейтмента
    body = s[:-1] if s.endswith(";") else s
    return ";" not in body

def _clean_sql(s: str) -> str:
    # снять ```sql ... ``` и лишние украшения
    s = s.strip()
    s = re.sub(r"^```(?:sql|SQL)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # убрать -- комментарии до конца строки
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    return s.strip()

def _is_safe(sql: str) -> tuple[bool, str]:
    if not ONLY_SELECT.match(sql or ""):
        return False, "Сгенерирован не SELECT-запрос."
    if FORBIDDEN.search(sql or ""):
        return False, "Обнаружены запрещённые операции (DDL/DML)."
    if not _single_statement(sql or ""):
        return False, "Разрешён только один SELECT без нескольких выражений."
    return True, ""

# ---------- ПРЕДСТАВЛЕНИЕ СХЕМЫ ----------
def _schema_to_text(schema: Dict[str, List[Tuple[str, str]]], database: str) -> str:
    """
    Простой текст со списком таблиц и колонок:
    - <table>(col1 TYPE, col2 TYPE, ...)
    """
    parts = [f"database: {database}"]
    for table, cols in schema.items():
        cols_s = ", ".join([f"{name} {ctype}" for name, ctype in cols])
        parts.append(f"- {table}({cols_s})")
    return "\n".join(parts)

# ---------- ДОБАВКА ИЗ RAG (★ NEW) ----------
def _rag_schema_appendix(
    question: str,
    *,
    chroma_path: Optional[str],
    collection_name: Optional[str],
    rag_k: int = 6,
) -> str:
    """
    Пытаемся достать выдержки из docs/ через retriever.
    Возвращаем сжатый текст для промпта, чтобы помочь JOIN-ам:
    ключи соединения, бизнес-значения колонок, примеры DDL.
    Если retriever недоступен — тихо возвращаем пустую строку.
    """
    if not chroma_path or not collection_name:
        return ""

    try:
        from retriever import retrieve  # локальный модуль проекта
    except Exception:
        return ""

    try:
        hits = retrieve(
            question,
            k=rag_k,
            chroma_path=chroma_path,
            collection_name=collection_name,
        )
        if not hits:
            return ""
        # Соберём компактную вставку: источник + первые ~400 символов
        blocks = []
        for h in hits:
            src = (h.get("source") or "").strip()
            txt = (h.get("text") or "").strip()
            if not txt:
                continue
            snippet = txt.replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            blocks.append(f"[{src}] {snippet}")
        if not blocks:
            return ""
        header = "\n# RAG hints (from docs):\n"
        return header + "\n".join(f"- {b}" for b in blocks)
    except Exception:
        return ""

# ---------- СИСТЕМНЫЙ ПРОМПТ ДЛЯ LLM ----------
def _build_system_prompt(schema_text: str, rag_appendix: str) -> str:
    """
    Жёстко настраиваем: только SELECT, можно JOIN, просим полные имена.
    RAG-вставка идёт внизу как подсказки.
    """
    base = f"""
    Ты — помощник по ClickHouse. Твоя задача — писать ТОЛЬКО ЧИТАЮЩИЕ SELECT-запросы (один стейтмент).
    Разрешены JOIN (LEFT) и оконные функции. Запрещены любые DDL/DML: CREATE, DROP, ALTER, RENAME,
    TRUNCATE, INSERT, UPDATE, DELETE, ATTACH, DETACH, OPTIMIZE, GRANT/REVOKE, SYSTEM, KILL и т.п.

    Правила формата и безопасности:
    - ВСЕГДА используй полные имена таблиц.
    - Делай JOIN только тогда, когда пользователь просит соеденить таблицы или добавить данные из одной таблицы в другую.
    - Выбирай корректные ключи для JOIN и явно указывай ON t1.key = t2.key. Не делай CROSS JOIN.
    - Если колонка отсутствует в схеме — не выдумывай её; лучше спроси уточнение.
    - Всегда добавляй LIMIT (например 200), если пользователь не указал иначе. 
    - Если пользователь пишет "полная таблица/все данные" то не ставь LIMIT.
    - Если пользователь пишет “переименуй/удали/замени/добавь колонку” — это означает изменить ВЫБОРКУ (SELECT),
    а не базу. Используй алиасы, CASE, replace*, вычисляемые поля.
    - Все таблицы и колонки доступны только из приведённой схемы ниже.
    - Всегда используй алиасы, взятые их RAG в качестве названия столбцов и заключай их в ``, Пример `название столбца из RAG`.
    - Если пользователь не указывает дату, предпологай что пользователь просит данные за максимальную дату в таблице.

    # SCHEMA SNAPSHOT
    {schema_text}
    """
    if rag_appendix:
        base += f"\n{rag_appendix}\n"
    return base

# ---------- LLM → SQL ----------
def _question_to_sql(
    question: str,
    system_prompt: str,
    *,
    model: str = "gpt-4o-mini"
) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    return _clean_sql(raw)

# ---------- ПРОСТАЯ ПРОВЕРКА, ЧТО НЕ ВЫШЛИ ЗА СПИСОК РАЗРЕШЁННЫХ ТАБЛИЦ (★ NEW) ----------
_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([a-z0-9_]+\.[a-z0-9_]+)", re.IGNORECASE)

def _extract_used_tables(sql: str) -> List[str]:
    return [m.group(1).lower() for m in _TABLE_RE.finditer(sql or "")]

def _enforce_allowed_tables(sql: str, database: str, allowed_tables: Optional[List[str]]) -> None:
    if not allowed_tables:
        return
    allowed_fqn = {f"{database}.{t}".lower() for t in allowed_tables}
    used = set(_extract_used_tables(sql))
    bad = [t for t in used if t not in allowed_fqn]
    if bad:
        raise ValueError(f"Запрос использует неразрешённые таблицы: {', '.join(sorted(bad))}")

# ---------- ПУБЛИЧНОЕ API ----------
def run_sql_assistant(
    question: str,
    database: str,
    allowed_tables: Optional[List[str]] = None,
    *,
    model: str = "gpt-4o-mini",
    # ★ NEW: параметры для RAG
    chroma_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    rag_k: int = 6,
) -> tuple[str, Any]:
    """
    Возвращает (sql, polars_df).
    - Генерирует безопасный SELECT (с JOIN при необходимости).
    - Для понимания схемы использует снимок из БД + RAG-подсказки.
    """
    # 1) снимок схемы из БД (как и раньше)
    ch = ClickHouse_client()
    # ВАЖНО: предполагается, что ClickHouse_client.get_schema(database, tables)
    # возвращает словарь: {table: [(col, type), ...], ...}
    schema = ch.get_schema(database=database, tables=allowed_tables)  # ← это у тебя уже было
    if not schema:
        raise RuntimeError("Схема пуста — проверь права и фильтр по таблицам.")

    schema_text = _schema_to_text(schema, database)

    # 2) добавим RAG-подсказки (по вопросу) — опционально (★ NEW)
    rag_appendix = _rag_schema_appendix(
        question,
        chroma_path=chroma_path,
        collection_name=collection_name,
        rag_k=rag_k,
    )

    # 3) системный промпт: схема + RAG hints
    sys_prompt = _build_system_prompt(schema_text, rag_appendix)

    # 4) генерим SQL
    sql = _question_to_sql(question, sys_prompt, model=model)

    # 5) валидация read-only и разрешённых таблиц
    ok, why = _is_safe(sql)
    if not ok:
        raise ValueError(f"Небезопасный SQL: {why}\n\n{sql}")

    _enforce_allowed_tables(sql, database, allowed_tables)

    # 6) гарантия LIMIT
    if re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE) is None:
        sql = sql.rstrip().rstrip(";")
        sql += " LIMIT 200"

    # 7) выполняем
    df = ch.query_run(sql)
    return sql, df
