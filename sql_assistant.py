# sql_assistant.py
import os, re
from typing import Dict, List, Tuple, Any
from openai import OpenAI
from clickhouse_client import ClickHouse_client

# --- жёсткие правила безопасности ---
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|RENAME|ATTACH|DETACH|OPTIMIZE|CREATE|KILL|SYSTEM)\b",
    re.IGNORECASE
)
ONLY_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE | re.DOTALL)

def _clean_sql(s: str) -> str:
    # снять ```sql ... ``` и лишние украшения
    s = s.strip()
    s = re.sub(r"^```(?:sql|SQL)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # убрать комментарии -- ... до конца строки (оставим /* */ при желании дополнить)
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    return s.strip()

def _is_safe(sql: str) -> tuple[bool, str]:
    if not ONLY_SELECT.match(sql):
        return False, "Сгенерирован не SELECT-запрос."
    if FORBIDDEN.search(sql):
        return False, "Обнаружены запрещённые операции (DDL/DML)."
    # ограничим количество запросов: один statement
    if ";" in sql.strip()[:-1]:
        return False, "Разрешён только один SELECT без нескольких выражений."
    return True, ""

def _schema_to_text(schema: Dict[str, List[Tuple[str, str]]], database: str) -> str:
    # компактное текстовое представление
    parts = [f"database: {database}"]
    for table, cols in schema.items():
        cols_s = ", ".join([f"{name} {ctype}" for name, ctype in cols])
        parts.append(f"- {table}({cols_s})")
    return "\n".join(parts)

def _build_system_prompt(schema_text: str) -> str:
    return (
        "Ты — помощник по аналитическим запросам ClickHouse.\n"
        "Пиши строго один безопасный SQL для ClickHouse, только SELECT с LIMIT.\n"
        "Не используй запрещённые операции (DDL/DML). Не используй внешние таблицы.\n"
        "Учитывай доступную схему ниже, выбирай корректные имена таблиц и колонок.\n\n"
        "СХЕМА:\n" + schema_text + "\n\n"
        "Требования к ответу: верни только SQL, без пояснений, без markdown-блоков."
    )

def _question_to_sql(question: str, schema_text: str, *, model="gpt-4o-mini") -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": _build_system_prompt(schema_text)},
        {"role": "user", "content": question}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    return _clean_sql(raw)

def run_sql_assistant(question: str,
                      database: str,
                      allowed_tables: list[str] | None = None,
                      *,
                      model: str = "gpt-4o-mini") -> tuple[str, Any]:
    """
    Возвращает (sql, polars_df).
    """
    # 1) снимок схемы
    ch = ClickHouse_client()
    schema = ch.get_schema(database=database, tables=allowed_tables)
    if not schema:
        raise RuntimeError("Схема пуста — проверь права и фильтр по таблицам.")

    schema_text = _schema_to_text(schema, database)

    # 2) генерим SQL
    sql = _question_to_sql(question, schema_text, model=model)

    # 3) валидируем
    ok, why = _is_safe(sql)
    if not ok:
        raise ValueError(f"Небезопасный SQL: {why}\n\n{sql}")

    # 4) гарантия LIMIT на больших таблицах
    if re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE) is None:
        sql = sql.rstrip().rstrip(";")
        sql += " LIMIT 100"

    # 5) исполняем
    df = ch.query_run(sql)
    return sql, df
