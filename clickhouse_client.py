# clickhouse_client.py
import clickhouse_connect
import os
import pandas as pd
import polars as pl
from dotenv import load_dotenv

load_dotenv()
ClickHouse_host = os.getenv('ClickHouse_host')
ClickHouse_port = os.getenv('ClickHouse_port')
ClickHouse_username = os.getenv('ClickHouse_username')
ClickHouse_password = os.getenv('ClickHouse_password')
CLICKHOUSE_DB = os.getenv('CLICKHOUSE_DB', 'db1')

class ClickHouse_client:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=ClickHouse_host,
            port=ClickHouse_port,
            username=ClickHouse_username,
            password=ClickHouse_password,
            secure=True,
            verify=False,
            database=CLICKHOUSE_DB
        )

    def query_run(self, query_text: str) -> pl.DataFrame:
        # Базовый запуск запроса с страховкой от `db1.db1.` при UNKNOWN_TABLE (Code: 60)
        try:
            result = self.client.query(query_text)
        except Exception as e:
            msg = str(e)
            # Если таблица «не найдена» и видно удвоенный префикс БД — выполним повтор с заменой
            dup = f"{CLICKHOUSE_DB}.{CLICKHOUSE_DB}."
            if ("Code: 60" in msg or "UNKNOWN_TABLE" in msg) and dup in query_text:
                fixed_query = query_text.replace(dup, f"{CLICKHOUSE_DB}.")
                result = self.client.query(fixed_query)
            else:
                raise
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        return pl.from_pandas(df)

    def get_schema(self, database: str, tables: list[str] | None = None) -> dict:
        """
        Возвращает снимок схемы: {table: [(name, type), ...], ...}
        """
        tb_filter = ""
        if tables:
            quoted = ",".join([f"'{t}'" for t in tables])
            tb_filter = f" AND table IN ({quoted})"
        sql = f"""
        SELECT table, name, type
        FROM system.columns
        WHERE database = '{database}' {tb_filter}
        ORDER BY table, position
        """
        res = self.client.query(sql)
        rows = list(res.result_rows)
        schema: dict[str, list[tuple[str, str]]] = {}
        for table, name, ctype in rows:
            schema.setdefault(table, []).append((name, ctype))
        return schema

    # -------------------- Saved Queries helpers (ClickHouse) --------------------
    # Сохранение/чтение пользовательских запросов (строки кода),
    # используем общую таблицу saved_queries в текущей БД (CLICKHOUSE_DB).

    def list_saved_queries(self, user_uuid: str, search_text: str | None = None, limit: int = 200) -> list[dict]:
        """Вернёт список сохранённых элементов для сайдбара.
        Только не удалённые, отсортированные по updated_at DESC.
        Поиск — только по названию (подстрока, нечувствительно к регистру).
        """
        where = ["is_deleted = 0"]
        # Общий каталог: фильтруем строго по данному user_uuid (в дальнейшем можно расширить IN (...))
        where.append(f"user_uuid = '{user_uuid}'")
        if search_text:
            # ILIKE доступен в CH начиная с определённых версий; для совместимости используем lower(title) LIKE lower('%...%')
            s = search_text.replace("'", "''")
            where.append(f"lower(title) LIKE lower('%{s}%')")
        sql = f"""
            SELECT toString(item_uuid) as item_uuid, title, updated_at
            FROM saved_queries FINAL
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC
            LIMIT {int(limit)}
        """
        try:
            res = self.client.query(sql)
            rows = list(res.result_rows)
            cols = res.column_names
        except Exception:
            return []
        out = []
        for r in rows:
            d = {cols[i]: r[i] for i in range(len(cols))}
            out.append(d)
        return out

    def get_saved_query(self, user_uuid: str, item_uuid: str) -> dict | None:
        """Вернёт полную запись сохранённого элемента по user_uuid + item_uuid."""
        uid = item_uuid.replace("'", "''")
        sql = f"""
            SELECT toString(user_uuid) as user_uuid,
                   toString(item_uuid) as item_uuid,
                   title, db, sql_code, table_code, plotly_code,
                   ifNull(pandas_code, '') as pandas_code,
                   is_deleted, created_at, updated_at
            FROM saved_queries FINAL
            WHERE user_uuid = '{user_uuid}' AND item_uuid = '{uid}' AND is_deleted = 0
            LIMIT 1
        """
        try:
            res = self.client.query(sql)
            if not res.result_rows:
                return None
            row = res.result_rows[0]
            cols = res.column_names
            return {cols[i]: row[i] for i in range(len(cols))}
        except Exception:
            return None

    def insert_saved_query(
        self,
        user_uuid: str,
        item_uuid: str,
        title: str,
        db: str,
        sql_code: str,
        table_code: str | None = None,
        plotly_code: str | None = None,
        pandas_code: str | None = None,
    ) -> None:
        """Добавляет новую запись в saved_queries. Поля created_at/updated_at задаются по умолчанию в CH."""
        table_code = table_code or ""
        plotly_code = plotly_code or ""
        pandas_code = pandas_code or ""
        data = [[user_uuid, item_uuid, title, db or "", sql_code, table_code, plotly_code, pandas_code, 0]]
        # Порядок аргументов: insert(table, data, column_names=[...])
        self.client.insert(
            "saved_queries",
            data,
            column_names=[
                "user_uuid",
                "item_uuid",
                "title",
                "db",
                "sql_code",
                "table_code",
                "plotly_code",
                "pandas_code",
                "is_deleted",
            ],
        )

    def rename_saved_query(self, user_uuid: str, item_uuid: str, new_title: str) -> None:
        """Переименовать элемент через вставку новой версии строки (ReplacingMergeTree)."""
        rec = self.get_saved_query(user_uuid, item_uuid)
        if not rec:
            return
        # Вставляем новую строку с тем же ключом и обновлённым title
        self.insert_saved_query(
            user_uuid=user_uuid,
            item_uuid=item_uuid,
            title=new_title,
            db=rec.get("db") or "",
            sql_code=rec.get("sql_code") or "",
            table_code=rec.get("table_code") or "",
            plotly_code=rec.get("plotly_code") or "",
            pandas_code=rec.get("pandas_code") or "",
        )

    def soft_delete_saved_query(self, user_uuid: str, item_uuid: str) -> None:
        """Мягкое удаление через вставку новой версии строки (is_deleted = 1)."""
        rec = self.get_saved_query(user_uuid, item_uuid)
        if not rec:
            return
        data = [[
            user_uuid,
            item_uuid,
            rec.get("title") or "",
            rec.get("db") or "",
            rec.get("sql_code") or "",
            rec.get("table_code") or "",
            rec.get("plotly_code") or "",
            rec.get("pandas_code") or "",
            1,  # is_deleted
        ]]
        self.client.insert(
            "saved_queries",
            data,
            column_names=[
                "user_uuid",
                "item_uuid",
                "title",
                "db",
                "sql_code",
                "table_code",
                "plotly_code",
                "pandas_code",
                "is_deleted",
            ],
        )
