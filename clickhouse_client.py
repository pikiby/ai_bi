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

class ClickHouse_client:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=ClickHouse_host,
            port=ClickHouse_port,
            username=ClickHouse_username,
            password=ClickHouse_password,
            secure=True,
            verify=False
        )

    def query_run(self, query_text: str) -> pl.DataFrame:
        result = self.client.query(query_text)
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
