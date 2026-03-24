"""Database connection and query helpers."""
import pymssql
import pandas as pd
import os

SERVER   = "fc-dwlive.weaverfintech.com"
DATABASE = "FC_STAGE"
DOMAIN   = "HOMECHOICE"
USERNAME = "reberle1"
PASSWORD = os.environ.get("DB_PASSWORD", "Home@Choice#")


def _new_connection(timeout: int = 300):
    return pymssql.connect(
        server=SERVER,
        user=f"{DOMAIN}\\{USERNAME}",
        password=PASSWORD,
        database=DATABASE,
        timeout=timeout,
        login_timeout=15,
    )


def query(sql: str, params=None, timeout: int = 300) -> pd.DataFrame:
    conn = _new_connection(timeout=timeout)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
