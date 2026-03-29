import os
import re
import importlib

try:
    _pg = importlib.import_module("psycopg2")
except ModuleNotFoundError:
    try:
        _pg = importlib.import_module("psycopg")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "No PostgreSQL driver found. Install one of: 'psycopg2-binary' or 'psycopg'."
        ) from e

IntegrityError = getattr(_pg, "IntegrityError", Exception)


def _convert_placeholders(query: str) -> str:
    # Convert sqlite-style placeholders to psycopg placeholders.
    return re.sub(r"\?", "%s", query)


class CursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        q = _convert_placeholders(query)
        if params is None:
            return self._cursor.execute(q)
        return self._cursor.execute(q, params)

    def executemany(self, query, seq_of_params):
        q = _convert_placeholders(query)
        return self._cursor.executemany(q, seq_of_params)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)

    def close(self):
        return self._cursor.close()


class ConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return CursorWrapper(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()


def connect(dsn=None):
    db_url = dsn or os.getenv("DATABASE_URL") or "postgresql://smartdoc:smartdoc@localhost:5432/smartdoc"
    try:
        conn = _pg.connect(db_url)
    except Exception as e:
        raise RuntimeError(
            "Database connection failed. Set DATABASE_URL for your environment. "
            "For local runs, use postgresql://smartdoc:smartdoc@localhost:5432/smartdoc. "
            "For docker-compose app container, use postgresql://smartdoc:smartdoc@db:5432/smartdoc. "
            f"Original error: {e}"
        ) from e
    conn.autocommit = False
    return ConnectionWrapper(conn)
