"""Local SQLite storage shared by the BSD Bank customer and admin pages."""
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
import os
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = Path(os.environ.get("BSD_BANK_SQLITE_PATH", ROOT / "data" / "customer_queries.sqlite3")).expanduser()


def _database_url() -> str | None:
    return (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
            or os.environ.get("POSTGRES_URL_UNPOOLED") or os.environ.get("POSTGRES_URL_NON_POOLING"))


def _connect():
    database_url = _database_url()
    if database_url:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL storage requires psycopg") from exc
        return psycopg.connect(database_url, row_factory=dict_row, connect_timeout=5)
    if os.environ.get("VERCEL"):
        raise RuntimeError("DATABASE_URL must be configured on Vercel; SQLite is local-only")
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_store() -> None:
    with _connection() as connection:
        if _database_url():
            connection.execute("""
                CREATE TABLE IF NOT EXISTS customer_queries (
                    id BIGSERIAL PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    submitted_at TIMESTAMPTZ NOT NULL
                )
            """)
        else:
            connection.execute("""
            CREATE TABLE IF NOT EXISTS customer_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                query TEXT NOT NULL,
                intent TEXT NOT NULL,
                confidence REAL NOT NULL,
                submitted_at TEXT NOT NULL
            )
        """)


def save_submission(customer_name: str, query: str, intent: str, confidence: float) -> int:
    with _connection() as connection:
        timestamp = datetime.now(timezone.utc).astimezone()
        placeholder = "%s" if _database_url() else "?"
        cursor = connection.execute(
            f"""INSERT INTO customer_queries
               (customer_name, query, intent, confidence, submitted_at)
               VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
               RETURNING id""",
            (customer_name, query, intent, float(confidence), timestamp),
        )
        inserted = cursor.fetchone()
        return int(inserted["id"] if _database_url() else inserted[0])


def get_submissions() -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """SELECT id, submitted_at, customer_name, query, intent, confidence
               FROM customer_queries ORDER BY id DESC"""
        ).fetchall()
    return [dict(row) for row in rows]
