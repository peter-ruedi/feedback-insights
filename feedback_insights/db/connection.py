"""Open database connections."""

from __future__ import annotations


def connect(database_url: str | None = None):
    """Connect to Postgres with pgvector's `vector` type registered.

    Use it as a context manager: `with connect() as conn: ...` commits on
    success, rolls back on error, and closes the connection.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    if database_url is None:
        from .. import config

        database_url = config.DATABASE_URL
    conn = psycopg.connect(database_url)
    try:
        register_vector(conn)
    except Exception:
        conn.close()
        raise

    return conn
