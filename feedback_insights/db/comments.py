"""Queries on the `comments` table. Every function takes an open connection."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from ..models.feedback import Comment, DatasetStats, Hit

INSERT_SQL = """
    INSERT INTO comments (source, source_id, body, rating, created_at)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (source, source_id) DO NOTHING
"""

UNEMBEDDED_SQL = "SELECT id, body FROM comments WHERE embedding IS NULL ORDER BY id LIMIT %s"

SET_EMBEDDING_SQL = "UPDATE comments SET embedding = %s WHERE id = %s"

# `<=>` is pgvector's cosine-distance operator; ORDER BY it uses the HNSW index.
NEAREST_SQL = """
    SELECT id, body, rating, embedding <=> %(vec)s AS distance
    FROM comments
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(vec)s
    LIMIT %(k)s
"""

# One statement on purpose: under PostgreSQL's default Read Committed isolation
# each statement gets its own snapshot, so separate queries could count different
# rows if an ingest commits in between (e.g. total=4 but ratings summing to 5).
# Every subquery in a single statement shares one snapshot, so the numbers agree.
STATS_SQL = """
    SELECT
        (SELECT count(*) FROM comments),
        (SELECT count(embedding) FROM comments),
        (SELECT coalesce(jsonb_object_agg(rating::text, n), '{}'::jsonb)
           FROM (SELECT rating, count(*) AS n FROM comments
                 WHERE rating IS NOT NULL GROUP BY rating) r),
        (SELECT coalesce(array_agg(DISTINCT source ORDER BY source), '{}')
           FROM comments)
"""


def insert_comments(conn, source: str, comments: Iterable[Comment]) -> int:
    """Insert comments, skipping ones already stored; return how many were new."""
    with conn.cursor() as cur:
        cur.executemany(
            INSERT_SQL,
            [(source, c.source_id, c.body, c.rating, c.created_at) for c in comments],
        )
        # With returning=False, psycopg sets rowcount to the total rows affected by
        # all the inserts; ON CONFLICT DO NOTHING rows count 0, so this is the
        # number of new comments. Counting before and after instead could include
        # rows another ingest committed in between.
        return cur.rowcount


def unembedded(conn, limit: int) -> list[tuple[int, str]]:
    """Up to `limit` (id, body) pairs for comments that have no embedding yet."""
    with conn.cursor() as cur:
        cur.execute(UNEMBEDDED_SQL, (limit,))
        return cur.fetchall()


def set_embeddings(conn, rows: Sequence[tuple[int, object]]) -> None:
    """Store embeddings, given (id, vector) pairs."""
    with conn.cursor() as cur:
        cur.executemany(SET_EMBEDDING_SQL, [(vec, row_id) for row_id, vec in rows])


def nearest(conn, vector, k: int = 20) -> list[Hit]:
    """The k comments whose embeddings are closest to `vector`, nearest first."""
    import numpy as np

    with conn.cursor() as cur:
        cur.execute(NEAREST_SQL, {"vec": np.asarray(vector, dtype=np.float32), "k": k})
        return [Hit(id=r[0], body=r[1], rating=r[2], distance=float(r[3])) for r in cur.fetchall()]


def stats(conn) -> DatasetStats:
    """Totals, embedded count, comments per rating and the dataset names."""
    with conn.cursor() as cur:
        cur.execute(STATS_SQL)
        total, embedded, by_rating, sources = cur.fetchone()
    return DatasetStats(total_comments=total, embedded_comments=embedded,
                        by_rating=dict(sorted(by_rating.items())), sources=list(sources))


# ---- whole jobs: open their own connection and commit --------------------

def save_comments(source: str, comments: Iterable[Comment], database_url: str | None = None) -> int:
    """Store comments for a dataset, skipping ones already stored; return how many were new."""
    from .connection import connect

    with connect(database_url) as conn:
        return insert_comments(conn, source, comments)


def fill_missing_embeddings(embed: Callable[[list[str]], Sequence], batch_size: int = 256,
                            database_url: str | None = None,
                            on_batch: Callable[[int], None] | None = None) -> int:
    """Embed every comment without an embedding, `batch_size` at a time; return the count.

    `embed` turns a list of texts into vectors, one per text. Each batch is
    committed on its own, so finished batches are kept if a later one fails.
    `on_batch` gets the running total after each batch (for progress output).
    """
    from .connection import connect

    total = 0
    with connect(database_url) as conn:
        while rows := unembedded(conn, batch_size):
            vectors = embed([body for _, body in rows])
            # A short result would leave rows without embeddings, and
            # unembedded() would return them forever: fail instead of looping.
            if len(vectors) != len(rows):
                raise RuntimeError(
                    f"embed() returned {len(vectors)} vectors for {len(rows)} texts"
                )
            set_embeddings(conn, [(row_id, vec) for (row_id, _), vec in zip(rows, vectors)])
            conn.commit()
            total += len(rows)
            if on_batch:
                on_batch(total)
    return total
