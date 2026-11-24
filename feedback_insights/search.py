"""Find the comments closest in meaning to a question."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    id: int
    body: str
    rating: int | None
    distance: float  # cosine distance: 0 = same meaning, 2 = opposite


# `<=>` is pgvector's cosine-distance operator; ORDER BY it uses the HNSW index.
SEARCH_SQL = """
    SELECT id, body, rating, embedding <=> %(vec)s AS distance
    FROM comments
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(vec)s
    LIMIT %(k)s
"""


def search_by_vector(conn, vector, k: int = 20) -> list[Hit]:
    """Return the k nearest comments to `vector`. `conn` needs register_vector()."""
    import numpy as np

    with conn.cursor() as cur:
        cur.execute(SEARCH_SQL, {"vec": np.asarray(vector, dtype=np.float32), "k": k})
        return [Hit(id=r[0], body=r[1], rating=r[2], distance=float(r[3])) for r in cur.fetchall()]


def search(conn, question: str, k: int = 20) -> list[Hit]:
    """Embed the question with the same model as the comments, then search."""
    from .embed import embed_texts

    [vector] = embed_texts([question])
    return search_by_vector(conn, vector, k)
