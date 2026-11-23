"""Embed every comment that doesn't have an embedding yet.

    python -m feedback_insights.embed

Safe to re-run: it only picks up rows where embedding IS NULL.
"""

from __future__ import annotations

import argparse
import time

from .config import DATABASE_URL, EMBEDDING_DIM, EMBEDDING_MODEL

_model = None


def get_model():
    """Load the sentence-transformers model once (first call downloads ~90 MB)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
        dim = _model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"{EMBEDDING_MODEL} outputs {dim} dims but the table expects "
                f"{EMBEDDING_DIM}; change vector({EMBEDDING_DIM}) in db/init.sql"
            )
    return _model


def embed_texts(texts: list[str]):
    """Return unit-length vectors, so cosine distance and dot product agree.

    Saturday afternoon's /ask endpoint reuses this to embed the question.
    """
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64)


def embed_missing(database_url: str, batch_size: int = 256) -> int:
    import psycopg
    from pgvector.psycopg import register_vector

    total = 0
    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, body FROM comments WHERE embedding IS NULL ORDER BY id LIMIT %s",
                    (batch_size,),
                )
                rows = cur.fetchall()
                if not rows:
                    return total
                vectors = embed_texts([body for _, body in rows])
                cur.executemany(
                    "UPDATE comments SET embedding = %s WHERE id = %s",
                    [(vec, row_id) for vec, (row_id, _) in zip(vectors, rows)],
                )
            conn.commit()
            total += len(rows)
            print(f"  embedded {total} so far")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    start = time.perf_counter()
    n = embed_missing(DATABASE_URL, args.batch_size)
    print(f"Embedded {n} comments in {time.perf_counter() - start:.1f}s.")


if __name__ == "__main__":
    main()
