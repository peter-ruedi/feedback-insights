"""Find the comments closest in meaning to a question."""

from __future__ import annotations

from .db import comments as comments_db
from .models.feedback import Hit


def search(conn, question: str, k: int = 20) -> list[Hit]:
    """Embed the question with the same model as the comments, then search."""
    from .embeddings import embed_texts

    [vector] = embed_texts([question])
    return comments_db.nearest(conn, vector, k)
