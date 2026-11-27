"""Turn text into vectors with a local sentence-transformers model.

Used at runtime to embed questions, and by setup/embed.py to embed comments.
"""

from __future__ import annotations

from .config import EMBEDDING_DIM, EMBEDDING_MODEL

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
                f"{EMBEDDING_MODEL} outputs {dim} dims but the database expects "
                f"{EMBEDDING_DIM}; change the embedding column size in db/init.sql"
            )
    return _model


def embed_texts(texts: list[str]):
    """Return unit-length vectors, so cosine distance and dot product agree.
    """
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64)
