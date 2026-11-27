"""Embed every comment that doesn't have an embedding yet.

    python -m setup.embed

Safe to re-run: it only picks up comments that have no embedding yet.
"""

from __future__ import annotations

import argparse
import time

from feedback_insights.config import DATABASE_URL
from feedback_insights.db import comments as comments_db
from feedback_insights.embeddings import embed_texts


def embed_missing(database_url: str, batch_size: int = 256) -> int:
    """Embed every comment that has no embedding yet; return how many were embedded."""
    return comments_db.fill_missing_embeddings(
        embed_texts, batch_size, database_url,
        on_batch=lambda total: print(f"  embedded {total} so far"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    start = time.perf_counter()
    n = embed_missing(DATABASE_URL, args.batch_size)
    print(f"Embedded {n} comments in {time.perf_counter() - start:.1f}s.")


if __name__ == "__main__":
    main()
