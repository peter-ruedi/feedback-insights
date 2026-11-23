"""Load a CSV of comments into the `comments` table.

Expected columns: `text` (required), and optionally `id`, `rating`, `created_at`.

    python -m feedback_insights.ingest data/yelp_sample.csv --source yelp_review_full
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

MIN_CHARS = 20  # skip one-word comments; they add noise to retrieval


@dataclass(frozen=True)
class Comment:
    source_id: str
    body: str
    rating: int | None = None
    created_at: datetime | None = None


def _clean(text: str) -> str:
    # Some datasets escape newlines as literal "\n"; collapse all whitespace.
    return " ".join(text.replace("\\n", " ").split())


def _parse_rating(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    rating = int(float(value))
    if not 1 <= rating <= 5:
        raise ValueError(f"rating out of range 1-5: {value!r}")
    return rating


def _parse_date(value: str | None) -> datetime | None:
    if value is None or value.strip() == "":
        return None
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def parse_rows(rows: Iterable[dict[str, str]]) -> Iterator[Comment]:
    """Turn CSV dict rows into Comments, skipping empty or very short text.

    Rows without an `id` get their 1-based row number, so reloading the same
    file is idempotent.
    """
    for n, row in enumerate(rows, start=1):
        body = _clean(row.get("text") or "")
        if len(body) < MIN_CHARS:
            continue
        yield Comment(
            source_id=(row.get("id") or "").strip() or str(n),
            body=body,
            rating=_parse_rating(row.get("rating")),
            created_at=_parse_date(row.get("created_at")),
        )


def read_csv(path: Path) -> list[Comment]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(f"{path} needs a 'text' column; found {reader.fieldnames}")
        return list(parse_rows(reader))


INSERT_SQL = """
    INSERT INTO comments (source, source_id, body, rating, created_at)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (source, source_id) DO NOTHING
"""


def load(comments: list[Comment], source: str, database_url: str) -> int:
    """Insert comments; return how many were new."""
    import psycopg

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM comments WHERE source = %s", (source,))
        before = cur.fetchone()[0]
        cur.executemany(
            INSERT_SQL,
            [(source, c.source_id, c.body, c.rating, c.created_at) for c in comments],
        )
        cur.execute("SELECT count(*) FROM comments WHERE source = %s", (source,))
        return cur.fetchone()[0] - before


def main() -> None:
    from .config import DATABASE_URL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--source", required=True, help="dataset name stored with each row")
    args = parser.parse_args()

    comments = read_csv(args.csv_path)
    added = load(comments, args.source, DATABASE_URL)
    print(f"Read {len(comments)} comments, inserted {added} new rows.")


if __name__ == "__main__":
    main()
