"""Runs the real pgvector query against a database.

Skipped unless TEST_DATABASE_URL is set. Uses a temporary table, so it never
touches your loaded comments:

    TEST_DATABASE_URL=postgresql://feedback:feedback@localhost:5432/feedback pytest
"""

import os

import pytest

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


def unit(*xs):
    """A 384-dim unit vector with the given leading values (rest zero)."""
    import numpy as np

    v = np.zeros(384, dtype=np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


@pytest.fixture
def conn():
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(URL) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(c)
        # A temp table named `comments` shadows the real one for this session only.
        c.execute("""CREATE TEMP TABLE comments (
            id BIGINT PRIMARY KEY, body TEXT, rating SMALLINT, embedding vector(384))""")
        rows = [
            (1, "about prices", 2, unit(1, 0)),
            (2, "mostly prices", 3, unit(0.9, 0.1)),
            (3, "about service", 5, unit(0, 1)),
            (4, "not embedded yet", 4, None),
        ]
        c.cursor().executemany("INSERT INTO comments VALUES (%s, %s, %s, %s)", rows)
        yield c


def test_returns_nearest_first(conn):
    from feedback_insights.search import search_by_vector

    hits = search_by_vector(conn, unit(1, 0), k=3)
    assert [h.id for h in hits] == [1, 2, 3]
    assert hits[0].distance == pytest.approx(0, abs=1e-5)
    assert hits[0].distance < hits[1].distance < hits[2].distance


def test_respects_k_and_skips_rows_without_embeddings(conn):
    from feedback_insights.search import search_by_vector

    assert [h.id for h in search_by_vector(conn, unit(0, 1), k=2)] == [3, 2]
    assert 4 not in [h.id for h in search_by_vector(conn, unit(1, 1), k=10)]
