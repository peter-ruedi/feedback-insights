"""Data-layer tests that need no database: fake cursors check what SQL runs."""

from feedback_insights.db import comments as comments_db


class OneShotCursor:
    """Fake cursor that records statements and returns one canned row."""

    def __init__(self, row):
        self.row, self.statements = row, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append(sql)

    def fetchone(self):
        return self.row


def test_stats_reads_everything_in_one_statement():
    # Separate statements could each see a different snapshot under Read
    # Committed, so totals and per-rating counts could disagree.
    cur = OneShotCursor((4, 3, {"5": 1, "2": 2, "4": 1}, ["yelp_review_full"]))
    conn = type("Conn", (), {"cursor": lambda self: cur})()

    stats = comments_db.stats(conn)

    assert len(cur.statements) == 1
    assert stats.total_comments == 4 and stats.embedded_comments == 3
    assert list(stats.by_rating) == ["2", "4", "5"]  # sorted for stable output
    assert stats.sources == ["yelp_review_full"]
