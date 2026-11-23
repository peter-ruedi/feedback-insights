from datetime import datetime, timezone

import pytest

from feedback_insights.ingest import parse_rows, read_csv


def test_cleans_whitespace_and_escaped_newlines():
    [c] = parse_rows([{"text": "Great   service.\\nWould come back again!"}])
    assert c.body == "Great service. Would come back again!"


def test_skips_empty_and_short_comments():
    rows = [{"text": ""}, {"text": "ok"}, {"text": "The pricing page was confusing to read."}]
    comments = list(parse_rows(rows))
    assert [c.source_id for c in comments] == ["3"]  # row number kept for idempotent reloads


def test_uses_given_id_rating_and_date():
    [c] = parse_rows([{
        "id": "abc", "text": "Support answered within an hour.",
        "rating": "4.0", "created_at": "2024-05-01T12:00:00Z",
    }])
    assert c.source_id == "abc"
    assert c.rating == 4
    assert c.created_at == datetime(2024, 5, 1, 12, tzinfo=timezone.utc)


def test_rejects_out_of_range_rating():
    with pytest.raises(ValueError):
        list(parse_rows([{"text": "This is long enough to keep.", "rating": "7"}]))


def test_read_csv_requires_text_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("comment\nhello there, this is a comment\n")
    with pytest.raises(ValueError, match="'text' column"):
        read_csv(path)


def test_read_csv_handles_quoted_commas(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text('id,text,rating\n1,"Fast, friendly, and cheap. Loved it.",5\n')
    [c] = read_csv(path)
    assert c.body == "Fast, friendly, and cheap. Loved it."
    assert c.rating == 5
