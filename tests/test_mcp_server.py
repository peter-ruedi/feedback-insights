import pytest

pytest.importorskip("mcp")
from feedback_insights import mcp_server  # noqa: E402
from feedback_insights.models.feedback import Hit  # noqa: E402

HITS = [
    Hit(id=7, body="Too pricey for what you get.", rating=2, distance=0.1),
    Hit(id=9, body="Prices are fair for the area.", rating=4, distance=0.35),
]


@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    def fake(question, k):
        calls.append((question, k))
        return HITS[:k]

    monkeypatch.setattr(mcp_server, "_search", fake)
    return calls


def test_search_feedback_returns_ranked_comments_with_similarity(fake_search):
    result = mcp_server.search_feedback("What about prices?")
    assert result.question == "What about prices?"
    assert [c.id for c in result.comments] == [7, 9]
    assert result.comments[0].text == HITS[0].body
    assert result.comments[0].similarity == 0.9
    assert result.comments[1].similarity == 0.65
    assert fake_search == [("What about prices?", 20)]


def test_search_feedback_clamps_k(fake_search):
    mcp_server.search_feedback("q", k=500)
    mcp_server.search_feedback("q", k=0)
    assert [k for _, k in fake_search] == [mcp_server.MAX_K, 1]


def test_prompt_takes_no_arguments_and_names_the_tools():
    # Claude Code splits prompt arguments on spaces, so the prompt has none.
    import inspect

    assert inspect.signature(mcp_server.analyze_feedback).parameters == {}
    text = mcp_server.analyze_feedback()
    assert "search_feedback" in text and "feedback_stats" in text


def test_tool_docstrings_exist():
    # Claude reads these to decide when to call each tool.
    assert "relevant to a question" in mcp_server.search_feedback.__doc__
    assert mcp_server.feedback_stats.__doc__


def test_feedback_stats_converts_the_data_layer_numbers(monkeypatch):
    import contextlib

    from feedback_insights.models.feedback import DatasetStats

    numbers = DatasetStats(total_comments=4, embedded_comments=3,
                           by_rating={"2": 1, "5": 3}, sources=["yelp_review_full"])
    monkeypatch.setattr(mcp_server, "connect", lambda: contextlib.nullcontext("conn"))
    monkeypatch.setattr(mcp_server.comments_db, "stats", lambda conn: numbers)

    stats = mcp_server.feedback_stats()

    assert isinstance(stats, mcp_server.Stats)
    assert stats.model_dump() == {"total_comments": 4, "embedded_comments": 3,
                                  "by_rating": {"2": 1, "5": 3}, "sources": ["yelp_review_full"]}
