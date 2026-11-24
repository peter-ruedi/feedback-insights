import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from feedback_insights.api import Deps, create_app  # noqa: E402
from feedback_insights.search import Hit  # noqa: E402
from feedback_insights.summarize import Answer, Theme  # noqa: E402

HITS = [
    Hit(id=7, body="Too pricey for what you get.", rating=2, distance=0.1),
    Hit(id=9, body="Prices are fair for the area.", rating=4, distance=0.3),
]


def search_fn(question, k):
    return HITS[:k]


def make_client(summarize_fn=None):
    return TestClient(create_app(lambda: Deps(search_fn, summarize_fn)))


def test_without_a_key_returns_ranked_comments_and_no_themes():
    with make_client() as client:
        r = client.post("/ask", json={"question": "What about prices?"})
    assert r.status_code == 200
    body = r.json()
    assert body["summarized"] is False
    assert body["themes"] == []
    assert [c["id"] for c in body["comments"]] == [7, 9]
    assert body["comments"][0] == {"id": 7, "body": HITS[0].body, "rating": 2, "similarity": 0.9}
    assert body["usage"]["retrieved"] == 2
    assert {"search_ms", "total_ms"} <= body["usage"].keys()


def test_with_a_summarizer_adds_themes():
    def summarize(question, hits):
        return Answer(themes=[Theme("Too expensive", "People find it pricey.", hits[:1])],
                      usage={"input_tokens": 100, "output_tokens": 20})

    with make_client(summarize) as client:
        body = client.post("/ask", json={"question": "What about prices?"}).json()
    assert body["summarized"] is True
    assert body["themes"][0]["title"] == "Too expensive"
    assert [c["id"] for c in body["themes"][0]["comments"]] == [7]
    assert body["usage"]["input_tokens"] == 100


def test_k_limits_the_comments():
    with make_client() as client:
        body = client.post("/ask", json={"question": "What about prices?", "k": 1}).json()
    assert len(body["comments"]) == 1


def test_rejects_empty_question():
    with make_client() as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422


def test_returns_502_when_the_llm_call_fails():
    def broken(question, hits):
        raise RuntimeError("bad key")

    with make_client(broken) as client:
        assert client.post("/ask", json={"question": "What about prices?"}).status_code == 502


def test_health():
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
