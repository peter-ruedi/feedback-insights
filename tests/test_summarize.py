from types import SimpleNamespace

from feedback_insights.search import Hit
from feedback_insights.summarize import build_user_message, parse_tool_input, summarize

HITS = [
    Hit(id=1, body="Way too expensive for the portion size.", rating=2, distance=0.2),
    Hit(id=2, body="Prices went up again, not worth it.", rating=1, distance=0.25),
    Hit(id=3, body="Friendly staff & quick service <3", rating=5, distance=0.6),
]


def test_user_message_lists_ids_ratings_and_escapes_text():
    msg = build_user_message("What about prices?", HITS)
    assert msg.startswith("Question: What about prices?")
    assert '<comment id="1" rating="2">Way too expensive' in msg
    assert "&amp; quick service &lt;3" in msg  # comment text can't break the tags


def test_parse_keeps_valid_ids_and_drops_invented_ones():
    themes, note = parse_tool_input(
        {"themes": [{"title": "Too expensive", "summary": "Prices feel high.", "comment_ids": [1, 2, 99, 2]}]},
        HITS,
    )
    assert [c.id for c in themes[0].comments] == [1, 2]  # 99 was never retrieved; dup removed
    assert note is None


def test_parse_drops_themes_with_no_valid_comments():
    themes, _ = parse_tool_input(
        {"themes": [{"title": "Made up", "summary": "", "comment_ids": [42]},
                    {"title": "Service", "summary": "Staff are nice.", "comment_ids": ["3"]}]},
        HITS,
    )
    assert [t.title for t in themes] == ["Service"]


def test_parse_handles_empty_answer_with_note():
    themes, note = parse_tool_input({"themes": [], "note": "Nothing about parking."}, HITS)
    assert themes == [] and note == "Nothing about parking."


class FakeClient:
    """Stands in for anthropic.Anthropic(); records the request, returns a canned tool call."""

    def __init__(self, tool_input):
        self.request = None
        self.messages = SimpleNamespace(create=self._create)
        self._tool_input = tool_input

    def _create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input=self._tool_input)],
            usage=SimpleNamespace(input_tokens=812, output_tokens=164),
        )


def test_summarize_forces_the_tool_and_reports_usage():
    client = FakeClient({"themes": [{"title": "Too expensive", "summary": "s", "comment_ids": [1]}]})
    answer = summarize(client, "test-model", "What about prices?", HITS)

    assert client.request["tool_choice"] == {"type": "tool", "name": "report_themes"}
    assert client.request["model"] == "test-model"
    assert answer.themes[0].comments[0].id == 1
    assert answer.usage["input_tokens"] == 812 and answer.usage["output_tokens"] == 164
    assert "llm_ms" in answer.usage


def test_summarize_skips_the_api_when_nothing_was_retrieved():
    client = FakeClient({})
    answer = summarize(client, "test-model", "anything", [])
    assert client.request is None
    assert answer.themes == []
