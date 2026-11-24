"""Ask Claude to group retrieved comments into themes, citing comment IDs.

Claude answers through a tool call with a fixed JSON schema, so the reply is
structured data rather than free text we have to parse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from html import escape

from .search import Hit

SYSTEM_PROMPT = """You analyze open-ended customer feedback.
You get a question and a set of comments, each with an id.
Group the comments that are relevant to the question into 2-5 themes.
Rules:
- Use only the comments provided. Do not invent opinions or numbers.
- Every theme must cite the ids of the comments that support it.
- Skip comments that are not relevant to the question.
- If no comments are relevant, return an empty list of themes and explain why in `note`.
Answer by calling the report_themes tool."""

THEMES_TOOL = {
    "name": "report_themes",
    "description": "Report the themes found in the comments, with the comment ids behind each.",
    "input_schema": {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "A few words, e.g. 'Prices feel too high'"},
                        "summary": {"type": "string", "description": "One or two sentences on what people say"},
                        "comment_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["title", "summary", "comment_ids"],
                },
            },
            "note": {"type": "string", "description": "Optional caveat, e.g. few relevant comments"},
        },
        "required": ["themes"],
    },
}


@dataclass
class Theme:
    title: str
    summary: str
    comments: list[Hit]


@dataclass
class Answer:
    themes: list[Theme]
    note: str | None = None
    usage: dict = field(default_factory=dict)


def build_user_message(question: str, hits: list[Hit]) -> str:
    lines = [f"Question: {question}", "", "Comments:"]
    for h in hits:
        rating = f' rating="{h.rating}"' if h.rating is not None else ""
        lines.append(f'<comment id="{h.id}"{rating}>{escape(h.body)}</comment>')
    return "\n".join(lines)


def parse_tool_input(data: dict, hits: list[Hit]) -> tuple[list[Theme], str | None]:
    """Turn Claude's tool input into Themes.

    Drops comment ids Claude didn't receive (so it can't cite made-up comments),
    removes duplicates, and drops themes left with no valid comments.
    """
    by_id = {h.id: h for h in hits}
    themes: list[Theme] = []
    for t in data.get("themes") or []:
        ids: list[int] = []
        for raw in t.get("comment_ids") or []:
            try:
                cid = int(raw)
            except (TypeError, ValueError):
                continue
            if cid in by_id and cid not in ids:
                ids.append(cid)
        title = (t.get("title") or "").strip()
        if ids and title:
            themes.append(Theme(title=title, summary=(t.get("summary") or "").strip(),
                                comments=[by_id[i] for i in ids]))
    note = (data.get("note") or "").strip() or None
    return themes, note


def summarize(client, model: str, question: str, hits: list[Hit], max_tokens: int = 2000) -> Answer:
    """Call Claude. `client` is an anthropic.Anthropic() instance."""
    if not hits:
        return Answer(themes=[], note="No comments found to analyze.")

    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        tools=[THEMES_TOOL],
        tool_choice={"type": "tool", "name": THEMES_TOOL["name"]},
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    )
    latency_ms = round((time.perf_counter() - start) * 1000)

    tool_input = next((b.input for b in response.content if b.type == "tool_use"), {})
    themes, note = parse_tool_input(tool_input, hits)
    usage = {
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "llm_ms": latency_ms,
    }
    return Answer(themes=themes, note=note, usage=usage)
