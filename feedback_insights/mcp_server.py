"""MCP server: lets Claude Desktop or Claude Code search the feedback.

Claude calls `search_feedback`, reads the comments it gets back, and writes
the themes itself, so no Anthropic API key is needed.

    python -m feedback_insights.mcp_server      # stdio; normally started by the MCP client

Note: with the stdio transport, stdout carries the protocol. Never print()
in this module; log to stderr instead.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict

from mcp.server import MCPServer

from . import config
from .db import comments as comments_db, connect
from .models.mcp import Comment, SearchResult, Stats

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("feedback_insights.mcp")

MAX_K = 50

mcp = MCPServer("feedback-insights")


# ---- data access (replaced with fakes in tests) -------------------------

def _search(question: str, k: int):
    from .search import search

    with connect() as conn:
        return search(conn, question, k)


def _stats() -> Stats:
    with connect() as conn:
        return Stats(**asdict(comments_db.stats(conn)))


# ---- tools ---------------------------------------------------------------

@mcp.tool()
def search_feedback(question: str, k: int = 20) -> SearchResult:
    """Find the customer comments most relevant to a question, by meaning (not keywords).

    Use this for any question about what customers say, e.g. "What do people say
    about prices?". Returns up to k comments, most relevant first, with their ids
    and star ratings. Cite comment ids when you summarize. Call it again with a
    rephrased question if the results look off-topic.
    """
    k = max(1, min(k, MAX_K))
    hits = _search(question, k)
    log.info("search_feedback question=%r k=%d returned=%d", question, k, len(hits))
    return SearchResult(
        question=question,
        comments=[Comment(id=h.id, text=h.body, rating=h.rating,
                          similarity=round(1 - h.distance, 3)) for h in hits],
    )

@mcp.tool()
def feedback_stats() -> Stats:
    """Totals for the feedback dataset: comment count and comments per star rating.

    Use it to put an answer in context (e.g. "out of 3,000 reviews") or to check
    how ratings are distributed.
    """
    return _stats()

@mcp.prompt()
def analyze_feedback() -> str:
    """Analyze customer feedback: themes with cited comments. Type your question after it."""
    # No arguments on purpose: Claude Code splits slash-command arguments on
    # spaces (even inside quotes), so a `question` argument only ever got the
    # first word. Claude sees the full text the user typed, so it answers that.
    return """Analyze the customer feedback to answer the question I typed with this \
command (or, if there is none, ask me what I want to know).

1. Call search_feedback with the question (k=20). If many results are off-topic,
   search again with a rephrased question.
2. Group the relevant comments into themes (2-5 unless I ask for a number). For
   each theme give a short title, one or two sentences on what people say, the
   comment ids behind it, and one short quote.
3. Use only what the comments say. Do not invent opinions or numbers. Skip
   comments that are not about the question.
4. End with one line on how many of the retrieved comments were relevant. If
   useful, call feedback_stats for the size of the dataset."""


def main() -> None:
    log.info("feedback-insights MCP server starting (database: %s)",
             config.DATABASE_URL.rsplit("@", 1)[-1])  # host/db only, no password
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
