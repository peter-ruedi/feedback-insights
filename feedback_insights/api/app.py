"""HTTP API.

    uvicorn feedback_insights.api:app --reload --port 8001
    curl -s localhost:8001/ask -H 'content-type: application/json' \
         -d '{"question": "What do people say about prices?"}'
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..summarize import Answer

log = logging.getLogger("feedback_insights")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---- request / response shapes -------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(20, ge=1, le=100, description="How many comments to retrieve")


class CommentOut(BaseModel):
    id: int
    body: str
    rating: int | None
    similarity: float = Field(description="1 = same meaning as the question, 0 = unrelated")


class ThemeOut(BaseModel):
    title: str
    summary: str
    comments: list[CommentOut]


class AskResponse(BaseModel):
    question: str
    comments: list[CommentOut] = Field(description="Retrieved comments, most relevant first")
    themes: list[ThemeOut] = Field(description="Claude's themes; empty when summaries are off")
    summarized: bool
    note: str | None = None
    usage: dict


# ---- dependencies (swapped for fakes in tests) ---------------------------

class Deps:
    """What the endpoint needs. Real versions are created at startup."""

    def __init__(self, search_fn, summarize_fn=None):
        self.search_fn = search_fn          # (question, k) -> list[Hit]
        self.summarize_fn = summarize_fn    # (question, hits) -> Answer, or None = summaries off


def real_deps() -> Deps:
    import psycopg
    from pgvector.psycopg import register_vector

    from ..embed import get_model
    from ..search import search

    get_model()  # load the embedding model now, not on the first request

    def search_fn(question, k):
        with psycopg.connect(config.DATABASE_URL) as conn:
            register_vector(conn)
            return search(conn, question, k)

    if not config.ANTHROPIC_API_KEY:
        log.info("No ANTHROPIC_API_KEY: /ask returns comments only (no Claude summaries)")
        return Deps(search_fn)

    import anthropic

    from ..summarize import summarize

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def summarize_fn(question, hits):
        return summarize(client, config.CLAUDE_MODEL, question, hits)

    log.info("Claude summaries on, model=%s", config.CLAUDE_MODEL)
    return Deps(search_fn, summarize_fn)


def to_out(hit) -> CommentOut:
    return CommentOut(id=hit.id, body=hit.body, rating=hit.rating,
                      similarity=round(1 - hit.distance, 3))


def create_app(deps_factory=real_deps) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = deps_factory()
        yield

    app = FastAPI(title="Feedback Insights", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        deps: Deps = app.state.deps
        t0 = time.perf_counter()
        hits = deps.search_fn(req.question, req.k)
        search_ms = round((time.perf_counter() - t0) * 1000)

        if deps.summarize_fn is None:
            answer = Answer(themes=[])
        else:
            try:
                answer = deps.summarize_fn(req.question, hits)
            except Exception as exc:  # e.g. bad API key, rate limit, network
                log.exception("summarize failed")
                raise HTTPException(status_code=502, detail=f"LLM call failed: {type(exc).__name__}")

        usage = {**answer.usage, "search_ms": search_ms, "retrieved": len(hits),
                 "total_ms": round((time.perf_counter() - t0) * 1000)}
        log.info("ask question=%r %s", req.question, " ".join(f"{k}={v}" for k, v in usage.items()))

        return AskResponse(
            question=req.question,
            comments=[to_out(h) for h in hits],
            summarized=deps.summarize_fn is not None,
            note=answer.note,
            usage=usage,
            themes=[
                ThemeOut(title=t.title, summary=t.summary, comments=[to_out(c) for c in t.comments])
                for t in answer.themes
            ],
        )

    return app


app = create_app()
