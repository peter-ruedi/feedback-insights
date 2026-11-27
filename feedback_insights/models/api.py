"""Request and response shapes for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
