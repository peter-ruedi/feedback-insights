"""Structured output of the MCP server's tools."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Comment(BaseModel):
    id: int
    text: str
    rating: int | None = Field(description="Star rating 1-5, if the dataset has one")
    similarity: float = Field(description="1 = same meaning as the question, 0 = unrelated")


class SearchResult(BaseModel):
    question: str
    comments: list[Comment] = Field(description="Most relevant first")


class Stats(BaseModel):
    total_comments: int
    embedded_comments: int
    by_rating: dict[str, int] = Field(description="Number of comments per star rating")
    sources: list[str]
