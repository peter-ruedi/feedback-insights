"""Core feedback models shared by ingest, search and summarize."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Comment:
    """A comment parsed from a CSV, before it is stored."""

    source_id: str
    body: str
    rating: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class Hit:
    """A stored comment returned by a similarity search."""

    id: int
    body: str
    rating: int | None
    distance: float  # cosine distance: 0 = same meaning, 2 = opposite


@dataclass(frozen=True)
class DatasetStats:
    total_comments: int
    embedded_comments: int
    by_rating: dict[str, int]  # star rating -> number of comments
    sources: list[str]


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
