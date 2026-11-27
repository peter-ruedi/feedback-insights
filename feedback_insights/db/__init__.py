"""Data layer: the only place that talks to PostgreSQL.

- connection: open a connection with pgvector types registered
- comments: every query on the `comments` table

Everything else calls these functions and never writes SQL itself.
"""

from .connection import connect

__all__ = ["connect"]
