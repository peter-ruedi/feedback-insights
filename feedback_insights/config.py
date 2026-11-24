import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time (e.g. in tests)
    pass

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://feedback:feedback@localhost:5432/feedback"
)
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384  # must match vector(384) in db/init.sql

# Optional. Without a key, /ask returns the most relevant comments and no summaries;
# Claude Desktop/Code can still write themes through the MCP server on your plan.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or None

# Claude model used to summarize themes when a key is set. Haiku is cheaper; Sonnet writes better summaries.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
