"""Starts the real MCP server with a fake search, for tests/test_mcp_protocol.py."""

from feedback_insights import mcp_server
from feedback_insights.models.feedback import Hit

mcp_server._search = lambda question, k: [
    Hit(id=1, body="Way too expensive.", rating=2, distance=0.2),
    Hit(id=2, body="Fair prices.", rating=4, distance=0.4),
][:k]

if __name__ == "__main__":
    mcp_server.main()
