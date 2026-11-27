"""End-to-end: start the server over stdio and talk to it like Claude Desktop does."""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# MCP 2.x defaults ClientSession.read_timeout_seconds to None, so a server that
# starts but never answers would hang the test run. One deadline covers the
# whole session: start-up, initialize() and the test's own calls.
SESSION_TIMEOUT_S = 30


async def _session_run(fn, server_script, timeout):
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "tests" / server_script)],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    async with asyncio.timeout(timeout):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)


def run(fn, server_script="fake_mcp_server.py", timeout=SESSION_TIMEOUT_S):
    return asyncio.run(_session_run(fn, server_script, timeout))


def test_lists_tools_and_prompt():
    async def check(s):
        tools = {t.name for t in (await s.list_tools()).tools}
        prompts = {p.name for p in (await s.list_prompts()).prompts}
        return tools, prompts

    tools, prompts = run(check)
    assert tools == {"search_feedback", "feedback_stats"}
    assert prompts == {"analyze_feedback"}


def test_search_feedback_over_the_protocol():
    async def call(s):
        return await s.call_tool("search_feedback", {"question": "prices?", "k": 1})

    result = run(call)
    assert not result.is_error
    assert result.structured_content["comments"] == [
        {"id": 1, "text": "Way too expensive.", "rating": 2, "similarity": 0.8}
    ]


def test_prompt_has_no_arguments():
    async def get(s):
        listed = (await s.list_prompts()).prompts
        msg = (await s.get_prompt("analyze_feedback", {})).messages[0]
        return listed, msg

    listed, msg = run(get)
    assert not listed[0].arguments
    assert "search_feedback" in msg.content.text


def test_silent_server_fails_fast_instead_of_hanging():
    async def never_reached(s):
        raise AssertionError("initialize() should not have succeeded")

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        run(never_reached, server_script="silent_mcp_server.py", timeout=2)
    assert time.monotonic() - start < 15  # bounded, including cleanup of the child
