"""Thin compatibility shim over the MCP Python SDK.

Supports mcp>=2 (`MCPServer`, `Client`) and falls back to mcp 1.x (`FastMCP`,
`ClientSession` + `streamablehttp_client`) so the codebase survives SDK upgrades.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server

    MCP_V2 = True
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore

    MCP_V2 = False


def create_server(name: str, instructions: str) -> Any:
    return _Server(name=name, instructions=instructions)


def http_app(server: Any) -> Any:
    """Starlette ASGI app speaking MCP Streamable HTTP at /mcp."""
    return server.streamable_http_app()


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def parse_tool_result(result: Any) -> tuple[bool, Any]:
    """Return (is_error, payload) for a CallToolResult from either SDK major version."""
    is_error = bool(_attr(result, "is_error", "isError", default=False))
    structured = _attr(result, "structured_content", "structuredContent")
    if structured is not None and not is_error:
        payload = structured.get("result", structured) if isinstance(structured, dict) else structured
        return is_error, payload
    texts = [c.text for c in (_attr(result, "content", default=[]) or []) if getattr(c, "type", "") == "text"]
    text = "\n".join(texts)
    try:
        return is_error, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return is_error, text


def tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": (tool.description or "").strip(),
        "input_schema": _attr(tool, "input_schema", "inputSchema", default={}),
    }


@asynccontextmanager
async def connect(target: Any, headers: dict[str, str] | None = None, timeout_s: float = 15.0):
    """Yield an object with `list_tools()` and `call_tool(name, args)`.

    `target` is either an in-process server object (tests / local) or a URL.
    """
    if MCP_V2:
        from mcp import Client

        if isinstance(target, str) and headers:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client

            async with create_mcp_http_client(headers=headers) as http:
                async with Client(streamable_http_client(target, http_client=http), read_timeout_seconds=timeout_s) as c:  # type: ignore[arg-type]
                    yield c
            return
        async with Client(target, read_timeout_seconds=timeout_s) as c:
            yield c
        return
    # ---- mcp 1.x fallback
    from mcp import ClientSession  # type: ignore
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore
    from mcp.shared.memory import create_connected_server_and_client_session  # type: ignore

    if isinstance(target, str):
        async with streamablehttp_client(target, headers=headers or {}) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                yield session
    else:
        async with create_connected_server_and_client_session(target._mcp_server) as session:
            yield session
