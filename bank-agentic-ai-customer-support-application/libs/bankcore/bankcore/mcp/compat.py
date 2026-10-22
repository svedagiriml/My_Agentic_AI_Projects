"""Thin compatibility layer over the official MCP Python SDK.

Targets SDK 2.x (``MCPServer`` / ``Client``) and falls back to 1.x (``FastMCP`` /
``ClientSession``). Keeping SDK-specific code in this one module makes upgrades cheap.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

try:  # ---- MCP SDK 2.x -------------------------------------------------------
    from mcp.client.client import Client as _Client
    from mcp.client.streamable_http import streamable_http_client as _http_transport
    from mcp.server.mcpserver import MCPServer as _Server

    SDK_MAJOR = 2
except ImportError:  # ---- MCP SDK 1.x ----------------------------------------
    from mcp import ClientSession as _ClientSession  # type: ignore
    from mcp.client.streamable_http import streamablehttp_client as _http_transport  # type: ignore
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore

    _Client = None
    SDK_MAJOR = 1


def new_server(name: str, instructions: str) -> Any:
    return _Server(name=name, instructions=instructions)


def http_app(server: Any) -> Any:
    """Stateless, JSON-response Streamable HTTP app (horizontally scalable on Cloud Run)."""
    if SDK_MAJOR == 2:
        return server.streamable_http_app(stateless_http=True, json_response=True, streamable_http_path="/")
    server.settings.stateless_http = True  # pragma: no cover
    server.settings.json_response = True
    server.settings.streamable_http_path = "/"
    return server.streamable_http_app()


def session_manager(server: Any):
    return server.session_manager


class _Normalised:
    """Uniform view of a list_tools / call_tool result across SDK versions."""

    @staticmethod
    def tools(result: Any) -> list[dict]:
        out = []
        for t in result.tools:
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
            out.append({"name": t.name, "description": t.description or "", "input_schema": schema})
        return out

    @staticmethod
    def call(result: Any) -> tuple[bool, dict | None, str]:
        is_error = bool(getattr(result, "is_error", None) or getattr(result, "isError", False))
        structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
        text = "".join(getattr(c, "text", "") for c in (result.content or []))
        return is_error, structured, text


@asynccontextmanager
async def open_client(target: Any, headers: dict[str, str] | None = None, timeout_s: float = 10.0):
    """Yield an object with ``list_tools()`` and ``call_tool(name, args)`` returning normalised data.

    ``target`` is either an MCP URL or an in-process server object (tests / sidecar mode).
    """
    if SDK_MAJOR == 2:
        if isinstance(target, str):
            import httpx2
            transport = _http_transport(target, http_client=httpx2.AsyncClient(headers=headers or {},
                                                                                timeout=timeout_s))
            client = _Client(transport, read_timeout_seconds=timeout_s)
        else:
            client = _Client(target, read_timeout_seconds=timeout_s)
        async with client as c:
            yield _Adapter(c)
        return
    async with _http_transport(target, headers=headers or {}, timeout=timeout_s) as (r, w, _):  # pragma: no cover
        async with _ClientSession(r, w) as session:
            await session.initialize()
            yield _Adapter(session)


class _Adapter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def list_tools(self) -> list[dict]:
        return _Normalised.tools(await self._inner.list_tools())

    async def call_tool(self, name: str, arguments: dict) -> tuple[bool, dict | None, str]:
        return _Normalised.call(await self._inner.call_tool(name, arguments))
