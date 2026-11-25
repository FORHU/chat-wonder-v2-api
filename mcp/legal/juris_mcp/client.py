"""Minimal Streamable-HTTP JSON-RPC client for https://juris.ph/mcp.

juris.ph is stateless (tools/call works without a session handshake). Cloudflare
blocks generic Python user-agents, so we send an identifying browser-compatible UA.
Transport failures are retried once per ADR-0002.

Transport/retry/unwrap logic lives in mcp_client.McpClient (shared with
uk_legal_mcp per ADR-0003); this module only supplies juris.ph-specific defaults.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp_client import McpClient, McpError

DEFAULT_URL = "https://juris.ph/mcp"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; chat-wonder-v2-api/1.0; +https://juris.ph/mcp)"
)

_client: Optional["JurisMcpClient"] = None


class JurisMcpError(McpError):
    """Raised when the juris.ph MCP server returns a JSON-RPC or protocol error."""


class JurisMcpClient(McpClient):
    def __init__(
        self,
        url: Optional[str] = None,
        timeout: float = 60.0,
        user_agent: Optional[str] = None,
    ):
        super().__init__(
            url=url or os.getenv("JURIS_MCP_URL") or DEFAULT_URL,
            timeout=timeout,
            headers={
                "User-Agent": user_agent
                or os.getenv("JURIS_MCP_USER_AGENT")
                or DEFAULT_USER_AGENT,
            },
        )

    def _call_tool_once(self, name: str, arguments: dict):
        try:
            return super()._call_tool_once(name, arguments)
        except McpError as exc:
            raise JurisMcpError(str(exc)) from exc


def get_client() -> JurisMcpClient:
    global _client
    if _client is None:
        _client = JurisMcpClient()
    return _client
