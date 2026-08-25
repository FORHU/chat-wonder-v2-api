"""Minimal Streamable-HTTP JSON-RPC client for https://uk-legal-mcp.fly.dev/mcp.

Stateless like juris.ph (tools/call works without a session handshake) and,
unlike juris.ph, needs no User-Agent spoofing — confirmed live, no Cloudflare
challenge in front of it. See ADR-0003.

Transport/retry/unwrap logic lives in mcp_client.McpClient (shared with
juris_mcp); this module only supplies UK-Legal-MCP-specific defaults.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp_client import McpClient, McpError

DEFAULT_URL = "https://uk-legal-mcp.fly.dev/mcp"

_client: Optional["UkLegalMcpClient"] = None


class UkLegalMcpError(McpError):
    """Raised when the UK Legal MCP server returns a JSON-RPC or protocol error."""


class UkLegalMcpClient(McpClient):
    def __init__(self, url: Optional[str] = None, timeout: float = 60.0):
        super().__init__(url=url or os.getenv("UK_LEGAL_MCP_URL") or DEFAULT_URL, timeout=timeout)

    def _call_tool_once(self, name: str, arguments: dict):
        try:
            return super()._call_tool_once(name, arguments)
        except McpError as exc:
            raise UkLegalMcpError(str(exc)) from exc


def get_client() -> UkLegalMcpClient:
    global _client
    if _client is None:
        _client = UkLegalMcpClient()
    return _client
