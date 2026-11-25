"""Minimal Streamable-HTTP JSON-RPC client for https://mcp.gomdori.app/law.

Stateless like juris.ph and the UK Legal MCP (tools/call works without a
session handshake) and, like the UK Legal MCP, needs no User-Agent spoofing
— confirmed live, no Cloudflare challenge in front of it, and no API key is
required for the hosted endpoint (shared quota). A free law.go.kr OC key
raises the quota; if KOREAN_LAW_MCP_API_KEY is set it is sent as the
`apikey` header per the server's own auth precedence.

Transport/retry/unwrap logic lives in mcp_client.McpClient (shared with
juris_mcp and uk_legal_mcp); this module only supplies Korean-Law-MCP-specific
defaults.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp_client import McpClient, McpError

DEFAULT_URL = "https://mcp.gomdori.app/law"

_client: Optional["KoreanLawMcpClient"] = None


class KoreanLawMcpError(McpError):
    """Raised when the Korean Law MCP server returns a JSON-RPC or protocol error."""


class KoreanLawMcpClient(McpClient):
    def __init__(self, url: Optional[str] = None, timeout: float = 60.0, api_key: Optional[str] = None):
        headers = {}
        key = api_key or os.getenv("KOREAN_LAW_MCP_API_KEY")
        if key:
            headers["apikey"] = key
        super().__init__(
            url=url or os.getenv("KOREAN_LAW_MCP_URL") or DEFAULT_URL,
            timeout=timeout,
            headers=headers or None,
        )

    def _call_tool_once(self, name: str, arguments: dict):
        try:
            return super()._call_tool_once(name, arguments)
        except McpError as exc:
            raise KoreanLawMcpError(str(exc)) from exc


def get_client() -> KoreanLawMcpClient:
    global _client
    if _client is None:
        _client = KoreanLawMcpClient()
    return _client
