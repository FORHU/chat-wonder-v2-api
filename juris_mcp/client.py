"""Minimal Streamable-HTTP JSON-RPC client for https://juris.ph/mcp.

juris.ph is stateless (tools/call works without a session handshake). Cloudflare
blocks generic Python user-agents, so we send an identifying browser-compatible UA.
Transport failures are retried once per ADR-0002.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests

DEFAULT_URL = "https://juris.ph/mcp"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; chat-wonder-v2-api/1.0; +https://juris.ph/mcp)"
)

logger = logging.getLogger(__name__)

_client: Optional["JurisMcpClient"] = None


class JurisMcpError(Exception):
    """Raised when the MCP server returns a JSON-RPC or protocol error."""


class JurisMcpClient:
    def __init__(
        self,
        url: Optional[str] = None,
        timeout: float = 60.0,
        user_agent: Optional[str] = None,
    ):
        self.url = (url or os.getenv("JURIS_MCP_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self._rpc_id = 0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "User-Agent": user_agent
                or os.getenv("JURIS_MCP_USER_AGENT")
                or DEFAULT_USER_AGENT,
            }
        )

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        """Call an MCP tool; retry once on transport / HTTP errors only."""
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                return self._call_tool_once(name, arguments or {})
            except requests.RequestException as exc:
                last_err = exc
                if attempt == 0:
                    logger.warning(
                        "Juris MCP transport error on %s (retrying once): %s", name, exc
                    )
                    time.sleep(0.4)
                    continue
                raise
        assert last_err is not None
        raise last_err

    def _call_tool_once(self, name: str, arguments: dict) -> Any:
        self._rpc_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = self._session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise JurisMcpError(f"Non-JSON MCP response: {response.text[:200]}") from exc

        if "error" in data and data["error"]:
            err = data["error"]
            raise JurisMcpError(
                err.get("message") if isinstance(err, dict) else str(err)
            )

        result = data.get("result") or {}
        return _unwrap_tool_result(result)


def _unwrap_tool_result(result: Any) -> Any:
    """Parse MCP tool result content blocks into a Python value."""
    if not isinstance(result, dict):
        return result

    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result

    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text") or "")

    if not texts:
        return result

    raw = "\n".join(texts).strip()
    if not raw:
        return result

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw, "raw_result": result}


def get_client() -> JurisMcpClient:
    global _client
    if _client is None:
        _client = JurisMcpClient()
    return _client
