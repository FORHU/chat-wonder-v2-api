"""Shared minimal Streamable-HTTP JSON-RPC client for MCP `tools/call`.

Extracted from juris_mcp/client.py per ADR-0003 so juris_mcp and uk_legal_mcp
share one transport/retry/unwrap implementation instead of two copies that
could drift apart. Server-specific defaults (URL, extra headers) live in each
server's own client module.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class McpError(Exception):
    """Raised when an MCP server returns a JSON-RPC or protocol error."""


class McpClient:
    def __init__(
        self,
        url: str,
        timeout: float = 60.0,
        headers: Optional[dict] = None,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._rpc_id = 0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **(headers or {}),
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
                        "MCP transport error on %s at %s (retrying once): %s",
                        name,
                        self.url,
                        exc,
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
            raise McpError(f"Non-JSON MCP response: {response.text[:200]}") from exc

        if "error" in data and data["error"]:
            err = data["error"]
            raise McpError(err.get("message") if isinstance(err, dict) else str(err))

        result = data.get("result") or {}
        return unwrap_tool_result(result)


def unwrap_tool_result(result: Any) -> Any:
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
