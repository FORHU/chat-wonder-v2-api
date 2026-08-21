"""UK Legal MCP client for live UK legal, parliamentary, and HMRC retrieval."""

from .client import UkLegalMcpClient, UkLegalMcpError, get_client

__all__ = ["UkLegalMcpClient", "UkLegalMcpError", "get_client"]
