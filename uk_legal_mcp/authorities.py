"""Compatibility shim — implementation lives in mcp.legal.uk_legal_mcp.authorities."""
from mcp.legal.uk_legal_mcp import authorities as _impl
import sys as _sys
_sys.modules[__name__] = _impl