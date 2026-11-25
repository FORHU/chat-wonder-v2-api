"""Compatibility shim — implementation lives in mcp.legal.uk_legal_mcp.client."""
from mcp.legal.uk_legal_mcp import client as _impl
import sys as _sys
_sys.modules[__name__] = _impl