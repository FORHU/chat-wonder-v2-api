"""Compatibility shim — implementation lives in mcp.legal.uk_legal_mcp.direct."""
from mcp.legal.uk_legal_mcp import direct as _impl
import sys as _sys
_sys.modules[__name__] = _impl