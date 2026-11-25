"""Compatibility shim — implementation lives in mcp.legal.uk_legal_mcp.scoring."""
from mcp.legal.uk_legal_mcp import scoring as _impl
import sys as _sys
_sys.modules[__name__] = _impl