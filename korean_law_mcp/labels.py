"""Compatibility shim — implementation lives in mcp.legal.korean_law_mcp.labels."""
from mcp.legal.korean_law_mcp import labels as _impl
import sys as _sys
_sys.modules[__name__] = _impl