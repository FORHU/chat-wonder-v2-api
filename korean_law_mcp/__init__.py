"""Compatibility shim — implementation lives in mcp.legal.korean_law_mcp."""
from mcp.legal.korean_law_mcp import *  # noqa: F401,F403
import mcp.legal.korean_law_mcp as _impl
import sys as _sys
_sys.modules[__name__] = _impl