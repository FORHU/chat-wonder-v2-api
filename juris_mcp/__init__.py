"""Compatibility shim — implementation lives in mcp.legal.juris_mcp."""
from mcp.legal.juris_mcp import *  # noqa: F401,F403
import mcp.legal.juris_mcp as _impl
import sys as _sys
_sys.modules[__name__] = _impl