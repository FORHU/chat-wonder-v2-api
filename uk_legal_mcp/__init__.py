"""Compatibility shim — implementation lives in mcp.legal.uk_legal_mcp."""
from mcp.legal.uk_legal_mcp import *  # noqa: F401,F403
import mcp.legal.uk_legal_mcp as _impl
import sys as _sys
_sys.modules[__name__] = _impl