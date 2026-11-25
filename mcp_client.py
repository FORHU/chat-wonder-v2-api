"""Compatibility shim — implementation lives in app.core.mcp_client."""
from app.core.mcp_client import *  # noqa: F401,F403
import app.core.mcp_client as _impl
import sys as _sys

_sys.modules[__name__] = _impl