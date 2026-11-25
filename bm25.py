"""Compatibility shim — implementation lives in app.core.bm25."""
from app.core.bm25 import *  # noqa: F401,F403
import app.core.bm25 as _impl
import sys as _sys

_sys.modules[__name__] = _impl