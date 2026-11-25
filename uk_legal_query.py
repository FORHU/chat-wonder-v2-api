"""Compatibility shim — implementation lives in app.legal.uk_legal_query."""
from app.legal.uk_legal_query import *  # noqa: F401,F403
import app.legal.uk_legal_query as _impl
import sys as _sys

_sys.modules[__name__] = _impl