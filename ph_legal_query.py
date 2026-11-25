"""Compatibility shim — implementation lives in app.legal.ph_legal_query."""
from app.legal.ph_legal_query import *  # noqa: F401,F403
import app.legal.ph_legal_query as _impl
import sys as _sys

_sys.modules[__name__] = _impl