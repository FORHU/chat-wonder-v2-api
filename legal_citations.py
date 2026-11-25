"""Compatibility shim — implementation lives in app.legal.legal_citations."""
from app.legal.legal_citations import *  # noqa: F401,F403
import app.legal.legal_citations as _impl
import sys as _sys

_sys.modules[__name__] = _impl