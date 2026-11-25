"""Compatibility shim — implementation lives in app.legal.exhibit_quotes."""
from app.legal.exhibit_quotes import *  # noqa: F401,F403
import app.legal.exhibit_quotes as _impl
import sys as _sys

_sys.modules[__name__] = _impl