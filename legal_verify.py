"""Compatibility shim — implementation lives in app.legal.legal_verify."""
from app.legal.legal_verify import *  # noqa: F401,F403
import app.legal.legal_verify as _impl
import sys as _sys

_sys.modules[__name__] = _impl