"""Compatibility shim — implementation lives in app.legal.legal_decisions."""
from app.legal.legal_decisions import *  # noqa: F401,F403
import app.legal.legal_decisions as _impl
import sys as _sys

_sys.modules[__name__] = _impl