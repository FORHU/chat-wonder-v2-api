"""Compatibility shim — implementation lives in app.legal.decisions_research."""
from app.legal.decisions_research import *  # noqa: F401,F403
import app.legal.decisions_research as _impl
import sys as _sys

_sys.modules[__name__] = _impl