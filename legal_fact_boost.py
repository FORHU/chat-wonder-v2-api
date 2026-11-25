"""Compatibility shim — implementation lives in app.legal.legal_fact_boost."""
from app.legal.legal_fact_boost import *  # noqa: F401,F403
import app.legal.legal_fact_boost as _impl
import sys as _sys

_sys.modules[__name__] = _impl