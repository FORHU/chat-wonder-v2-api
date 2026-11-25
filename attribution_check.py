"""Compatibility shim — implementation lives in app.legal.attribution_check."""
from app.legal.attribution_check import *  # noqa: F401,F403
import app.legal.attribution_check as _impl
import sys as _sys

_sys.modules[__name__] = _impl