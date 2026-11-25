"""Compatibility shim — implementation lives in app.core.confidence."""
from app.core.confidence import *  # noqa: F401,F403
import app.core.confidence as _impl
import sys as _sys

_sys.modules[__name__] = _impl