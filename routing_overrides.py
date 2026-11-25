"""Compatibility shim — implementation lives in app.core.routing_overrides."""
from app.core.routing_overrides import *  # noqa: F401,F403
import app.core.routing_overrides as _impl
import sys as _sys

_sys.modules[__name__] = _impl