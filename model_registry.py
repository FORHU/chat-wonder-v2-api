"""Compatibility shim — implementation lives in app.core.model_registry."""
from app.core.model_registry import *  # noqa: F401,F403
import app.core.model_registry as _impl
import sys as _sys

_sys.modules[__name__] = _impl