"""Compatibility shim — implementation lives in app.core.model_router."""
from app.core.model_router import *  # noqa: F401,F403
import app.core.model_router as _impl
import sys as _sys

_sys.modules[__name__] = _impl