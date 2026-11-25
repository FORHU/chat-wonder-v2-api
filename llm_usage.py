"""Compatibility shim — implementation lives in app.core.llm_usage."""
from app.core.llm_usage import *  # noqa: F401,F403
import app.core.llm_usage as _impl
import sys as _sys

_sys.modules[__name__] = _impl