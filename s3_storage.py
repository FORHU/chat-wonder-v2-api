"""Compatibility shim — implementation lives in app.core.s3_storage."""
from app.core.s3_storage import *  # noqa: F401,F403
import app.core.s3_storage as _impl
import sys as _sys

_sys.modules[__name__] = _impl