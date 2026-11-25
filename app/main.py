"""Application entry module.

During the domain-folder move this re-exports the FastAPI app from the
historical composition root. Callers keep using ``the_server`` / ``run:app``.
Later commits invert ownership so this module constructs the app.
"""

from the_server import app

__all__ = ["app"]
