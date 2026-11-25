# -*- coding: utf-8 -*-
"""FastAPI composition root.

Owns the application object and mounts thin routers. Importing ``the_server``
registers the remaining legacy routes and startup hooks onto this same ``app``.
``run.py`` keeps ``from the_server import app`` via the shim in ``the_server``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health as health_router

app = FastAPI(title="Chat Wonder v2 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router.router)

# Side-effect import: registers chat/WS/legal/StyleOS routes and lifespan hooks.
import the_server as _legacy  # noqa: E402,F401

__all__ = ["app"]
