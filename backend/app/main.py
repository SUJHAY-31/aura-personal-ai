"""
AURA backend ASGI application.

This module constructs the FastAPI instance and mounts HTTP routers.
Run with: uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI

from .api.chat import router as chat_router
from .api.health import router as health_router
from .utils.metadata import APP_DESCRIPTION, APP_TITLE, APP_VERSION

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health_router)
app.include_router(chat_router)
