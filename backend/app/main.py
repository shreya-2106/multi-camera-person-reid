from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.auth import router as auth_router
from backend.app.api.profile import router as profile_router
from backend.app.api.searches import router as searches_router
from backend.app.db.init_db import init_db

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


app = FastAPI(
    title="Multi-Camera Person Re-ID API",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.middleware("http")
async def log_search_request_duration(request, call_next):
    """Record upload-through-response latency without changing API payloads."""
    started = time.perf_counter()
    response = await call_next(request)
    if request.url.path == "/api/searches":
        logger.info(
            "search_request_timing method=%s status=%d total_seconds=%.3f",
            request.method,
            response.status_code,
            time.perf_counter() - started,
        )
    return response


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(searches_router)
app.include_router(profile_router)


# Serve the Phase 15 frontend after the API routes.
# API endpoints remain under /api/*.
app.mount(
    "/",
    StaticFiles(
        directory=DASHBOARD_DIR,
        html=True,
    ),
    name="dashboard",
)
