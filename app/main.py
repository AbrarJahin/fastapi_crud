from datetime import datetime, timezone
from fastapi import FastAPI

import logging

from app.api.items import router as items_router
from app.api.agent import router as agent_router

_logger = logging.getLogger(__name__)

app = FastAPI(
    title="FastAPI SQLite CRUD",
    version="1.0.0",
    description="FastAPI + SQLAlchemy + SQLite + Alembic (Conda-managed)",
)

# Register routers
app.include_router(items_router)
app.include_router(agent_router)

@app.on_event("startup")
async def startup() -> None:
    _logger.info("✅ FastAPI startup complete")
    _logger.info("Routes registered: %d", len(app.routes))

@app.on_event("shutdown")
async def shutdown() -> None:
    _logger.info("🛑 FastAPI shutdown")

@app.get("/health")
def health():
    _logger.info("Health check called")
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat()
    }
