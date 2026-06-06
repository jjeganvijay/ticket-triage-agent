"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Ticket Triage Agent API",
    description="AI-powered support ticket classifier using Google Gemini.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "ticket-triage-agent"}
