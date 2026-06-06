"""FastAPI route handlers for the Ticket Triage Agent."""
from __future__ import annotations

import os
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.agent.react_agent import TicketTriageAgent
from src.database.db import get_all_results, get_result_by_id, save_results
from src.export.csv_exporter import export_to_csv
from src.models.ticket import TriageResult

router = APIRouter()

TICKETS_FOLDER = os.getenv("TICKETS_FOLDER", "data/sample_tickets")
DB_PATH = os.getenv("DB_PATH", "database/tickets.db")
CSV_PATH = os.getenv("CSV_PATH", "output/results.csv")


# ── Request / Response schemas ────────────────────────────────────────────────

class TriageRequest(BaseModel):
    tickets_folder: str = TICKETS_FOLDER


class TriageResponse(BaseModel):
    run_id: str
    total_tickets: int
    processed: int
    errors: int
    results: List[TriageResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/triage", response_model=TriageResponse, tags=["Triage"])
def run_triage(request: TriageRequest, background_tasks: BackgroundTasks) -> TriageResponse:
    """
    Run the full ReAct triage loop on the given tickets folder.
    Saves results to SQLite and schedules a CSV export in the background.
    """
    agent = TicketTriageAgent()
    run = agent.run(request.tickets_folder, verbose=False)

    if run.results:
        save_results(run.results, db_path=DB_PATH)
        background_tasks.add_task(export_to_csv, run.results, CSV_PATH)

    return TriageResponse(
        run_id=run.run_id,
        total_tickets=run.total_tickets,
        processed=run.processed,
        errors=run.errors,
        results=run.results,
    )


@router.get("/results", response_model=List[TriageResult], tags=["Results"])
def list_results() -> List[TriageResult]:
    """Return all triage results stored in the database."""
    return get_all_results(db_path=DB_PATH)


@router.get("/results/{ticket_id}", response_model=TriageResult, tags=["Results"])
def get_result(ticket_id: str) -> TriageResult:
    """Return the triage result for a specific ticket ID."""
    result = get_result_by_id(ticket_id, db_path=DB_PATH)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return result


@router.get("/results/{ticket_id}/status", tags=["Results"])
def get_status(ticket_id: str) -> dict:
    """Return just the category and priority for a ticket (lightweight)."""
    result = get_result_by_id(ticket_id, db_path=DB_PATH)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return {
        "ticket_id": result.ticket_id,
        "category": result.category,
        "priority": result.priority,
    }
