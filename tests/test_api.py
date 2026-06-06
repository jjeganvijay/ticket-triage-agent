"""Unit tests for FastAPI endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def _mock_run(tmp_path: Path) -> MagicMock:
    """Build a mock AgentRun with one result."""
    from datetime import datetime
    from src.models.ticket import AgentRun, Category, Priority, TriageResult

    result = TriageResult(
        ticket_id="T001",
        description="Login crash.",
        category=Category.BUG,
        priority=Priority.P2_HIGH,
        reasoning="Software defect.",
        processed_at=datetime(2024, 1, 1, 12, 0, 0),
    )
    run = AgentRun(run_id="test-run-id", total_tickets=1, processed=1, errors=0)
    run.results = [result]
    return run


@patch("src.api.routes.TicketTriageAgent")
@patch("src.api.routes.save_results")
@patch("src.api.routes.export_to_csv")
def test_triage_endpoint(
    mock_export: MagicMock,
    mock_save: MagicMock,
    mock_agent_cls: MagicMock,
    tmp_path: Path,
) -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = _mock_run(tmp_path)
    mock_agent_cls.return_value = mock_agent

    resp = client.post(
        "/api/v1/triage",
        json={"tickets_folder": str(tmp_path)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["processed"] == 1
    assert body["run_id"] == "test-run-id"
    assert len(body["results"]) == 1


@patch("src.api.routes.get_all_results")
def test_list_results_endpoint(mock_get_all: MagicMock) -> None:
    from datetime import datetime
    from src.models.ticket import Category, Priority, TriageResult

    mock_get_all.return_value = [
        TriageResult(
            ticket_id="T001",
            description="test",
            category=Category.BUG,
            priority=Priority.P2_HIGH,
            reasoning="reason",
            processed_at=datetime(2024, 1, 1),
        )
    ]
    resp = client.get("/api/v1/results")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@patch("src.api.routes.get_result_by_id")
def test_get_result_not_found(mock_get: MagicMock) -> None:
    mock_get.return_value = None
    resp = client.get("/api/v1/results/NOTEXIST")
    assert resp.status_code == 404
