"""Unit tests for the ticket loader."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.loader.ticket_loader import load_tickets
from src.models.ticket import Ticket


@pytest.fixture()
def ticket_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample ticket JSON files."""
    tickets = [
        {"ticket_id": "T001", "description": "I was charged twice."},
        {"ticket_id": "T002", "description": "App crashes on login."},
    ]
    for t in tickets:
        (tmp_path / f"{t['ticket_id']}.json").write_text(json.dumps(t), encoding="utf-8")
    return tmp_path


def test_load_tickets_returns_correct_count(ticket_dir: Path) -> None:
    tickets = load_tickets(ticket_dir)
    assert len(tickets) == 2


def test_load_tickets_returns_ticket_objects(ticket_dir: Path) -> None:
    tickets = load_tickets(ticket_dir)
    assert all(isinstance(t, Ticket) for t in tickets)


def test_load_tickets_correct_ids(ticket_dir: Path) -> None:
    tickets = load_tickets(ticket_dir)
    ids = {t.ticket_id for t in tickets}
    assert ids == {"T001", "T002"}


def test_load_tickets_empty_folder(tmp_path: Path) -> None:
    tickets = load_tickets(tmp_path)
    assert tickets == []


def test_load_tickets_missing_folder() -> None:
    with pytest.raises(FileNotFoundError):
        load_tickets("/nonexistent/path/abc123")


def test_load_tickets_skips_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("NOT_JSON", encoding="utf-8")
    (tmp_path / "T001.json").write_text(
        json.dumps({"ticket_id": "T001", "description": "OK"}), encoding="utf-8"
    )
    tickets = load_tickets(tmp_path)
    assert len(tickets) == 1
    assert tickets[0].ticket_id == "T001"
