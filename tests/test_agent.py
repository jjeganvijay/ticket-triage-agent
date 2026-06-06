"""Unit tests for the ReAct agent loop."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.react_agent import TicketTriageAgent
from src.models.ticket import Category, Priority, TriageResult


def _make_mock_classifier(category: str = "Bug", priority: str = "P2 High") -> MagicMock:
    mock = MagicMock()
    mock.classify.return_value = {
        "category": category,
        "priority": priority,
        "reasoning": "Mocked reasoning.",
    }
    return mock


@pytest.fixture()
def ticket_dir(tmp_path: Path) -> Path:
    tickets = [
        {"ticket_id": "T001", "description": "Login crash."},
        {"ticket_id": "T002", "description": "Charged twice."},
    ]
    for t in tickets:
        (tmp_path / f"{t['ticket_id']}.json").write_text(json.dumps(t))
    return tmp_path


def test_agent_run_produces_results(ticket_dir: Path) -> None:
    agent = TicketTriageAgent(classifier=_make_mock_classifier())
    run = agent.run(str(ticket_dir), verbose=False)
    assert run.processed == 2
    assert run.total_tickets == 2
    assert run.errors == 0
    assert len(run.results) == 2


def test_agent_run_result_types(ticket_dir: Path) -> None:
    agent = TicketTriageAgent(classifier=_make_mock_classifier())
    run = agent.run(str(ticket_dir), verbose=False)
    for r in run.results:
        assert isinstance(r, TriageResult)
        assert isinstance(r.category, Category)
        assert isinstance(r.priority, Priority)


def test_agent_run_has_steps(ticket_dir: Path) -> None:
    agent = TicketTriageAgent(classifier=_make_mock_classifier())
    run = agent.run(str(ticket_dir), verbose=False)
    # 1 discovery step + 2 classify steps + 1 final step = 4 minimum
    assert len(run.steps) >= 4


def test_agent_run_empty_folder(tmp_path: Path) -> None:
    agent = TicketTriageAgent(classifier=_make_mock_classifier())
    run = agent.run(str(tmp_path), verbose=False)
    assert run.total_tickets == 0
    assert run.processed == 0


def test_agent_run_missing_folder() -> None:
    agent = TicketTriageAgent(classifier=_make_mock_classifier())
    run = agent.run("/nonexistent/path", verbose=False)
    assert run.processed == 0
    # Should capture the error in an observation, not raise
    assert any("ERROR" in s.observation for s in run.steps)


def test_agent_on_result_callback(ticket_dir: Path) -> None:
    collected = []
    agent = TicketTriageAgent(classifier=_make_mock_classifier())
    agent.run(str(ticket_dir), verbose=False, on_result=collected.append)
    assert len(collected) == 2
