"""Unit tests for the Gemini classifier (mocked)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.classifier.gemini_classifier import GeminiClassifier, _build_prompt
from src.models.ticket import Ticket


@pytest.fixture()
def sample_ticket() -> Ticket:
    return Ticket(ticket_id="T001", description="I was charged twice for my subscription.")


def test_build_prompt_contains_ticket_id(sample_ticket: Ticket) -> None:
    prompt = _build_prompt(sample_ticket)
    assert "T001" in prompt


def test_build_prompt_contains_description(sample_ticket: Ticket) -> None:
    prompt = _build_prompt(sample_ticket)
    assert "charged twice" in prompt


def test_build_prompt_contains_few_shot_examples(sample_ticket: Ticket) -> None:
    prompt = _build_prompt(sample_ticket)
    assert "Example 1" in prompt
    assert "Example 6" in prompt


@patch("src.classifier.gemini_classifier.genai")
def test_classify_returns_correct_keys(mock_genai: MagicMock, sample_ticket: Ticket) -> None:
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {"category": "Billing", "priority": "P2 High", "reasoning": "Double charge."}
    )
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key", "MODEL": "gemini-2.5-flash"}):
        classifier = GeminiClassifier()
        result = classifier.classify(sample_ticket)

    assert set(result.keys()) == {"category", "priority", "reasoning"}
    assert result["category"] == "Billing"
    assert result["priority"] == "P2 High"


@patch("src.classifier.gemini_classifier.genai")
def test_classify_strips_markdown_fences(mock_genai: MagicMock, sample_ticket: Ticket) -> None:
    mock_response = MagicMock()
    mock_response.text = (
        "```json\n"
        '{"category": "Bug", "priority": "P1 Critical", "reasoning": "Crash."}\n'
        "```"
    )
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        classifier = GeminiClassifier()
        result = classifier.classify(sample_ticket)

    assert result["category"] == "Bug"


@patch("src.classifier.gemini_classifier.genai")
def test_classify_raises_on_bad_json(mock_genai: MagicMock, sample_ticket: Ticket) -> None:
    mock_response = MagicMock()
    mock_response.text = "NOT_VALID_JSON"
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        classifier = GeminiClassifier()
        with pytest.raises(ValueError, match="Invalid JSON"):
            classifier.classify(sample_ticket)


def test_classifier_raises_without_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        import os
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            GeminiClassifier()
