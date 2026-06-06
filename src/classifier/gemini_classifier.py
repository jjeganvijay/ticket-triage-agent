"""
Gemini Classifier — uses Google Gemini API (google-genai SDK) with few-shot
prompting to classify support tickets into categories and priorities.
"""
from __future__ import annotations

import json
import logging
import os
import re

from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.models.ticket import Ticket

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot examples
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "description": "I was charged twice for my subscription this month.",
        "output": {
            "category": "Billing",
            "priority": "P2 High",
            "reasoning": (
                "Duplicate billing is a direct financial impact requiring prompt "
                "resolution to maintain customer trust."
            ),
        },
    },
    {
        "description": "The app crashes every time I try to upload a profile picture on iOS 17.",
        "output": {
            "category": "Bug",
            "priority": "P2 High",
            "reasoning": (
                "A reproducible crash on a specific platform is a software defect "
                "that degrades user experience and requires a patch."
            ),
        },
    },
    {
        "description": "Can you add a dark mode option to the dashboard?",
        "output": {
            "category": "Feature Request",
            "priority": "P4 Low",
            "reasoning": (
                "Dark mode is a UX enhancement; it improves comfort but does not "
                "block any core functionality."
            ),
        },
    },
    {
        "description": "My entire account was deleted without warning and I lost all my data permanently.",
        "output": {
            "category": "Bug",
            "priority": "P1 Critical",
            "reasoning": (
                "Unintended data loss and account deletion is a critical system failure "
                "requiring immediate investigation."
            ),
        },
    },
    {
        "description": "How do I export my monthly reports to PDF?",
        "output": {
            "category": "Other",
            "priority": "P4 Low",
            "reasoning": (
                "This is a general usage inquiry that can be addressed via documentation "
                "or standard support channels."
            ),
        },
    },
    {
        "description": "Payment gateway returns 500 on checkout, blocking all purchases.",
        "output": {
            "category": "Bug",
            "priority": "P1 Critical",
            "reasoning": (
                "A broken checkout flow stops all revenue generation and requires "
                "an immediate hotfix."
            ),
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You are an expert support ticket triage agent. "
    "You classify support tickets accurately and consistently."
)


def _build_prompt(ticket: Ticket) -> str:
    """Assemble the few-shot classification prompt."""
    examples_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        examples_block += (
            f"Example {i}:\n"
            f'Description: "{ex["description"]}"\n'
            f'Output:\n```json\n{json.dumps(ex["output"], indent=2)}\n```\n\n'
        )

    return f"""You are an expert support ticket triage agent.

## Categories
- Bug: software defects, crashes, unexpected behaviour, data corruption.
- Feature Request: new features, enhancements, or UX improvements.
- Billing: payment issues, subscription problems, refund/charge disputes.
- Other: general inquiries, how-to questions, documentation requests.

## Priority Levels
- P1 Critical: production outage, data loss, security breach, payment down.
- P2 High: major feature broken, significant user impact, no workaround.
- P3 Medium: minor bug, workaround available, moderate impact.
- P4 Low: cosmetic issues, nice-to-have features, general questions.

## Few-Shot Examples
{examples_block}## Task
Classify the ticket below. Respond ONLY with a valid JSON object — no markdown fences, no extra text.

Ticket ID : {ticket.ticket_id}
Description: "{ticket.description}"

Required JSON:
{{
  "category": "<Bug|Feature Request|Billing|Other>",
  "priority": "<P1 Critical|P2 High|P3 Medium|P4 Low>",
  "reasoning": "<one or two sentence explanation>"
}}"""


class GeminiClassifier:
    """Wraps the Gemini generative model for ticket classification."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self._model_name = os.getenv("MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=api_key)
        logger.info("GeminiClassifier ready — model: %s", self._model_name)

    def classify(self, ticket: Ticket, max_retries: int = 5, initial_delay: float = 1.0) -> dict:
        """
        Classify a single ticket via Gemini.

        Returns:
            dict with keys ``category``, ``priority``, ``reasoning``.

        Raises:
            ValueError: If the model response cannot be parsed as JSON.
        """
        import time
        from google.genai.errors import APIError

        prompt = _build_prompt(ticket)
        delay = initial_delay

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug("Sending ticket %s to Gemini (attempt %d/%d)", ticket.ticket_id, attempt, max_retries)
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                    ),
                )
                text = response.text.strip()
                break
            except Exception as exc:
                if attempt == max_retries:
                    logger.error("All %d attempts failed to classify ticket %s: %s", max_retries, ticket.ticket_id, exc)
                    raise
                logger.warning(
                    "Attempt %d failed for ticket %s due to transient error: %s. Retrying in %.2f seconds...",
                    attempt, ticket.ticket_id, exc, delay
                )
                time.sleep(delay)
                delay *= 2.0

        # Strip markdown code fences if the model adds them anyway
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Bad JSON from Gemini for %s: %s\nRaw: %s", ticket.ticket_id, exc, text)
            raise ValueError(f"Invalid JSON from Gemini: {exc}") from exc

        logger.info(
            "Ticket %s -> %s / %s",
            ticket.ticket_id,
            result.get("category"),
            result.get("priority"),
        )
        return result
