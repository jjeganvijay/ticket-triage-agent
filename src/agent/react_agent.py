"""
ReAct-style Agent Loop for Ticket Triage.

Pattern per iteration:
    Thought  → what does the agent want to do next?
    Action   → which tool/function does it call?
    Observation → what was the result?

The agent continues until all tickets are classified and persisted.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Callable, List

from src.classifier.gemini_classifier import GeminiClassifier
from src.loader.ticket_loader import load_tickets
from src.models.ticket import AgentRun, AgentStep, Ticket, TriageResult

logger = logging.getLogger(__name__)


class TicketTriageAgent:
    """
    ReAct agent that discovers, classifies, and records support tickets.

    Each phase of work is wrapped in a Thought→Action→Observation triple
    so the full reasoning trace is preserved in the returned :class:`AgentRun`.
    """

    def __init__(self, classifier: GeminiClassifier | None = None) -> None:
        self.classifier = classifier or GeminiClassifier()
        self._step_n = 0

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _make_step(self, thought: str, action: str, observation: str) -> AgentStep:
        self._step_n += 1
        step = AgentStep(
            step=self._step_n,
            thought=thought,
            action=action,
            observation=observation,
        )
        logger.info("[Step %d] THOUGHT: %s", step.step, thought)
        logger.info("[Step %d] ACTION : %s", step.step, action)
        logger.info("[Step %d] OBS    : %s", step.step, observation)
        return step

    @staticmethod
    def _print_step(step: AgentStep) -> None:
        print(
            f"\n{'─'*60}\n"
            f"[Step {step.step}]\n"
            f"  Thought    : {step.thought}\n"
            f"  Action     : {step.action}\n"
            f"  Observation: {step.observation}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        tickets_folder: str,
        *,
        verbose: bool = True,
        on_result: Callable[[TriageResult], None] | None = None,
    ) -> AgentRun:
        """
        Execute the full ReAct triage loop.

        Args:
            tickets_folder: Directory containing ``*.json`` ticket files.
            verbose: Print step traces to stdout.
            on_result: Optional callback invoked after each ticket is classified.

        Returns:
            :class:`AgentRun` with full trace and list of :class:`TriageResult`.
        """
        self._step_n = 0
        run = AgentRun(run_id=str(uuid.uuid4()))

        def emit(step: AgentStep) -> None:
            run.steps.append(step)
            if verbose:
                self._print_step(step)

        # ── Step 1: Discover tickets ──────────────────────────────────
        thought = "I need to discover all support tickets waiting for triage."
        action = f"load_tickets(folder='{tickets_folder}')"
        try:
            tickets: List[Ticket] = load_tickets(tickets_folder)
            ids = [t.ticket_id for t in tickets]
            observation = f"Found {len(tickets)} ticket(s): {ids}"
        except (FileNotFoundError, NotADirectoryError) as exc:
            observation = f"ERROR — {exc}"
            emit(self._make_step(thought, action, observation))
            return run

        emit(self._make_step(thought, action, observation))
        run.total_tickets = len(tickets)

        if not tickets:
            emit(
                self._make_step(
                    "No tickets found. Nothing to do.",
                    "exit()",
                    "Agent finished with 0 tickets.",
                )
            )
            return run

        # ── Steps 2…N+1: Classify each ticket ────────────────────────
        results: List[TriageResult] = []
        for ticket in tickets:
            desc_preview = (
                ticket.description[:70] + "…"
                if len(ticket.description) > 70
                else ticket.description
            )
            thought = f"Classify ticket {ticket.ticket_id}: '{desc_preview}'"
            action = f"gemini_classifier.classify(ticket_id='{ticket.ticket_id}')"

            try:
                raw = self.classifier.classify(ticket)
                result = TriageResult(
                    ticket_id=ticket.ticket_id,
                    description=ticket.description,
                    category=raw["category"],
                    priority=raw["priority"],
                    reasoning=raw["reasoning"],
                    processed_at=datetime.utcnow(),
                )
                results.append(result)
                run.processed += 1
                observation = (
                    f"→ {result.category} / {result.priority}. "
                    f"Reasoning: {result.reasoning}"
                )
                if on_result:
                    on_result(result)
            except Exception as exc:  # noqa: BLE001
                run.errors += 1
                observation = f"ERROR classifying {ticket.ticket_id}: {exc}"
                logger.error(observation)

            emit(self._make_step(thought, action, observation))

        run.results = results

        # ── Final step ───────────────────────────────────────────────
        emit(
            self._make_step(
                "All tickets processed. Triage complete.",
                "return AgentRun(results)",
                (
                    f"Processed {run.processed}/{run.total_tickets} tickets "
                    f"({run.errors} error(s))."
                ),
            )
        )
        return run
