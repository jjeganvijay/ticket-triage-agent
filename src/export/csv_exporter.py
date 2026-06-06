"""CSV Exporter — writes triage results to a CSV file."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

from src.models.ticket import TriageResult

logger = logging.getLogger(__name__)

CSV_HEADERS = ["ticket_id", "description", "category", "priority", "reasoning", "processed_at"]


def export_to_csv(results: List[TriageResult], output_path: str | Path) -> Path:
    """
    Write triage results to a CSV file.

    Args:
        results: List of :class:`TriageResult` objects.
        output_path: Destination file path (created if absent).

    Returns:
        Resolved :class:`Path` of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "ticket_id": r.ticket_id,
                    "description": r.description,
                    "category": r.category.value,
                    "priority": r.priority.value,
                    "reasoning": r.reasoning,
                    "processed_at": r.processed_at.isoformat(),
                }
            )

    logger.info("Exported %d result(s) to '%s'", len(results), path)
    return path
