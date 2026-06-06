"""Ticket Loader — reads JSON support tickets from a directory."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.models.ticket import Ticket

logger = logging.getLogger(__name__)


def load_tickets(folder_path: str | Path) -> List[Ticket]:
    """
    Load all JSON ticket files from *folder_path*.

    Args:
        folder_path: Path to a directory containing ``*.json`` ticket files.

    Returns:
        A sorted list of :class:`Ticket` objects.

    Raises:
        FileNotFoundError: If *folder_path* does not exist.
        NotADirectoryError: If *folder_path* is not a directory.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Ticket folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    tickets: List[Ticket] = []
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", folder)
        return tickets

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            ticket = Ticket(**data)
            tickets.append(ticket)
            logger.debug("Loaded ticket %s from %s", ticket.ticket_id, json_file.name)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.error("Skipping %s — parse error: %s", json_file.name, exc)

    logger.info("Loaded %d ticket(s) from '%s'", len(tickets), folder)
    return tickets
