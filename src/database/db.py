"""SQLite persistence layer for triage results."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from src.models.ticket import Category, Priority, TriageResult

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "database/tickets.db"


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the tickets table if it does not exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triage_results (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id    TEXT    NOT NULL UNIQUE,
                description  TEXT    NOT NULL,
                category     TEXT    NOT NULL,
                priority     TEXT    NOT NULL,
                reasoning    TEXT    NOT NULL,
                processed_at TEXT    NOT NULL
            )
            """
        )
        conn.commit()
    logger.info("Database initialised at '%s'", db_path)


def save_results(
    results: List[TriageResult],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    Upsert triage results into the database.

    Returns:
        Number of rows inserted or replaced.
    """
    init_db(db_path)
    rows = [
        (
            r.ticket_id,
            r.description,
            r.category.value,
            r.priority.value,
            r.reasoning,
            r.processed_at.isoformat(),
        )
        for r in results
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO triage_results
                (ticket_id, description, category, priority, reasoning, processed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                description  = excluded.description,
                category     = excluded.category,
                priority     = excluded.priority,
                reasoning    = excluded.reasoning,
                processed_at = excluded.processed_at
            """,
            rows,
        )
        conn.commit()
    logger.info("Saved %d result(s) to database.", len(rows))
    return len(rows)


def get_all_results(db_path: str | Path = DEFAULT_DB_PATH) -> List[TriageResult]:
    """Return all triage results from the database."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM triage_results ORDER BY processed_at DESC"
        ).fetchall()
    from datetime import datetime

    return [
        TriageResult(
            ticket_id=row["ticket_id"],
            description=row["description"],
            category=Category(row["category"]),
            priority=Priority(row["priority"]),
            reasoning=row["reasoning"],
            processed_at=datetime.fromisoformat(row["processed_at"]),
        )
        for row in rows
    ]


def get_result_by_id(
    ticket_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Optional[TriageResult]:
    """Return the triage result for a specific ticket ID, or None."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM triage_results WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    if row is None:
        return None
    from datetime import datetime

    return TriageResult(
        ticket_id=row["ticket_id"],
        description=row["description"],
        category=Category(row["category"]),
        priority=Priority(row["priority"]),
        reasoning=row["reasoning"],
        processed_at=datetime.fromisoformat(row["processed_at"]),
    )
