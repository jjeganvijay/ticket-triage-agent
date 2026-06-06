"""
CLI entry point — runs the Ticket Triage Agent end-to-end.

Usage:
    python main.py
    python main.py --folder data/sample_tickets --csv output/results.csv --db database/tickets.db
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.agent.react_agent import TicketTriageAgent
from src.database.db import save_results
from src.export.csv_exporter import export_to_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-powered Ticket Triage Agent")
    parser.add_argument(
        "--folder",
        default="data/sample_tickets",
        help="Folder containing ticket JSON files (default: data/sample_tickets)",
    )
    parser.add_argument(
        "--csv",
        default="output/results.csv",
        help="Output CSV path (default: output/results.csv)",
    )
    parser.add_argument(
        "--db",
        default="database/tickets.db",
        help="SQLite database path (default: database/tickets.db)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("\n[AGENT] Ticket Triage Agent -- starting run\n" + "=" * 60)

    agent = TicketTriageAgent()
    run = agent.run(args.folder, verbose=not args.quiet)

    if not run.results:
        print("\n[WARN] No results produced. Check the tickets folder and API key.")
        return 1

    # Persist
    save_results(run.results, db_path=args.db)
    csv_path = export_to_csv(run.results, output_path=args.csv)

    print(f"\n{'='*60}")
    print("[OK] Triage complete!")
    print(f"   Tickets processed : {run.processed}/{run.total_tickets}")
    print(f"   Errors            : {run.errors}")
    print(f"   CSV output        : {csv_path}")
    print(f"   Database          : {args.db}")
    print(f"   Run ID            : {run.run_id}")
    print()

    # Summary table
    print(f"{'Ticket ID':<10} {'Category':<18} {'Priority':<14} Reasoning")
    print("-" * 80)
    for r in run.results:
        print(
            f"{r.ticket_id:<10} {r.category.value:<18} {r.priority.value:<14} "
            f"{r.reasoning[:50]}..."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
