"""Database package."""
from src.database.db import get_all_results, get_result_by_id, init_db, save_results

__all__ = ["init_db", "save_results", "get_all_results", "get_result_by_id"]
