"""SQLite persistence for app-owned state (the manual watchlist).

The scanner keeps its own point-in-time log separately, in scanner/store.py.

Lives in stockstellar.db at the project root. Schema is created lazily on first
connect, so there's no migration step yet.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "stockstellar.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS watchlist (
            symbol    TEXT PRIMARY KEY,
            added_at  REAL NOT NULL
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS app_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )
    return c


def list_symbols() -> list[str]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT symbol FROM watchlist ORDER BY added_at")]


def add_symbol(symbol: str) -> None:
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol cannot be empty")
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
            (sym, time.time()),
        )


def remove_symbol(symbol: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.strip().upper(),))


# ---- App state (key/value) ----------------------------------------------------

def get_state(key: str, default: str = "") -> str:
    with _conn() as c:
        row = c.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else default


def set_state(key: str, value: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
