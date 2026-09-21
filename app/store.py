"""SQLite persistence for app-owned state (watchlist, alerts, trade log).

Lives in stocksteller.db at the project root. Schema is created lazily on first
connect, so there's no migration step yet.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "stocksteller.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS watchlist (
            symbol    TEXT PRIMARY KEY,
            added_at  REAL NOT NULL
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS orders (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         REAL NOT NULL,
            symbol            TEXT NOT NULL,
            side              TEXT NOT NULL,
            quantity          REAL NOT NULL,
            order_type        TEXT NOT NULL,
            limit_price       REAL,
            status            TEXT NOT NULL,
            fill_price        REAL,
            realized_pnl      REAL NOT NULL DEFAULT 0,
            rejection_reason  TEXT
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


# ---- Order log ----------------------------------------------------------------

def log_order(
    *,
    symbol: str,
    side: str,
    quantity: float,
    order_type: str,
    limit_price: float | None,
    status: str,
    fill_price: float | None,
    realized_pnl: float = 0.0,
    rejection_reason: str | None = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO orders
               (timestamp, symbol, side, quantity, order_type, limit_price,
                status, fill_price, realized_pnl, rejection_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), symbol, side, quantity, order_type, limit_price,
                status, fill_price, realized_pnl, rejection_reason,
            ),
        )
        return int(cur.lastrowid or 0)


def list_orders(limit: int = 50) -> list[dict]:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM orders ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def daily_realized_pnl(since_ts: float) -> float:
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM orders WHERE timestamp >= ? AND status='filled'",
            (since_ts,),
        ).fetchone()
        return float(row[0] or 0)


# ---- App state (key/value, used for kill switch) ------------------------------

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


def halted() -> bool:
    return get_state("halted", "0") == "1"


def set_halted(value: bool) -> None:
    set_state("halted", "1" if value else "0")
