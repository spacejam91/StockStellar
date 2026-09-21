"""Point-in-time scan log, in the app's own stockstellar.db.

Tables are namespaced `scan_*` so they sit alongside the app's watchlist/orders
tables without collision, and follow app/store.py's conventions: lazy
CREATE TABLE IF NOT EXISTS on connect, module-level functions, no migrations.

The discipline this module exists to enforce: a score is written once, as it
looked that day, and never edited. Forward returns arrive later, in their own
table. There is deliberately no "rescore history" function -- recomputing a past
score with today's universe, today's listings or today's restated fundamentals
is survivorship plus lookahead bias, and it manufactures an edge that isn't
there.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "stockstellar.db"

SCORE_COLS = ["ticker", "market", "sector", "side", "composite_raw", "composite_z",
              "score_pct", "n_qualifiers", "pillar_volume", "pillar_displacement",
              "pillar_range", "pillar_structure", "pillar_relstrength",
              "pillar_confirmation", "pillar_quality", "pillar_value", "pillar_revisions"]

PICK_COLS = ["rank", "ticker", "market", "sector", "side", "composite_z", "score_pct",
             "n_qualifiers", "close", "atr14", "rvol", "ret_z", "range_ratio",
             "gap_atr", "drivers", "flags"]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS scan_run (
        run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        run_ts        REAL NOT NULL,
        as_of_date    TEXT NOT NULL,
        provider      TEXT NOT NULL,
        universe_size INTEGER NOT NULL,
        n_eligible    INTEGER NOT NULL,
        config_json   TEXT NOT NULL,
        code_version  TEXT
    )""")
    # Full cross-section, every eligible name, every session: the IC is computed
    # from this, and picks alone cannot tell you whether the score ranks anything.
    c.execute(f"""CREATE TABLE IF NOT EXISTS scan_score (
        run_id     INTEGER NOT NULL,
        as_of_date TEXT NOT NULL,
        {', '.join(f'{col} {"TEXT" if col in ("ticker","market","sector","side") else "REAL"}' for col in SCORE_COLS)},
        PRIMARY KEY (as_of_date, ticker, run_id)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_scan_score_date ON scan_score(as_of_date)")
    c.execute("CREATE INDEX IF NOT EXISTS ix_scan_score_ticker ON scan_score(ticker, as_of_date)")
    c.execute(f"""CREATE TABLE IF NOT EXISTS scan_pick (
        run_id     INTEGER NOT NULL,
        as_of_date TEXT NOT NULL,
        {', '.join(f'{col} {"TEXT" if col in ("ticker","market","sector","side","drivers","flags") else ("INTEGER" if col in ("rank","n_qualifiers") else "REAL")}' for col in PICK_COLS)},
        PRIMARY KEY (as_of_date, ticker, run_id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS scan_day (
        run_id           INTEGER NOT NULL,
        as_of_date       TEXT NOT NULL,
        n_eligible       INTEGER NOT NULL,
        n_over_threshold INTEGER NOT NULL,
        n_qualified      INTEGER NOT NULL,
        n_picks          INTEGER NOT NULL,
        max_composite_z  REAL,
        ceiling_z        REAL,
        empty_day        INTEGER NOT NULL,
        PRIMARY KEY (as_of_date, run_id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS forward_return (
        as_of_date TEXT NOT NULL,
        ticker     TEXT NOT NULL,
        fwd_1d     REAL,
        fwd_5d     REAL,
        fwd_20d    REAL,
        filled_at  REAL,
        PRIMARY KEY (as_of_date, ticker)
    )""")
    c.execute("""CREATE VIEW IF NOT EXISTS v_pick_outcome AS
        SELECT p.as_of_date, p.ticker, p.side, p.composite_z, p.score_pct, p.drivers,
               CASE WHEN p.side='short' THEN -f.fwd_1d  ELSE f.fwd_1d  END AS signed_1d,
               CASE WHEN p.side='short' THEN -f.fwd_5d  ELSE f.fwd_5d  END AS signed_5d,
               CASE WHEN p.side='short' THEN -f.fwd_20d ELSE f.fwd_20d END AS signed_20d
        FROM scan_pick p LEFT JOIN forward_return f
          ON f.as_of_date = p.as_of_date AND f.ticker = p.ticker""")
    return c


def code_version() -> str:
    """Content hash of the scoring path, so a logged score traces to its code."""
    import hashlib
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in sorted(("config.py", "mathx.py", "metrics.py", "gates.py", "rank.py",
                        "combine.py", "select.py")):
        p = here / name
        if p.exists():
            h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()[:12]


def _rows(df: pd.DataFrame, cols: list[str], run_id: int, as_of: str) -> list[tuple]:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d = d[cols].replace({np.nan: None})
    return [(run_id, as_of, *rec) for rec in d.itertuples(index=False, name=None)]


def write_scan(*, as_of, scored: pd.DataFrame, picks: pd.DataFrame, summary: pd.DataFrame,
               config: dict, universe_size: int, provider: str) -> int:
    """Write one session. Re-running a date creates a new run_id rather than
    overwriting, so a re-run is visible instead of silent."""
    as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    day = scored[scored["date"] == pd.Timestamp(as_of)]
    day_picks = picks[picks["date"] == pd.Timestamp(as_of)] if len(picks) else picks

    with _conn() as c:
        cur = c.execute(
            """INSERT INTO scan_run (run_ts, as_of_date, provider, universe_size,
                                     n_eligible, config_json, code_version)
               VALUES (?,?,?,?,?,?,?)""",
            (time.time(), as_of_str, provider, int(universe_size), int(len(day)),
             json.dumps(config, default=str, sort_keys=True), code_version()))
        run_id = int(cur.lastrowid or 0)

        if len(day):
            c.executemany(
                f"INSERT OR REPLACE INTO scan_score (run_id, as_of_date, {','.join(SCORE_COLS)}) "
                f"VALUES ({','.join('?' * (len(SCORE_COLS) + 2))})",
                _rows(day, SCORE_COLS, run_id, as_of_str))
        if len(day_picks):
            c.executemany(
                f"INSERT OR REPLACE INTO scan_pick (run_id, as_of_date, {','.join(PICK_COLS)}) "
                f"VALUES ({','.join('?' * (len(PICK_COLS) + 2))})",
                _rows(day_picks, PICK_COLS, run_id, as_of_str))

        row = summary[summary["date"] == pd.Timestamp(as_of)]
        if len(row):
            r = row.iloc[0]
            c.execute(
                """INSERT OR REPLACE INTO scan_day (run_id, as_of_date, n_eligible,
                       n_over_threshold, n_qualified, n_picks, max_composite_z,
                       ceiling_z, empty_day) VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, as_of_str, int(r["n_eligible"]), int(r["n_over_threshold"]),
                 int(r["n_qualified"]), int(r["n_picks"]),
                 r["max_composite_z"], r["ceiling_z"], int(bool(r["empty_day"]))))
    return run_id


def backfill_forward_returns(bars: pd.DataFrame, horizons=(1, 5, 20)) -> int:
    """Fill outcomes for logged (date, ticker) pairs whose future has happened.

    Writes only the outcome table; scores are never touched. Run it nightly.
    """
    with _conn() as c:
        logged = pd.read_sql_query("SELECT DISTINCT as_of_date, ticker FROM scan_score", c)
    if logged.empty:
        return 0

    b = bars[["date", "ticker", "close"]].sort_values(["ticker", "date"]).copy()
    g = b.groupby("ticker", sort=False)["close"]
    for h in horizons:
        b[f"fwd_{h}d"] = g.shift(-h) / b["close"] - 1.0

    logged["date"] = pd.to_datetime(logged["as_of_date"])
    cols = [f"fwd_{h}d" for h in horizons]
    m = logged.merge(b.drop(columns=["close"]), on=["date", "ticker"], how="left").dropna(
        subset=cols, how="all")
    if m.empty:
        return 0
    now = time.time()
    rows = [(r.as_of_date, r.ticker,
             *[None if pd.isna(getattr(r, c)) else float(getattr(r, c)) for c in cols], now)
            for r in m.itertuples(index=False)]
    with _conn() as c:
        c.executemany(
            f"INSERT OR REPLACE INTO forward_return (as_of_date, ticker, {','.join(cols)}, filled_at)"
            f" VALUES ({','.join('?' * (len(cols) + 3))})", rows)
    return len(rows)


# ---- read-back ---------------------------------------------------------------

def latest_scan_date() -> str | None:
    with _conn() as c:
        row = c.execute("SELECT MAX(as_of_date) FROM scan_day").fetchone()
        return row[0] if row and row[0] else None


def picks_for(as_of: str | None = None) -> list[dict]:
    """Picks for a session (default: the latest logged), newest run only."""
    as_of = as_of or latest_scan_date()
    if not as_of:
        return []
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT p.* FROM scan_pick p
               JOIN (SELECT MAX(run_id) rid FROM scan_pick WHERE as_of_date = ?) m
                 ON m.rid = p.run_id
               WHERE p.as_of_date = ? ORDER BY p.rank""", (as_of, as_of)).fetchall()
        return [dict(r) for r in rows]


def day_for(as_of: str | None = None) -> dict | None:
    as_of = as_of or latest_scan_date()
    if not as_of:
        return None
    with _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute(
            """SELECT d.* FROM scan_day d
               JOIN (SELECT MAX(run_id) rid FROM scan_day WHERE as_of_date = ?) m
                 ON m.rid = d.run_id WHERE d.as_of_date = ?""", (as_of, as_of)).fetchone()
        return dict(row) if row else None


def provider_breakdown() -> list[dict]:
    """Sessions logged per provider. Check this before reading any measurement."""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT provider, COUNT(DISTINCT as_of_date) AS sessions,
                      MIN(as_of_date) AS first_date, MAX(as_of_date) AS last_date
               FROM scan_run GROUP BY provider ORDER BY sessions DESC""").fetchall()
        return [dict(r) for r in rows]


def day_history(limit: int = 60) -> list[dict]:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT d.* FROM scan_day d
               JOIN (SELECT as_of_date, MAX(run_id) rid FROM scan_day GROUP BY as_of_date) m
                 ON m.as_of_date = d.as_of_date AND m.rid = d.run_id
               ORDER BY d.as_of_date DESC LIMIT ?""", (limit,)).fetchall()
        return [dict(r) for r in rows]


def scores_frame(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Logged cross-section as a frame, for the evaluation harness."""
    q = """SELECT s.* FROM scan_score s
           JOIN (SELECT as_of_date, MAX(run_id) rid FROM scan_score GROUP BY as_of_date) m
             ON m.as_of_date = s.as_of_date AND m.rid = s.run_id"""
    where = []
    if start:
        where.append(f"s.as_of_date >= '{start}'")
    if end:
        where.append(f"s.as_of_date <= '{end}'")
    if where:
        q += " WHERE " + " AND ".join(where)
    with _conn() as c:
        df = pd.read_sql_query(q, c)
    if not df.empty:
        df["date"] = pd.to_datetime(df["as_of_date"])
    return df


def score_band_base_rates(horizon: int = 5, band: int = 10, provider: str | None = None,
                          include_mock: bool = False) -> pd.DataFrame:
    """What actually followed each score band, measured from the log.

    This is the only legitimate probability-shaped output the scanner has: a
    measured base rate reported with n and a binomial standard error, never an
    asserted one. Read the SE column before reading anything else -- a band with
    a handful of observations is not a finding.

    Rows scored against the mock provider are EXCLUDED by default. Mock bars are
    a random walk, so mixing them into a base rate dilutes a real measurement
    toward 50% with a deceptively small standard error -- the one number here
    that has to be trustworthy would be the first thing corrupted. Pass
    include_mock=True only to sanity-check the harness itself.
    """
    col = f"fwd_{horizon}d"
    where = [f"f.{col} IS NOT NULL"]
    params: list = []
    if provider is not None:
        where.append("r.provider = ?")
        params.append(provider)
    elif not include_mock:
        where.append("r.provider <> 'mock'")
    q = f"""SELECT MIN(CAST(s.score_pct / {band} AS INT) * {band}, {100 - band}) AS score_band,
                   COUNT(*) AS n,
                   AVG(CASE WHEN (CASE WHEN s.side='short' THEN -f.{col} ELSE f.{col} END) > 0
                            THEN 1.0 ELSE 0.0 END) AS up_rate,
                   AVG(CASE WHEN s.side='short' THEN -f.{col} ELSE f.{col} END) AS mean_ret
            FROM scan_score s
            JOIN scan_run r ON r.run_id = s.run_id
            JOIN forward_return f
              ON f.as_of_date = s.as_of_date AND f.ticker = s.ticker
            WHERE {' AND '.join(where)}
            GROUP BY score_band ORDER BY score_band"""
    with _conn() as c:
        df = pd.read_sql_query(q, c, params=params)
    if df.empty:
        return df
    df["up_rate_pct"] = (100 * df["up_rate"]).round(1)
    df["mean_ret_pct"] = (100 * df["mean_ret"]).round(3)
    df["se_pp"] = (100 * np.sqrt(df["up_rate"] * (1 - df["up_rate"]) / df["n"])).round(1)
    return df[["score_band", "n", "up_rate_pct", "se_pp", "mean_ret_pct"]]
