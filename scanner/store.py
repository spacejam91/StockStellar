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
import threading
import time
import uuid
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
             "gap_atr", "thr_rvol", "thr_ret_z", "thr_range_ratio", "thr_gap_atr",
             "ceiling_z", "n_market", "drivers", "flags"]


# run_id identifies a run ACROSS MACHINES, because that is how it is used: the
# CSV mirror merges rows from this laptop and from CI into one database, and
# every read joins scan_score/scan_pick/scan_day to scan_run on it.
#
# It used to be AUTOINCREMENT, which is a per-database rowid. Two machines each
# produce run_id 1, 2, 3..., so after an import a CI cross-section joins to
# whichever scan_run row happened to land on that integer -- relabelling a real
# session with another machine's provider and universe size. A mock run
# imported from a laptop could silently mark a live CI session synthetic, and
# `MAX(run_id)` -- which every "newest run" query uses -- could return an older
# run from the other machine.
#
# So: millisecond epoch in the high digits, a stable per-machine salt in the low
# three. Unique across machines, and still monotonic, which is what MAX(run_id)
# needs to mean "newest". Legacy small ids stay valid and always sort older.
_MACHINE_SALT = uuid.getnode() % 1000
_ID_LOCK = threading.Lock()
_last_run_id = 0


def _new_run_id(c: sqlite3.Connection) -> int:
    global _last_run_id
    with _ID_LOCK:
        if not _last_run_id:
            row = c.execute("SELECT MAX(run_id) FROM scan_run").fetchone()
            _last_run_id = int(row[0] or 0)
        rid = int(time.time() * 1000) * 1000 + _MACHINE_SALT
        if rid <= _last_run_id:
            # Same millisecond (write_scan runs once per session in a loop), or
            # a clock that went backwards. Step by a full 1000 so the low three
            # digits stay this machine's salt.
            rid = _last_run_id + 1000
        _last_run_id = rid
        return rid


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS scan_run (
        run_id        INTEGER PRIMARY KEY,
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
        run_id = _new_run_id(c)
        c.execute(
            """INSERT INTO scan_run (run_id, run_ts, as_of_date, provider, universe_size,
                                     n_eligible, config_json, code_version)
               VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, time.time(), as_of_str, provider, int(universe_size), int(len(day)),
             json.dumps(config, default=str, sort_keys=True), code_version()))

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
    # COALESCE, not REPLACE. A row survives the dropna above if ANY horizon is
    # present, so a name whose 1d outcome exists but whose 20d bar is missing --
    # a shallower refetch, a delisting, a provider gap -- used to REPLACE a row
    # that already held a measured 20d return with NULL. The log's whole value
    # is that a matured outcome is final, and this quietly unmatured them.
    sets = ", ".join(f"{h} = COALESCE(excluded.{h}, forward_return.{h})" for h in cols)
    with _conn() as conn:
        conn.executemany(
            f"""INSERT INTO forward_return (as_of_date, ticker, {','.join(cols)}, filled_at)
                VALUES ({','.join('?' * (len(cols) + 3))})
                ON CONFLICT(as_of_date, ticker) DO UPDATE SET {sets},
                    filled_at = excluded.filled_at""", rows)
    return len(rows)


CA_SUFFIXES = (".TO", ".V", ".CN", ".NE")


def backfill_market() -> int:
    """Set scan_score.market from the ticker suffix where it is NULL.

    Not a rescore and not a lookahead: which exchange a symbol trades on is a
    property of the STRING, decided the same way the provider decides it, and
    it does not depend on anything that happened after the session.

    It is needed because the CSV mirror used to drop market and sector from
    scan_score on the theory that data/universe_*.csv could supply them later.
    A database rebuilt from git -- which is what this one is -- therefore came
    back with both columns NULL for every session ever logged. market is
    recoverable; sector is not, and joining today's listing file to a two-year
    old cross-section would be survivorship bias, so those stay NULL and
    honest.
    """
    marks = " OR ".join("ticker LIKE ?" for _ in CA_SUFFIXES)
    params = [f"%{sfx}" for sfx in CA_SUFFIXES]
    with _conn() as c:
        cur = c.execute(
            f"UPDATE scan_score SET market = CASE WHEN {marks} THEN 'CA' ELSE 'US' END "
            f"WHERE market IS NULL", params)
        return int(cur.rowcount or 0)


# ---- read-back ---------------------------------------------------------------


# A session below this share of its neighbours' median eligible count is a
# degraded fetch, not a quiet market.
DEGRADED_FRAC = 0.70


def _synthetic_filter(include_synthetic: bool) -> tuple[str, list]:
    """SQL fragment excluding generated providers, and its params.

    The same rule the base rates use, applied to the DISPLAY surfaces too. A
    mock session dated today outranked a real one dated the last trading day
    and became "latest", so the page rendered synthetic bars. It was labelled
    "mock", which is the only reason it was caught -- but a page that is
    supposed to show the market must not be able to show generated data at all.
    """
    if include_synthetic:
        return "", []
    from app.market_data import SYNTHETIC_PROVIDERS
    names = sorted(SYNTHETIC_PROVIDERS)
    return f" AND r.provider NOT IN ({','.join('?' * len(names))})", names


def latest_scan_date(include_synthetic: bool = False,
                     include_degraded: bool = False) -> str | None:
    """The newest session worth showing as "today".

    Degraded sessions are skipped for the same reason synthetic ones are: the
    page is supposed to show the market, and a session where the provider
    returned a third of it does not. It is still logged, still reachable from
    the history table, and still carries its banner -- it just does not get to
    be the front page. A CI run logged 330 of ~3,080 eligible names and became
    the published session; every percentile on it was computed against a market
    that was not there.
    """
    clause, params = _synthetic_filter(include_synthetic)
    with _conn() as c:
        rows = c.execute(
            f"""SELECT d.as_of_date, d.n_eligible FROM scan_day d
                JOIN scan_run r ON r.run_id = d.run_id
                WHERE 1=1{clause}
                ORDER BY d.as_of_date DESC, d.run_id DESC""", params).fetchall()
    if not rows:
        return None
    if include_degraded:
        return rows[0][0]
    # One row per date, newest run first.
    seen: dict[str, int] = {}
    for date, n in rows:
        seen.setdefault(date, int(n or 0))
    dates = sorted(seen, reverse=True)
    counts = [seen[d] for d in dates[:30]]
    if len(counts) < 6:
        return dates[0]                      # no baseline to judge against
    for d in dates:
        others = [seen[x] for x in dates[:30] if x != d]
        med = float(np.median(others)) if others else 0.0
        if med <= 0 or seen[d] >= DEGRADED_FRAC * med:
            return d
    return dates[0]                          # everything looks thin; show the newest


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


def day_for(as_of: str | None = None, include_synthetic: bool = False) -> dict | None:
    as_of = as_of or latest_scan_date(include_synthetic)
    if not as_of:
        return None
    with _conn() as c:
        c.row_factory = sqlite3.Row
        # Join scan_run for `provider`. Callers need to know which data source
        # produced a session and cannot infer it from the environment: the env
        # var reflects how the process is configured NOW, not how this row was
        # produced. Reading it from os.getenv mislabels real sessions as mock
        # the moment the variable is unset, which is exactly backwards -- it
        # makes live output look like test output.
        clause, params = _synthetic_filter(include_synthetic)
        row = c.execute(
            f"""SELECT d.*, r.provider FROM scan_day d
                JOIN scan_run r ON r.run_id = d.run_id
                WHERE d.as_of_date = ?{clause}
                ORDER BY d.run_id DESC LIMIT 1""", (as_of, *params)).fetchone()
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


def day_history(limit: int = 60, include_synthetic: bool = False) -> list[dict]:
    clause, params = _synthetic_filter(include_synthetic)
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            f"""SELECT d.*, r.provider FROM scan_day d
                JOIN scan_run r ON r.run_id = d.run_id
                JOIN (SELECT as_of_date, MAX(run_id) rid FROM scan_day GROUP BY as_of_date) m
                  ON m.as_of_date = d.as_of_date AND m.rid = d.run_id
                WHERE 1=1{clause}
                ORDER BY d.as_of_date DESC LIMIT ?""", (*params, limit)).fetchall()
    out = [dict(r) for r in rows]
    # Mark the thin ones in place, so the history table can say which sessions
    # not to read rather than presenting 330 eligible names next to 3,096 as if
    # they were the same kind of number.
    counts = [int(r["n_eligible"] or 0) for r in out]
    if len(counts) >= 6:
        for i, r in enumerate(out):
            others = counts[:i] + counts[i + 1:]
            med = float(np.median(others))
            r["degraded"] = bool(med > 0 and counts[i] < DEGRADED_FRAC * med)
    return out


# A published session is not self-describing. n_eligible is just a number on the
# page, and the one that matters -- "is this smaller than it should be" -- needs
# the sessions around it. CI and this laptop scanned the SAME session and
# recorded 1,076 vs 3,092 eligible names, a 65% collapse caused by the data
# provider rate-limiting a datacenter IP, and the page rendered the thin one as
# an ordinary quiet day. composite_z is a percentile of whoever showed up, so a
# third of the market missing does not make the list shorter, it makes every
# score in it wrong.
def session_health(day: dict | None, history: list[dict]) -> dict | None:
    """Is this session's eligible count consistent with the ones around it?"""
    if not day or not history:
        return None
    others = [int(r["n_eligible"]) for r in history
              if r["as_of_date"] != day["as_of_date"] and r.get("n_eligible")]
    if len(others) < 5:
        return None                      # not enough baseline to call anything
    med = float(np.median(others))
    if med <= 0:
        return None
    n = int(day["n_eligible"])
    frac = n / med
    return {
        "n_eligible": n,
        "median": int(round(med)),
        "frac": round(frac, 3),
        "shortfall_pct": round(100 * (1 - frac), 1),
        "degraded": bool(frac < DEGRADED_FRAC),
        "sessions_compared": len(others),
    }


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

    Rows from EVERY synthetic provider (see SYNTHETIC_PROVIDERS) are excluded
    by default. 'mock' and 'null' are random walks and would dilute a real
    measurement toward 50% with a deceptively small standard error. 'signal'
    is worse: it plants a strong artificial edge by design, so including it
    would report a fabricated edge as a measurement. The one number here that
    has to be trustworthy would be the first thing corrupted. Pass
    include_mock=True only to sanity-check the harness itself.
    """
    col = f"fwd_{horizon}d"
    where = [f"f.{col} IS NOT NULL"]
    params: list = []
    if provider is not None:
        where.append("r.provider = ?")
        params.append(provider)
    elif not include_mock:
        # Exclude EVERY synthetic provider, not just 'mock'. This filter used
        # to read `r.provider <> 'mock'`, which silently admitted 'null' and
        # 'signal' once those were added -- and 'signal' plants an artificial
        # edge on purpose, so its rows would have shown up here as a measured
        # one. Fail loudly on an unknown provider rather than guessing: a
        # missing measurement is obvious, a fabricated one is not.
        from app.market_data import SYNTHETIC_PROVIDERS
        marks = ",".join("?" * len(SYNTHETIC_PROVIDERS))
        where.append(f"r.provider NOT IN ({marks})")
        params.extend(sorted(SYNTHETIC_PROVIDERS))
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
