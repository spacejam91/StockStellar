"""Append-only text mirror of the point-in-time log.

The log is the one artifact in this project that must never be destroyed: a
score is written once, as it looked that day, and its value comes entirely
from not having been touched since.

Committing the SQLite file itself was a mistake and cost a real 90-session log.
A binary blob cannot be merged, so `git pull` resolves it by overwrite -- CI
built a fresh one-session database, committed it, and the next pull replaced
ninety sessions of local history with it. The repo also grew by a full ~700 KB
blob per run.

So the DB stays local and untracked, and git carries CSVs instead. Text merges
line-wise, two machines that scanned different sessions both keep their rows,
and a conflict is readable rather than a binary stalemate.

    export_log()   DB  -> data/log/*.csv   (after a scan)
    import_log()   CSV -> DB               (before a scan, on a fresh checkout)

import_log() is additive. It inserts rows the DB does not already have and
never updates or deletes an existing one, so replaying an old export cannot
rewrite history that is already recorded.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from scanner.store import DB_PATH, _conn

log = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "data" / "log"

# scan_score is the bulk of the log: every eligible name, every session. It is
# written one gzipped file PER SESSION rather than one big CSV, for two
# reasons. A daily run then adds a new small file instead of rewriting a
# 54 MB one, and two machines that scanned different days can never conflict,
# because they touch different files.
#
# Only the columns something actually reads are kept. scripts/backtest.py is
# the sole consumer and uses 8 of 19; the nine pillar_* columns and
# composite_raw are reconstructible from a rescan and are not worth carrying
# in git forever. market and sector are properties of the ticker, joinable
# from data/universe_*.csv, so they are dropped too.
SCORE_EXPORT_COLS = ["run_id", "ticker", "side", "composite_z", "score_pct", "n_qualifiers"]

# table -> the columns that identify a row uniquely
KEYS = {
    "scan_run": ["run_id"],
    "scan_day": ["run_id", "as_of_date"],
    "scan_score": ["run_id", "as_of_date", "ticker"],
    "scan_pick": ["run_id", "as_of_date", "ticker"],
    "forward_return": ["as_of_date", "ticker"],
}


def export_log(out: Path | None = None) -> dict[str, int]:
    """Write the log to text. Sorted, so diffs stay small and readable."""
    out = out or LOG_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    counts = {}
    with _conn() as c:
        # scan_score: one gzipped file per session, written once.
        try:
            dates = pd.read_sql_query(
                "SELECT DISTINCT as_of_date FROM scan_score ORDER BY as_of_date", c)
            n = 0
            for d in dates["as_of_date"]:
                f = out / "scores" / f"{d}.csv.gz"
                if f.exists():
                    continue          # a logged session never changes
                df = pd.read_sql_query(
                    f"SELECT {','.join(SCORE_EXPORT_COLS)} FROM scan_score "
                    f"WHERE as_of_date = ? ORDER BY ticker", c, params=(d,))
                df.to_csv(f, index=False, compression="gzip")
                n += len(df)
            counts["scan_score"] = n
        except (pd.errors.DatabaseError, sqlite3.Error):
            pass

        for table, keys in KEYS.items():
            if table == "scan_score":
                continue
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", c)
            except (pd.errors.DatabaseError, sqlite3.Error):
                continue
            if df.empty:
                counts[table] = 0
                continue
            df = df.sort_values([k for k in keys if k in df.columns])
            df.to_csv(out / f"{table}.csv", index=False)
            counts[table] = len(df)
    return counts


def import_log(src: Path | None = None) -> dict[str, int]:
    """Load CSV rows the DB does not already have. Never updates or deletes.

    Additive on purpose: an import must be able to run on any machine, in any
    order, without rewriting a score that is already logged.
    """
    src = src or LOG_DIR
    if not src.exists():
        return {}
    added = {}
    with _conn() as c:
        for table, keys in KEYS.items():
            if table == "scan_score":
                files = sorted((src / "scores").glob("*.csv.gz")) if (src / "scores").exists() else []
                if not files:
                    continue
                parts = []
                for f in files:
                    d = pd.read_csv(f, compression="gzip")
                    d["as_of_date"] = f.name.removesuffix(".csv.gz")
                    parts.append(d)
                incoming = pd.concat(parts, ignore_index=True)
            else:
                f = src / f"{table}.csv"
                if not f.exists():
                    continue
                incoming = pd.read_csv(f)
            if incoming.empty:
                added[table] = 0
                continue
            try:
                have = pd.read_sql_query(f"SELECT {','.join(keys)} FROM {table}", c)
            except (pd.errors.DatabaseError, sqlite3.Error):
                have = pd.DataFrame(columns=keys)

            if not have.empty:
                merged = incoming.merge(have.assign(_seen=1), on=keys, how="left")
                incoming = merged[merged["_seen"].isna()].drop(columns=["_seen"])
            if incoming.empty:
                added[table] = 0
                continue

            cols = ",".join(incoming.columns)
            marks = ",".join("?" * len(incoming.columns))
            c.executemany(
                f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({marks})",
                incoming.where(pd.notna(incoming), None).itertuples(index=False, name=None))
            added[table] = len(incoming)
    return added


def verify(src: Path | None = None) -> list[str]:
    """Every session in scan_day must have a score partition on disk.

    The CI commit step staged only data/log/*.csv, so data/log/scores/ was never
    added to git and every cross-section the scheduled scan produced died with
    the runner -- scan_day and scan_run recorded sessions whose scores did not
    exist anywhere. Silent, and only visible as a back-test that mysteriously
    ignores recent history.
    """
    src = src or LOG_DIR
    day_csv = src / "scan_day.csv"
    if not day_csv.exists():
        return []
    dates = set(pd.read_csv(day_csv)["as_of_date"].astype(str))
    have = {f.name.removesuffix(".csv.gz") for f in (src / "scores").glob("*.csv.gz")} \
        if (src / "scores").exists() else set()
    return sorted(dates - have)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["export", "import", "status", "verify"])
    args = ap.parse_args()

    if args.action == "export":
        for t, n in export_log().items():
            where = 'data/log/scores/*.csv.gz' if t == 'scan_score' else f'data/log/{t}.csv'
            print(f"  {t:18s} {n:>8,} rows -> {where}")
    elif args.action == "import":
        res = import_log()
        if not res:
            print("  nothing to import (data/log/ absent or empty)")
        for t, n in res.items():
            print(f"  {t:18s} {n:>8,} new rows imported")
    elif args.action == "verify":
        missing = verify()
        if missing:
            print(f"  {len(missing)} logged session(s) have NO score partition:")
            for d in missing[:10]:
                print(f"    {d}")
            print("  the cross-section for those sessions is not in git")
            return 1
        print("  every logged session has a score partition")
    else:
        print(f"  db: {DB_PATH}  exists={DB_PATH.exists()}")
        with _conn() as c:
            for t in KEYS:
                try:
                    n = pd.read_sql_query(f"SELECT COUNT(*) n FROM {t}", c).n[0]
                except Exception:                                # noqa: BLE001
                    n = "—"
                f = LOG_DIR / f"{t}.csv"
                m = f"{sum(1 for _ in open(f)) - 1:,}" if f.exists() else "—"
                print(f"  {t:18s} db={n:>10}   csv={m:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
