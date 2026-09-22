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
# Only the columns something actually reads are kept: the nine pillar_* columns
# and composite_raw are reconstructible from a rescan and are not worth carrying
# in git forever.
#
# market and sector ARE carried, despite being properties of the ticker. They
# were dropped on the theory that data/universe_*.csv could supply them -- but
# that file is TODAY's listing, and joining it to a two-year-old cross-section
# is survivorship bias by construction: a delisted name simply has no row, and
# a name that changed sector gets today's. A database rebuilt from git lost
# both columns for all history, which is the one thing this mirror exists to
# prevent. They compress to almost nothing.
SCORE_EXPORT_COLS = ["run_id", "ticker", "market", "sector", "side",
                     "composite_z", "score_pct", "n_qualifiers"]

# table -> the columns that identify a row uniquely
KEYS = {
    "scan_run": ["run_id"],
    "scan_day": ["run_id", "as_of_date"],
    "scan_score": ["run_id", "as_of_date", "ticker"],
    "scan_pick": ["run_id", "as_of_date", "ticker"],
    "forward_return": ["as_of_date", "ticker"],
}

# Tables written as partitioned directories rather than one whole-file CSV.
PARTITIONED = {"scan_score": "scores", "forward_return": "forward"}

# Deterministic bytes: gzip stamps mtime into its header by default, so an
# unchanged table would produce a different file every night and show up as a
# git change. With mtime=0 an unchanged partition is byte-identical and is
# skipped, which is what keeps the nightly diff to the sessions that moved.
_GZ = {"method": "gzip", "mtime": 0}


def _write_if_changed(df: pd.DataFrame, path: Path) -> bool:
    """Write a gzipped partition only when its contents differ. True if written."""
    import io
    buf = io.BytesIO()
    df.to_csv(buf, index=False, compression=_GZ)
    data = buf.getvalue()
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def _union_with_disk(df: pd.DataFrame, path: Path, keys: list[str],
                     read=pd.read_csv) -> pd.DataFrame:
    """DB rows plus any row already on disk that the DB does not have.

    Import is additive; export was not. It rewrote each whole-file CSV straight
    from the database, so a row that existed in git but not in THIS machine's
    database -- a session another machine scanned, on a laptop that skipped the
    import -- was silently dropped from the mirror by the next export. It cost
    a real scan_run row minutes after this module's own docstring was quoted
    back about never destroying the log.

    So export unions instead of truncating. The database wins on conflict; disk
    rows it has never seen survive.
    """
    if not path.exists():
        return df
    try:
        disk = read(path)
    except Exception as e:                                       # noqa: BLE001
        log.warning("could not read %s (%s) -- exporting database rows only", path, e)
        return df
    if disk.empty:
        return df
    k = [c for c in keys if c in df.columns and c in disk.columns]
    if not k:
        return df
    both = pd.concat([df, disk], ignore_index=True)
    return both.drop_duplicates(subset=k, keep="first")


def _partition_run_id(path: Path) -> int | None:
    """The run_id a score partition was written from, or None if unreadable."""
    try:
        head = pd.read_csv(path, compression="gzip", nrows=1)
    except Exception:                                            # noqa: BLE001
        return None
    if head.empty or "run_id" not in head.columns:
        return None
    return int(head["run_id"].iloc[0])


def export_log(out: Path | None = None, rebuild: bool = False) -> dict[str, int]:
    """Write the log to text. Sorted, so diffs stay small and readable.

    `rebuild` re-writes every score partition from the database instead of
    skipping the ones already current. Needed exactly once, when a column is
    added to SCORE_EXPORT_COLS: a partition is otherwise written once and never
    revisited, so history would keep the old column set forever while the
    database held the full one.
    """
    out = out or LOG_DIR
    out.mkdir(parents=True, exist_ok=True)
    for sub in PARTITIONED.values():
        (out / sub).mkdir(exist_ok=True)
    counts = {}
    with _conn() as c:
        # scan_score: one gzipped file per session, holding the NEWEST run only.
        #
        # Two bugs lived here. The export selected every row for a date with no
        # run filter, so a re-scanned session exported both runs concatenated
        # and the back-test counted that cross-section twice. And the file was
        # skipped whenever it merely existed, so a re-scan -- the thing you do
        # after fixing a bug -- could never reach git at all: the stale partition
        # sat there permanently while the database held the correction.
        #
        # Newest-run-only matches how scores_frame() reads the table back, so
        # the CSV mirror and the database now answer the same question.
        try:
            latest = pd.read_sql_query(
                "SELECT as_of_date, MAX(run_id) AS rid FROM scan_score GROUP BY as_of_date", c)
            n = 0
            for d, rid in latest.itertuples(index=False):
                f = out / "scores" / f"{d}.csv.gz"
                if not rebuild and f.exists() and _partition_run_id(f) == int(rid):
                    continue
                df = pd.read_sql_query(
                    f"SELECT {','.join(SCORE_EXPORT_COLS)} FROM scan_score "
                    f"WHERE as_of_date = ? AND run_id = ? ORDER BY ticker",
                    c, params=(d, int(rid)))
                if _write_if_changed(df, f):
                    n += len(df)
            counts["scan_score"] = n
        except (pd.errors.DatabaseError, sqlite3.Error):
            pass

        # forward_return: partitioned BY YEAR, not one growing file.
        #
        # It was a single CSV rewritten in full every night. At ~3,000 eligible
        # names a session it gains ~100 MB a year, and git stores a whole new
        # blob per rewrite -- so the repository grew by the entire file daily
        # and the file itself would cross GitHub's 100 MB hard limit inside two
        # years, at which point the push fails and the log stops. Outcomes
        # mature ~20 sessions after their date, so a past year's partition
        # becomes immutable and is then written once, forever.
        try:
            fr = pd.read_sql_query("SELECT * FROM forward_return", c)
            n = 0
            if not fr.empty:
                fr["_year"] = fr["as_of_date"].astype(str).str.slice(0, 4)
                for year, part in fr.groupby("_year", sort=True):
                    f = out / "forward" / f"{year}.csv.gz"
                    part = _union_with_disk(
                        part.drop(columns=["_year"]), f, KEYS["forward_return"],
                        read=lambda q: pd.read_csv(q, compression="gzip"))
                    part = part.sort_values(["as_of_date", "ticker"])
                    if _write_if_changed(part, f):
                        n += len(part)
            counts["forward_return"] = n
            # The whole-file predecessor, removed only once the partitions
            # demonstrably hold at least as many rows. Not "once a partition
            # directory exists": the first version of this check tested a
            # glob generator for truth -- always True -- and deleted a 23 MB
            # file against an EMPTY partition set, on a machine whose database
            # happened to hold no outcomes at all. Caught by a backup taken a
            # minute earlier, which is not a control.
            legacy = out / "forward_return.csv"
            if legacy.exists():
                have = sum(len(pd.read_csv(f, compression="gzip"))
                           for f in (out / "forward").glob("*.csv.gz"))
                with legacy.open() as fh:
                    was = max(sum(1 for _ in fh) - 1, 0)
                if have >= was:
                    legacy.unlink()
                else:
                    log.warning("keeping data/log/forward_return.csv: %d rows on disk vs "
                                "%d in the year partitions -- import it first", was, have)
        except (pd.errors.DatabaseError, sqlite3.Error):
            pass

        for table, keys in KEYS.items():
            if table in PARTITIONED:
                continue
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", c)
            except (pd.errors.DatabaseError, sqlite3.Error):
                continue
            path = out / f"{table}.csv"
            df = _union_with_disk(df, path, keys)
            if df.empty:
                counts[table] = 0
                continue
            df = df.sort_values([k for k in keys if k in df.columns])
            df.to_csv(path, index=False)
            counts[table] = len(df)
    return counts


def import_log(src: Path | None = None) -> dict[str, int]:
    """Load CSV rows the DB does not already have. Never updates or deletes.

    Additive on purpose: an import must be able to run on any machine, in any
    order, without rewriting a score that is already logged.

    forward_return is the one exception, and only in the direction that cannot
    lose anything: a NULL horizon is filled from the CSV, a value already
    present is left alone. Without that, a 20-day outcome that matured on CI
    could never reach a laptop whose row was written while it was still NULL --
    strict additivity would see the key already present and skip it forever.
    """
    src = src or LOG_DIR
    if not src.exists():
        return {}
    added = {}
    with _conn() as c:
        for table, keys in KEYS.items():
            incoming = _read_source(src, table)
            if incoming is None or incoming.empty:
                if incoming is not None:
                    added[table] = 0
                continue

            if table == "forward_return":
                added[table] = _merge_forward(c, incoming)
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


def _read_source(src: Path, table: str) -> pd.DataFrame | None:
    """Rows on disk for one table, whether whole-file or partitioned."""
    sub = PARTITIONED.get(table)
    if sub is None:
        f = src / f"{table}.csv"
        return pd.read_csv(f) if f.exists() else None

    parts = []
    d = src / sub
    for f in sorted(d.glob("*.csv.gz")) if d.exists() else []:
        part = pd.read_csv(f, compression="gzip")
        if table == "scan_score":
            # The date is the filename, not a column -- that is what lets a
            # session's partition be written once and never touched again.
            part["as_of_date"] = f.name.removesuffix(".csv.gz")
        parts.append(part)
    # The whole-file predecessor of a now-partitioned table. Read it too, so a
    # checkout made before the migration still imports cleanly.
    legacy = src / f"{table}.csv"
    if legacy.exists():
        parts.append(pd.read_csv(legacy))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _merge_forward(c: sqlite3.Connection, incoming: pd.DataFrame) -> int:
    """Insert outcomes, filling NULL horizons but never overwriting a value."""
    horizons = [h for h in ("fwd_1d", "fwd_5d", "fwd_20d") if h in incoming.columns]
    if not horizons:
        return 0
    cols = ["as_of_date", "ticker", *horizons] + (
        ["filled_at"] if "filled_at" in incoming.columns else [])
    rows = incoming[cols].where(pd.notna(incoming[cols]), None)
    sets = ", ".join(f"{h} = COALESCE(forward_return.{h}, excluded.{h})" for h in horizons)
    c.executemany(
        f"""INSERT INTO forward_return ({','.join(cols)})
            VALUES ({','.join('?' * len(cols))})
            ON CONFLICT(as_of_date, ticker) DO UPDATE SET {sets}""",
        rows.itertuples(index=False, name=None))
    return len(rows)


def verify(src: Path | None = None) -> dict[str, list[str]]:
    """What the mirror is missing or has stale, before the commit that ships it.

    Two failures, both silent, both only visible much later as a back-test that
    mysteriously ignores recent history:

    `missing` -- a session in scan_day with no score partition on disk. The CI
    commit step staged only data/log/*.csv, so data/log/scores/ was never added
    to git and every cross-section the scheduled scan produced died with the
    runner.

    `stale` -- a partition written from an OLDER run than the one scan_day
    records for that session. That is a re-scan whose correction never reached
    git, so the database and the mirror disagree about the same day.
    """
    src = src or LOG_DIR
    day_csv = src / "scan_day.csv"
    if not day_csv.exists():
        return {"missing": [], "stale": []}
    day = pd.read_csv(day_csv)
    day["as_of_date"] = day["as_of_date"].astype(str)

    scores = src / "scores"
    have = {f.name.removesuffix(".csv.gz"): f for f in scores.glob("*.csv.gz")} \
        if scores.exists() else {}
    missing = sorted(set(day["as_of_date"]) - set(have))

    # Staleness is measured against scan_score, not scan_day. A scan_day row can
    # legitimately be newer than the scores it summarises -- that is the shape
    # of the sessions CI logged before its commit step was fixed, whose
    # cross-sections were destroyed with the runner and can never be recovered.
    # Comparing to scan_day would flag those forever and train the check to be
    # ignored, which is how a real re-scan would then slip past.
    try:
        with _conn() as c:
            newest = pd.read_sql_query(
                "SELECT as_of_date, MAX(run_id) AS rid FROM scan_score GROUP BY as_of_date", c)
    except (pd.errors.DatabaseError, sqlite3.Error):
        return {"missing": missing, "stale": []}
    stale = sorted(str(d) for d, rid in newest.itertuples(index=False)
                   if str(d) in have and (_partition_run_id(have[str(d)]) or rid) != int(rid))
    return {"missing": missing, "stale": stale}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["export", "import", "status", "verify", "repair"])
    ap.add_argument("--rebuild", action="store_true",
                    help="re-write every score partition (after a column change)")
    args = ap.parse_args()

    if args.action == "export":
        for t, n in export_log(rebuild=args.rebuild).items():
            sub = PARTITIONED.get(t)
            where = f'data/log/{sub}/*.csv.gz' if sub else f'data/log/{t}.csv'
            print(f"  {t:18s} {n:>8,} rows -> {where}")
    elif args.action == "import":
        res = import_log()
        if not res:
            print("  nothing to import (data/log/ absent or empty)")
        for t, n in res.items():
            print(f"  {t:18s} {n:>8,} new rows imported")
    elif args.action == "repair":
        from scanner.store import backfill_market
        n = backfill_market()
        print(f"  scan_score.market recovered from the ticker suffix on {n:,} rows")
        print("  run `export --rebuild` to carry it into the mirror")
    elif args.action == "verify":
        res = verify()
        bad = False
        if res["missing"]:
            bad = True
            print(f"  {len(res['missing'])} logged session(s) have NO score partition:")
            for d in res["missing"][:10]:
                print(f"    {d}")
            print("  the cross-section for those sessions is not in git")
        if res["stale"]:
            bad = True
            print(f"  {len(res['stale'])} partition(s) are older than the logged run:")
            for d in res["stale"][:10]:
                print(f"    {d}")
            print("  a re-scan corrected these sessions and the correction is not in git")
        if bad:
            return 1
        print("  every logged session has a current score partition")
    else:
        print(f"  db: {DB_PATH}  exists={DB_PATH.exists()}")
        with _conn() as c:
            for t in KEYS:
                try:
                    n = pd.read_sql_query(f"SELECT COUNT(*) n FROM {t}", c).n[0]
                except Exception:                                # noqa: BLE001
                    n = "—"
                sub = PARTITIONED.get(t)
                if sub:
                    d = LOG_DIR / sub
                    m = f"{len(list(d.glob('*.csv.gz')))} files" if d.exists() else "—"
                else:
                    f = LOG_DIR / f"{t}.csv"
                    m = f"{sum(1 for _ in open(f)) - 1:,}" if f.exists() else "—"
                print(f"  {t:18s} db={n:>10}   csv={m:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
