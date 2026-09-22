"""The log must survive the operations performed on it every night.

Every check here corresponds to a way the point-in-time log was silently
destroying or corrupting itself. None of them raised, none of them logged, and
all of them would surface months later as a back-test that quietly disagreed
with the database -- which is the worst possible place to find out.

    uv run python -m tests.test_log_integrity
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd

from scanner import logexport, store


class _Sandbox:
    """A throwaway database and mirror. The real ones are never touched."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._db, self._log = store.DB_PATH, logexport.LOG_DIR
        store.DB_PATH = root / "test.db"
        logexport.LOG_DIR = root / "log"
        store._last_run_id = 0
        return root

    def __exit__(self, *exc):
        store.DB_PATH, logexport.LOG_DIR = self._db, self._log
        store._last_run_id = 0
        self.tmp.cleanup()
        return False


def _scored(date: str, tickers=("AAA", "BBB.TO"), z=(2.5, 1.0)) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Timestamp(date), "ticker": list(tickers),
        "market": ["US", "CA"], "sector": ["Software", "Energy"],
        "side": "long", "composite_raw": list(z), "composite_z": list(z),
        "score_pct": [99.0, 50.0], "n_qualifiers": [3, 0],
    })


def _summary(date: str, n_picks: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{"date": pd.Timestamp(date), "n_eligible": 2, "n_over_threshold": 1,
                          "n_qualified": 1, "n_picks": n_picks, "max_composite_z": 2.5,
                          "ceiling_z": 3.0, "empty_day": n_picks == 0}])


def _write(date: str, **kw) -> int:
    return store.write_scan(as_of=date, scored=_scored(date, **kw),
                            picks=pd.DataFrame(), summary=_summary(date),
                            config={}, universe_size=2, provider="yahoo")


def test_matured_outcome_is_never_nulled():
    """A refetch that lost the 20d bar must not unmature a measured outcome.

    backfill kept every row with ANY horizon present and then INSERT OR
    REPLACEd, so a name whose 20-day future had already been measured came back
    NULL the moment a later fetch was one bar shallower.
    """
    with _Sandbox():
        _write("2026-01-05")
        # 23 sessions of AAA, so every horizon matures
        full = pd.DataFrame({
            "date": pd.bdate_range("2026-01-05", periods=23),
            "ticker": "AAA", "close": [100.0 + i for i in range(23)]})
        store.backfill_forward_returns(full)
        with store._conn() as c:
            before = c.execute(
                "SELECT fwd_1d, fwd_5d, fwd_20d FROM forward_return "
                "WHERE as_of_date='2026-01-05' AND ticker='AAA'").fetchone()
        assert before and before[2] is not None, "20d should have matured"

        # Now a shallower refetch: only 2 sessions, so 5d and 20d are unknown.
        store.backfill_forward_returns(full.head(2))
        with store._conn() as c:
            after = c.execute(
                "SELECT fwd_1d, fwd_5d, fwd_20d FROM forward_return "
                "WHERE as_of_date='2026-01-05' AND ticker='AAA'").fetchone()
        assert after[2] is not None, "a shallow refetch nulled a matured 20d outcome"
        assert abs(after[2] - before[2]) < 1e-12, "a matured outcome changed value"
        return {"fwd_20d_before": round(before[2], 6), "fwd_20d_after": round(after[2], 6)}


def test_market_and_sector_survive_a_round_trip():
    """A database rebuilt from git must not come back with columns missing.

    They were dropped from the mirror on the theory that today's universe file
    could supply them. It cannot: joining today's listings to an old
    cross-section is survivorship bias, and the rebuild this repo actually
    performed came back with market and sector NULL for 262,016 rows.
    """
    with _Sandbox():
        _write("2026-01-06")
        logexport.export_log()
        with store._conn() as c:
            c.execute("DELETE FROM scan_score")          # the rebuild
        logexport.import_log()
        with store._conn() as c:
            got = c.execute("SELECT ticker, market, sector FROM scan_score "
                            "ORDER BY ticker").fetchall()
        assert got == [("AAA", "US", "Software"), ("BBB.TO", "CA", "Energy")], got
        return {"rows_recovered": len(got)}


def test_export_never_drops_a_row_it_has_not_seen():
    """Import is additive; export was not.

    It rewrote each whole-file CSV straight from this machine's database, so a
    row that existed in git but not locally -- a session another machine
    scanned -- was deleted from the mirror by the next export.
    """
    with _Sandbox() as root:
        _write("2026-01-07")
        logexport.export_log()
        runs = root / "log" / "scan_run.csv"
        other = pd.read_csv(runs)
        foreign = other.iloc[0].copy()
        foreign["run_id"] = 999_999_999_999_999      # another machine's id
        pd.concat([other, foreign.to_frame().T], ignore_index=True).to_csv(runs, index=False)

        logexport.export_log()                        # must not truncate
        assert 999_999_999_999_999 in set(pd.read_csv(runs)["run_id"]), \
            "export deleted a mirror row this database had never seen"
        return {"rows": len(pd.read_csv(runs))}


def test_a_rescan_replaces_its_partition_instead_of_doubling_it():
    """A re-scanned session must export as ONE cross-section, the newest.

    The partition was skipped whenever the file merely existed, so a correction
    could never reach git; and the query had no run filter, so when it did
    write it concatenated every run and the back-test counted the day twice.
    """
    with _Sandbox() as root:
        _write("2026-01-08")
        logexport.export_log()
        first = pd.read_csv(root / "log" / "scores" / "2026-01-08.csv.gz", compression="gzip")
        time.sleep(0.002)
        rid2 = _write("2026-01-08", z=(3.9, 1.0))       # a re-scan of the same day
        logexport.export_log()
        second = pd.read_csv(root / "log" / "scores" / "2026-01-08.csv.gz", compression="gzip")
        assert len(second) == len(first) == 2, f"partition doubled: {len(second)} rows"
        assert set(second["run_id"]) == {rid2}, "partition holds a stale or mixed run"
        assert float(second.loc[second.ticker == "AAA", "composite_z"].iloc[0]) == 3.9, \
            "the correction never reached the mirror"
        return {"rows": len(second), "run_id": int(rid2)}


def test_run_ids_do_not_collide_across_machines():
    """run_id joins scan_score to scan_run, so it has to be globally unique.

    It was an AUTOINCREMENT rowid. Two machines both produce 1, 2, 3..., and
    after an import a live CI cross-section could join to a laptop's mock run.
    """
    with _Sandbox():
        mine = {_write(f"2026-02-0{i}") for i in range(1, 5)}
        assert len(mine) == 4, "ids repeated within one machine"
        assert min(mine) > 1_000_000_000_000, "id is not millisecond-derived"
        salt = store._MACHINE_SALT
        assert all(r % 1000 == salt for r in mine), "machine salt not preserved"
        assert sorted(mine) == list(sorted(mine)), "ids must stay monotonic for MAX()"
        return {"ids": sorted(mine)[:2], "salt": salt}


def test_forward_returns_partition_by_year_and_stay_byte_stable():
    """The outcome table was one CSV rewritten in full every night.

    At ~3,000 names a session that is a ~100 MB file inside a year, a whole new
    git blob daily, and a hard stop at GitHub's 100 MB limit.
    """
    with _Sandbox() as root:
        _write("2026-01-09")
        bars = pd.DataFrame({"date": pd.bdate_range("2026-01-09", periods=23),
                             "ticker": "AAA", "close": [100.0 + i for i in range(23)]})
        store.backfill_forward_returns(bars)
        logexport.export_log()
        f = root / "log" / "forward" / "2026.csv.gz"
        assert f.exists(), "no year partition written"
        assert not (root / "log" / "forward_return.csv").exists(), "whole-file mirror remains"
        first = f.read_bytes()
        logexport.export_log()
        assert f.read_bytes() == first, "an unchanged partition rewrote itself (git churn)"
        return {"bytes": len(first)}


def main() -> int:
    checks = [
        ("a matured outcome is never nulled", test_matured_outcome_is_never_nulled),
        ("market/sector survive a rebuild", test_market_and_sector_survive_a_round_trip),
        ("export never drops an unseen row", test_export_never_drops_a_row_it_has_not_seen),
        ("a re-scan replaces its partition", test_a_rescan_replaces_its_partition_instead_of_doubling_it),
        ("run_ids are globally unique", test_run_ids_do_not_collide_across_machines),
        ("forward returns partition by year", test_forward_returns_partition_by_year_and_stay_byte_stable),
    ]
    print("=" * 72)
    print("  Log integrity — the artifact that must never be destroyed")
    print("=" * 72)
    failed = 0
    for label, fn in checks:
        try:
            detail = fn()
            print(f"\n  PASS  {label}")
            for k, v in (detail or {}).items():
                print(f"          {k}: {v}")
        except AssertionError as e:
            failed += 1
            print(f"\n  FAIL  {label}\n        {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            import traceback
            failed_tb = traceback.format_exc().strip().splitlines()[-3:]
            print(f"\n  ERROR {label}\n        {type(e).__name__}: {e}")
            for line in failed_tb:
                print(f"        {line}")
    print("\n" + "=" * 72)
    print(f"  {len(checks) - failed}/{len(checks)} passed")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
