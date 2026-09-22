"""An empty day must always be producible, including the degenerate ones.

Returning nothing is this project's most important output. It is also the one
most easily broken, because every path that produces it is a path nobody looks
at when things are working.

The bug this file exists to catch was real: with zero eligible names,
gates.scaled_thresholds() returns an empty frame, and add_qualifiers() then
called .drop(columns=["n_eligible"]) on it and raised KeyError. So a session
where every name was gated out CRASHED the scan instead of reporting an empty
day -- the precise inverse of the requirement.

It surfaced while truncating history to reproduce an unrelated CI discrepancy,
which is to say: by accident. Hence a test.

    uv run python -m tests.test_empty_day
"""

from __future__ import annotations

import pandas as pd

from app.market_data import MockMarketData
from scanner import gates, metrics
from scanner.config import ScanConfig
from scanner.pipeline import run_scan


def _bars(n_tickers: int = 20, n_sessions: int = 40) -> pd.DataFrame:
    return MockMarketData(n_tickers=n_tickers, n_sessions=n_sessions).daily_bars()


def test_zero_eligible_does_not_crash():
    """Far too little history for the 250-session gate: nobody is eligible."""
    for n_sessions in (20, 40, 60, 100):
        bars = _bars(n_sessions=n_sessions)
        g = gates.apply(metrics.compute(bars), ScanConfig())
        assert int(g["eligible"].sum()) == 0, (
            f"{n_sessions} sessions should gate everyone out (min_sessions=250)")
        assert "n_qualifiers" in g.columns, "n_qualifiers must exist even with nobody eligible"
        assert int(g["n_qualifiers"].max()) == 0
    return {"session_counts_tested": [20, 40, 60, 100]}


def test_mock_provider_survives_short_histories():
    """MockMarketData planted events after a 60-session warm-up and called
    rng.choice on an empty range when asked for <= 60 sessions."""
    for n in (10, 40, 60, 61, 120):
        b = MockMarketData(n_tickers=5, n_sessions=n).daily_bars()
        assert len(b) == 5 * n, f"expected {5*n} rows for n_sessions={n}, got {len(b)}"
    return {"session_counts_tested": [10, 40, 60, 61, 120]}


def test_scan_reports_an_empty_day_rather_than_failing():
    """End to end: a universe nobody can qualify in yields a clean empty result."""
    res = run_scan(provider=MockMarketData(n_tickers=20, n_sessions=40), persist=False)
    assert res.is_empty_day, "expected an empty day"
    assert res.today().empty
    return {"as_of": str(res.as_of.date()), "picks": 0}


def test_impossible_threshold_still_returns_cleanly():
    """A threshold above the attainable ceiling returns nothing forever.

    That is a configuration error, not a market condition, and it must not be
    mistaken for a quiet day -- but it must not crash either.
    """
    cfg = ScanConfig()
    cfg.composite_threshold = 99.0          # unreachable at any universe size
    res = run_scan(provider=MockMarketData(n_tickers=60, n_sessions=400),
                   cfg=cfg, persist=False)
    assert res.is_empty_day
    return {"threshold": cfg.composite_threshold, "picks": len(res.today())}


def main() -> int:
    checks = [
        ("zero eligible does not crash", test_zero_eligible_does_not_crash),
        ("mock survives short histories", test_mock_provider_survives_short_histories),
        ("scan reports an empty day", test_scan_reports_an_empty_day_rather_than_failing),
        ("impossible threshold returns cleanly", test_impossible_threshold_still_returns_cleanly),
    ]
    print("=" * 70)
    print("  Empty days — the output that must never break")
    print("=" * 70)
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
            print(f"\n  ERROR {label}\n        {type(e).__name__}: {e}")
    print("\n" + "=" * 70)
    print(f"  {len(checks) - failed}/{len(checks)} passed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
