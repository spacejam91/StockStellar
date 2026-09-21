"""CLI:  uv run python -m scanner [options]

    # today's watchlist from the offline mock provider
    uv run python -m scanner

    # real bars, and build 120 sessions of calibration log
    MARKET_DATA=yahoo uv run python -m scanner --history-days 120 --backfill

    # what the log says so far
    uv run python -m scanner --report
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from app.market_data import get_provider
from scanner import run_scan
from scanner.config import ScanConfig
from scanner.pipeline import format_watchlist
from scanner import gates as gates_mod
from scanner import rank as rank_mod
from scanner import store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scanner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=("mock", "yahoo"), help="override MARKET_DATA")
    ap.add_argument("--as-of", help="session to score (YYYY-MM-DD, default: latest)")
    ap.add_argument("--history-days", type=int, default=0, help="also score the N prior sessions")
    ap.add_argument("--threshold", type=float, help="override composite_z floor")
    ap.add_argument("--min-qualifiers", type=int, help="override absolute qualifiers required")
    ap.add_argument("--no-persist", action="store_true", help="do not write to the log")
    ap.add_argument("--backfill", action="store_true", help="fill forward returns after scanning")
    ap.add_argument("--report", action="store_true", help="print measurements from the log")
    ap.add_argument("--diagnose", action="store_true", help="gate rejections + input coverage")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 140)

    cfg = ScanConfig()
    if args.threshold is not None:
        cfg.composite_threshold = args.threshold
    if args.min_qualifiers is not None:
        cfg.min_qualifiers = args.min_qualifiers

    provider = get_provider(args.provider)
    result = run_scan(provider=provider, cfg=cfg, as_of=args.as_of,
                      history_days=args.history_days, persist=not args.no_persist)

    print(format_watchlist(result))

    h = result.health
    if h.get("sessions", 0) > 1:
        print(f"\nempty days over the scored window: {100 * h['empty_day_share']:.0f}% "
              f"(target >= {100 * h['empty_day_target']:.0f}%)"
              + ("  <-- threshold too loose" if h["threshold_too_loose"] else ""))
    if h.get("threshold_unreachable"):
        print("WARNING: composite_threshold exceeds the attainable ceiling for at least one "
              "session/market -- those groups can never produce a pick.")

    if args.diagnose:
        df = result.scored
        print("\n--- input coverage, latest session (null % by market) ---")
        cov = rank_mod.coverage(df)
        if not cov.empty:
            print(cov[cov["null_pct"] > 0].to_string(index=False) or "  (no nulls)")
        print("\n--- pillar availability ---")
        pillars = [c for c in df.columns if c.startswith("pillar_")]
        print(df[pillars].notna().mean().mul(100).round(1).to_string())

    if args.backfill:
        bars = provider.daily_bars()
        bars["date"] = pd.to_datetime(bars["date"])
        n = store.backfill_forward_returns(bars)
        print(f"\nforward returns filled/updated: {n:,} rows")

    if args.report:
        hist = store.day_history(limit=200)
        if hist:
            empty = sum(1 for r in hist if r["empty_day"]) / len(hist)
            print(f"\n--- log: {len(hist)} sessions, {100 * empty:.0f}% empty ---")
        for pb in store.provider_breakdown():
            print(f"  provider {pb['provider']}: {pb['sessions']} sessions "
                  f"({pb['first_date']} -> {pb['last_date']})")
        # mock rows are a random walk; including them would dilute a real
        # measurement toward 50% with a deceptively tight standard error
        br = store.score_band_base_rates(horizon=5, include_mock=(provider.name == "mock"))
        if br.empty:
            print("no outcomes logged yet -- run with --history-days and --backfill first")
        else:
            print("\n--- what followed each score band, 5d (measured, not predicted) ---")
            print(br.to_string(index=False))
            print("\nRead the SE column first: a band with few observations is not a finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
