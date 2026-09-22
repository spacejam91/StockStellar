"""Selection must compare strength, not group size — and the page must say so.

Each check here corresponds to a way the scanner produced a defensible-looking
list for an indefensible reason. They are cheap, deterministic, and run before
anything is published.

    uv run python -m tests.test_selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from scanner import gates, select
from scanner.config import ScanConfig
from scanner.mathx import max_attainable_z

DATE = pd.Timestamp("2026-06-01")


def _scored(n_us: int, n_ca: int) -> pd.DataFrame:
    """A cross-section whose z-scores are the honest rank transform per market."""
    rows = []
    for market, n in (("US", n_us), ("CA", n_ca)):
        z = np.linspace(-max_attainable_z(n), max_attainable_z(n), n)
        for i in range(n):
            rows.append({
                "date": DATE, "ticker": f"{market}{i:04d}", "market": market,
                "sector": f"S{i % 7}", "side": "long",
                "composite_z": float(z[i]), "score_pct": 100.0 * (i + 0.5) / n,
                "n_qualifiers": 3, "close": 20.0, "atr14": 1.0, "rvol": 4.0,
                "ret_z": 3.0, "range_ratio": 2.5, "gap_atr": 1.2,
                "pillar_confirmation": 1.0,
                "ceiling_z": max_attainable_z(n),
            })
    return pd.DataFrame(rows)


def test_a_small_market_is_not_crowded_out_by_a_large_one():
    """composite_z's ceiling is set by group size, so one merged sort is rigged.

    max_attainable_z(2784) = 3.57 and max_attainable_z(295) = 2.93, so every US
    name above 2.93 outranks every CA name that can possibly exist. Measured on
    the real log, Canada cleared the threshold on 85 of 88 sessions, supplied
    140 of 1,965 candidates, and won 9 of 258 slots.
    """
    cfg = ScanConfig()
    picks = select.select(_scored(2784, 295), cfg)
    by_market = picks["market"].value_counts().to_dict()
    assert by_market.get("CA", 0) == cfg.max_picks, (
        f"CA got {by_market.get('CA', 0)} of its own {cfg.max_picks} slots: {by_market}")
    assert by_market.get("US", 0) == cfg.max_picks, by_market
    return {"ceiling_US": round(max_attainable_z(2784), 2),
            "ceiling_CA": round(max_attainable_z(295), 2), "picks": by_market}


def test_rank_is_reported_against_the_names_it_was_ranked_against():
    """"#1 of 1076" for a name that competed against 301 is off by a factor of 3."""
    picks = select.select(_scored(2784, 295), ScanConfig())
    ca = picks[picks["market"] == "CA"]
    us = picks[picks["market"] == "US"]
    assert set(ca["n_market"]) == {295}, f"CA denominator wrong: {set(ca['n_market'])}"
    assert set(us["n_market"]) == {2784}, f"US denominator wrong: {set(us['n_market'])}"
    return {"CA": 295, "US": 2784}


def test_the_applied_threshold_travels_with_the_pick():
    """A scaled scan's real bar is a trailing quantile, not the config number.

    Without carrying it, the page drew every meter against the absolute level
    and told readers a qualifier had failed when it had cleared that session's
    actual bar.
    """
    s = _scored(400, 200)
    for col, val in (("thr_rvol", 2.1), ("thr_ret_z", 2.2),
                     ("thr_range_ratio", 1.8), ("thr_gap_atr", 0.9)):
        s[col] = val
    picks = select.select(s, ScanConfig())
    assert not picks.empty
    assert set(picks["thr_rvol"]) == {2.1}, "the applied RVOL bar was not carried"
    assert picks[list(select.THR_FIELDS)].notna().all().all(), "a threshold went missing"
    return {t: float(picks[t].iloc[0]) for t in select.THR_FIELDS}


def test_a_broken_atr_denominator_is_gated_out():
    """gap_atr, range_ratio and ma_dist_atr all divide by ATR.

    LB.TO was published with ATR14 of $0.08 on a $40.66 close, so a 17-cent gap
    read as +2.13 ATR and took a top slot. Nothing happened to that stock.
    """
    cfg = ScanConfig()
    df = pd.DataFrame({
        "date": DATE, "ticker": ["QUIET", "NORMAL", "CORRUPT"], "market": "US",
        "close": [40.66, 40.0, 5.0], "atr14": [0.08, 1.6, 4.0],
        "dv_med20": 5e7, "session_n": 400, "gap_days": 1.0,
        "ret_std60": 0.02, "vol_med20": 1e6,
        "rvol": 1.0, "ret_z": 0.5, "range_ratio": 1.0, "gap_atr": 0.1,
    })
    g = gates.apply(df, cfg)
    passed = dict(zip(g["ticker"], g["gate_volatility"]))
    assert passed["NORMAL"], "a 4% ATR is an ordinary stock"
    assert not passed["QUIET"], "an ATR of 0.2% of price makes every ATR ratio noise"
    assert not passed["CORRUPT"], "an ATR of 80% of price is a bad bar, not a market"
    return {t: round(float(a / c), 4) for t, a, c
            in zip(df["ticker"], df["atr14"], df["close"])}


def test_the_empty_day_floor_is_measured_per_market():
    """Two independent lists go empty together far less often than either alone.

    The >=20% floor is a statement about ONE list. Judging the combined figure
    after selection went per market would read an unchanged bar as loosened.
    """
    summary = pd.DataFrame({"date": pd.bdate_range("2026-06-01", periods=10),
                            "n_picks": [1] * 10, "empty_day": [False] * 10,
                            "ceiling_z": 3.5})
    # US produces on every session; CA on two of ten.
    picks = pd.DataFrame(
        [{"date": d, "market": "US"} for d in summary["date"]] +
        [{"date": d, "market": "CA"} for d in summary["date"][:2]])
    h = select.health(summary, picks, ScanConfig())
    assert h["empty_day_share"] == 0.0, "no session was empty overall"
    assert h["empty_day_share_by_market"]["CA"] == 0.8
    assert h["empty_day_share_by_market"]["US"] == 0.0
    assert h["threshold_too_loose"], "US never goes empty; that is the bar being too loose"
    return h["empty_day_share_by_market"]


def test_meter_bars_vary_with_their_data():
    """The fill scaled to the value's own magnitude, so every bar was identical.

    width = 100 * v / max(floor, v * 1.15) = 87.0% for any v above the floor.
    1,061 of 1,315 published bars -- 80.7% -- were pixel identical.
    """
    env = Environment(loader=FileSystemLoader("app/templates"),
                      autoescape=select_autoescape(["html"]))
    meter = env.get_template("_macros.html").module.meter
    widths = []
    for v in (0.4, 1.0, 1.9, 2.6, 4.0, 9.0):
        html = str(meter("RVOL", v, 2.0, "%.1f", "x"))
        widths.append(html.split('style="width: ')[1].split('%')[0])
    assert len(set(widths)) >= 5, f"bars do not vary with the data: {widths}"
    assert widths[-1] == "100.0", "a value past the scale must clamp, not overflow"
    assert "over" in str(meter("RVOL", 9.0, 2.0, "%.1f", "x")), "a clamped bar must be marked"
    # The tick is at a fixed place, which is what makes two rows comparable.
    ticks = {str(meter("x", v, 2.0, "%.1f", "")).split('left: ')[1].split('%')[0]
             for v in (0.4, 4.0)}
    assert len(ticks) == 1, f"the threshold tick moved between rows: {ticks}"
    return {"widths": widths, "tick": ticks.pop()}


def main() -> int:
    checks = [
        ("a small market keeps its slots", test_a_small_market_is_not_crowded_out_by_a_large_one),
        ("rank denominator is the market", test_rank_is_reported_against_the_names_it_was_ranked_against),
        ("applied thresholds travel", test_the_applied_threshold_travels_with_the_pick),
        ("a broken ATR is gated out", test_a_broken_atr_denominator_is_gated_out),
        ("empty-day floor is per market", test_the_empty_day_floor_is_measured_per_market),
        ("meter bars vary with data", test_meter_bars_vary_with_their_data),
    ]
    print("=" * 72)
    print("  Selection & display — comparing strength, not group size")
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
            print(f"\n  ERROR {label}\n        {type(e).__name__}: {e}")
    print("\n" + "=" * 72)
    print(f"  {len(checks) - failed}/{len(checks)} passed")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
