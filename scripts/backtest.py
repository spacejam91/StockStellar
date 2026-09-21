"""Back-test the composite against the point-in-time log.

    uv run python scripts/backtest.py [--horizon 5]

Runs the tests from docs/PARAMETERS.md that decide whether the score ranks
anything at all. Every number comes from rows written once, as they looked that
session, joined to outcomes that happened afterwards -- never recomputed with
today's universe, which would be survivorship plus lookahead and would
manufacture an edge that is not there.

How to read the output, because the headline numbers mislead on their own:

  IC            Real cross-sectional equity signals sit at 0.02-0.05. Anything
                near 0.30 is a lookahead bug, not a discovery.
  |t|           The bar is 3.0, not 2.0 (Harvey, Liu & Zhu 2016) -- 2.0 is too
                weak given how many predictors get tested. AND overlapping
                forward windows autocorrelate the IC series, understating its
                standard error, so |t| here is INFLATED. Treat it as indicative
                and add a Newey-West correction before quoting it anywhere.
  hit rate      Meaningless alone. In an up year everything hits. Only the
                edge over the same sessions' universe base rate counts.
  cost drag     35bps round trip is the default. On a daily-turnover 3-name
                list this has historically eaten most of the gross.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.market_data import SYNTHETIC_PROVIDERS  # noqa: E402
from scanner.store import DB_PATH  # noqa: E402

COST_BPS = 35.0


def load(horizon: int) -> pd.DataFrame:
    """Scores joined to outcomes, synthetic providers excluded."""
    marks = ",".join("?" * len(SYNTHETIC_PROVIDERS))
    q = f"""
        SELECT s.as_of_date, s.ticker, s.market, s.sector, s.side,
               s.composite_z, s.score_pct, s.n_qualifiers,
               f.fwd_{horizon}d AS fwd, r.provider
        FROM scan_score s
        JOIN scan_run r ON r.run_id = s.run_id
        JOIN forward_return f
          ON f.as_of_date = s.as_of_date AND f.ticker = s.ticker
        WHERE f.fwd_{horizon}d IS NOT NULL
          AND r.provider NOT IN ({marks})
    """
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(q, c, params=sorted(SYNTHETIC_PROVIDERS))
    # Signed in the direction the scanner actually called.
    df["edge"] = np.where(df["side"] == "short", -df["fwd"], df["fwd"])
    df["signed_z"] = df["composite_z"] * np.where(df["side"] == "short", -1.0, 1.0)
    return df


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 10:
        return np.nan
    ra, rb = a[ok].rank(), b[ok].rank()
    if not ra.std() or not rb.std():
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


def test_ic(df: pd.DataFrame, col: str, label: str) -> None:
    ics = (df.groupby("as_of_date")
             .apply(lambda g: _spearman(g[col], g["fwd"]), include_groups=False)
             .dropna())
    n = len(ics)
    if n < 2:
        print(f"    {label:14s} not enough sessions ({n})")
        return
    mean, sd = ics.mean(), ics.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if sd else np.nan
    verdict = ("looks like a lookahead bug" if abs(mean) > 0.15 else
               "within the range of a real signal" if abs(mean) >= 0.02 else
               "indistinguishable from noise")
    print(f"    {label:14s} IC {mean:+.4f}  |t| {abs(t):5.2f}  n={n:>4} sessions   {verdict}")


def test_deciles(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["signed_z", "fwd"]).copy()
    d["decile"] = (d.groupby("as_of_date")["signed_z"]
                    .transform(lambda s: pd.qcut(s.rank(method="first"), 10,
                                                 labels=False, duplicates="drop")))
    tbl = (d.groupby("decile")
             .agg(n=("edge", "size"), mean_edge=("edge", "mean"),
                  up=("edge", lambda s: float((s > 0).mean())))
             .reset_index())
    mono = _spearman(tbl["decile"].astype(float), tbl["mean_edge"])
    print(f"    decile monotonicity (want > 0.6): {mono:+.2f}")
    for _, r in tbl.iterrows():
        bar = "#" * max(0, min(40, int(abs(r.mean_edge) * 4000)))
        print(f"      d{int(r.decile)}  n={int(r.n):>6}  mean {r.mean_edge:+.4f}  "
              f"up {r.up:5.1%}  {bar}")


def test_edge_vs_base(df: pd.DataFrame, cfg) -> None:
    sel = df[(df["composite_z"] >= cfg.composite_threshold)
             & (df["n_qualifiers"] >= cfg.min_qualifiers)]
    if len(sel) < 30:
        print(f"    selected rows: {len(sel)} — too few to measure")
        return
    dates = set(sel["as_of_date"])
    uni = df[df["as_of_date"].isin(dates)]
    s_up, u_up = float((sel["edge"] > 0).mean()), float((uni["edge"] > 0).mean())
    se = np.sqrt(s_up * (1 - s_up) / len(sel)) * 100
    gross = sel["edge"].mean() * 100
    net = gross - COST_BPS / 100.0
    print(f"    selected      n={len(sel):>6}  up {s_up:6.2%}   mean {gross:+.3f}%")
    print(f"    universe      n={len(uni):>6}  up {u_up:6.2%}   (same sessions)")
    print(f"    EDGE          {100*(s_up-u_up):+.2f} pp  +/- {se:.2f} pp  "
          f"({'inside' if abs(s_up-u_up)*100 < 2*se else 'OUTSIDE'} 2 SE)")
    print(f"    cost drag     gross {gross:+.3f}%  ->  net {net:+.3f}% "
          f"after {COST_BPS:.0f}bps round trip"
          + ("   << costs eat it entirely" if gross > 0 and net <= 0 else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    from scanner.config import ScanConfig
    cfg = ScanConfig()

    df = load(args.horizon)
    if df.empty:
        print("no real-provider rows with forward returns yet — run the scanner daily")
        return 0

    print("=" * 72)
    print(f"  BACK-TEST — {args.horizon}d horizon")
    print("=" * 72)
    print(f"  rows {len(df):,}   sessions {df.as_of_date.nunique()}   "
          f"names {df.ticker.nunique():,}   providers {sorted(df.provider.unique())}")
    print(f"  span {df.as_of_date.min()} .. {df.as_of_date.max()}")

    print("\n  1-2. INFORMATION COEFFICIENT")
    test_ic(df, "composite_z", "composite_z")
    test_ic(df, "signed_z", "signed_z")
    print("       composite_z is magnitude-only (raw = max(long, short)), so its IC")
    print("       against signed returns is ~0 by construction. signed_z is the")
    print("       directional one and the row that matters.")

    print("\n  4. DECILE MONOTONICITY")
    test_deciles(df)

    print("\n  3 + 6. EDGE OVER BASE RATE, AND COST DRAG")
    test_edge_vs_base(df, cfg)

    print("\n" + "=" * 72)
    print("  |t| is inflated here: overlapping forward windows autocorrelate the IC")
    print("  series and understate its standard error. Add Newey-West before")
    print("  quoting a t-stat. Nothing above is a forecast.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
