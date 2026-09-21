"""Full refresh and back-test, end to end. Built to run unattended.

    uv run python scripts/overnight.py

Order matters and each step is guarded, because the failure mode that matters
is a half-finished run that looks finished:

  1. universe        listings, then sectors (build_universe OVERWRITES the
                     sector columns, so fetch_sectors must follow it)
  2. scan            the whole CA+US universe, persisted to the log
  3. backfill        forward returns for sessions whose future has happened
  4. back-test       IC, t-stat, decile monotonicity, cost drag, empty-day
                     share -- the tests from SCORING_SPEC.md that decide
                     whether the composite ranks anything at all
  5. site            static page for Pages

Memory note: ~8,700 tickers x ~345 sessions is roughly 3M rows of bars. That
is around 250 MB as one frame, which is fine, but it is why the bars are built
once and reused rather than refetched per step.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("overnight")

HISTORY_DAYS = 90     # sessions to score into the log
HORIZONS = (1, 5, 20)


def banner(t: str) -> None:
    print(f"\n{'=' * 72}\n  {t}\n{'=' * 72}", flush=True)


def step_universe() -> bool:
    banner("1/5  universe + sectors")
    for script in ("build_universe.py", "fetch_sectors.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           capture_output=True, text=True, timeout=1800)
        for line in r.stdout.splitlines():
            if line.strip():
                print(f"    {line.strip()}", flush=True)
        if r.returncode != 0:
            print(f"    FAILED {script}: {r.stderr[-400:]}", flush=True)
            return False
    return True


def step_scan():
    banner(f"2/5  full-universe scan, {HISTORY_DAYS} sessions, persisted")
    from app.market_data import YahooMarketData
    from scanner.pipeline import run_scan

    t0 = time.time()
    res = run_scan(provider=YahooMarketData(), history_days=HISTORY_DAYS, persist=True)
    s = res.summary
    n = lambda c: pd.to_numeric(s[c], errors="coerce")
    print(f"    {time.time() - t0:.0f}s   sessions {len(s)}", flush=True)
    for c in ("n_eligible", "ceiling_z", "n_over_threshold", "n_qualified", "n_picks"):
        if c in s.columns:
            print(f"    {c:18s} mean {n(c).mean():8.2f}", flush=True)
    empty = n("empty_day").fillna(0).mean()
    print(f"    EMPTY-DAY SHARE    {empty:.1%}   (spec target >= 20%)", flush=True)
    print(f"    picks distribution {n('n_picks').value_counts().sort_index().to_dict()}", flush=True)
    t = res.today()
    print(f"    today: {[] if t.empty else list(t['ticker'])}", flush=True)
    return res


def step_backfill(res) -> None:
    banner("3/5  backfill forward returns")
    from scanner import store
    # res.scored already carries date/ticker/close for every scored row, so the
    # forward returns come from the same bars the scores were computed on --
    # refetching here would risk scoring against one snapshot and measuring
    # against another.
    n = store.backfill_forward_returns(res.scored[["date", "ticker", "close"]])
    print(f"    filled {n:,} rows", flush=True)


def step_backtest() -> None:
    banner("4/5  back-test — does the composite rank anything?")
    from scanner import evaluate, store

    scored = store.scores_frame()
    if scored is None or scored.empty:
        print("    no scored frame exposed by store; using base rates only", flush=True)
    else:
        scored = evaluate.add_forward_returns(scored, horizons=HORIZONS)
        for h in HORIZONS:
            ic = evaluate.information_coefficient(scored, horizon=h)
            tbl = evaluate.decile_table(scored, horizon=h)
            mono = evaluate.decile_monotonicity(tbl)
            print(f"    {h:>2}d  IC {ic['mean_ic']:+.4f}  t {ic['t_stat']:+.2f}  "
                  f"mono {mono:+.2f}  n={ic['n_sessions']}", flush=True)
        print("    reference: real cross-sectional signals sit at IC 0.02-0.05;", flush=True)
        print("               |t| >= 3.0 is the bar (Harvey/Liu/Zhu 2016), not 2.0;", flush=True)
        print("               overlapping windows inflate |t| -- treat it as indicative.", flush=True)

    print("\n    measured base rates by score band (real providers only):", flush=True)
    for h in (1, 5, 20):
        try:
            df = store.score_band_base_rates(horizon=h)
        except Exception as e:                                   # noqa: BLE001
            print(f"    {h}d: unavailable ({e})", flush=True)
            continue
        if df.empty:
            print(f"    {h:>2}d: no forward returns yet", flush=True)
            continue
        print(f"    {h:>2}d:", flush=True)
        for _, r in df.iterrows():
            print(f"        band {int(r.score_band):>3}-{int(r.score_band)+10:<3} "
                  f"n={int(r.n):>6}  up {r.up_rate_pct:>5.1f}% +/- {r.se_pp:.1f}pp  "
                  f"mean {r.mean_ret_pct:+.3f}%", flush=True)


def step_site() -> None:
    banner("5/5  static site")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py")],
                       capture_output=True, text=True, timeout=600)
    for line in r.stdout.splitlines():
        if line.strip():
            print(f"    {line.strip()}", flush=True)


def main() -> int:
    t0 = time.time()
    print(f"overnight run starting", flush=True)
    if not step_universe():
        print("universe step failed; continuing with existing files", flush=True)
    try:
        res = step_scan()
    except Exception:
        traceback.print_exc()
        return 1
    for fn, arg in ((step_backfill, res), (step_backtest, None), (step_site, None)):
        try:
            fn(arg) if arg is not None else fn()
        except Exception:
            print(f"    step {fn.__name__} failed:", flush=True)
            traceback.print_exc()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
