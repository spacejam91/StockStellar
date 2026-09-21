"""Tests 1, 2 and 4 — does the composite rank anything at all?

Every function here needs a point-in-time log: one row per name per session,
written once and never edited. Recomputing history with today's universe,
today's listings or restated fundamentals is survivorship plus lookahead bias,
and it will manufacture an edge that is not there.

Calibration numbers to keep in mind when reading output:
  - Real cross-sectional equity signals sit at IC 0.02-0.05. A 0.30 means a
    lookahead bug, not a discovery — check that forward returns start the
    session AFTER the score.
  - |t| >= 3.0, not 2.0 (Harvey, Liu & Zhu 2016). 2.0 is too weak a bar given
    how many predictors get tested. Raise it further for every variant tried,
    and count the variants honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_returns(scored: pd.DataFrame, horizons=(1, 5, 20)) -> pd.DataFrame:
    """Forward returns starting the session AFTER the score. The shift is the
    whole ballgame — off by one here and the IC looks spectacular and is a lie."""
    out = scored.sort_values(["ticker", "date"], kind="mergesort").copy()
    g = out.groupby("ticker", sort=False)["close"]
    for h in horizons:
        out[f"fwd_{h}d"] = g.transform(lambda s, h=h: s.shift(-h) / s - 1.0)
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 10:
        return np.nan
    ra, rb = a[ok].rank(), b[ok].rank()
    sa, sb = ra.std(), rb.std()
    if not sa or not sb or np.isnan(sa) or np.isnan(sb):
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def information_coefficient(scored: pd.DataFrame, horizon: int = 5,
                            score_col: str = "composite_z") -> dict:
    """Test 1 + Test 2. Spearman per session over the full eligible cross-section."""
    fwd = f"fwd_{horizon}d"
    if fwd not in scored.columns:
        scored = add_forward_returns(scored, horizons=(horizon,))
    elig = scored[scored.get("eligible", True)]

    ics = (
        elig.groupby("date", sort=True)
        .apply(lambda g: _spearman(g[score_col], g[fwd]), include_groups=False)
        .dropna()
    )
    n = len(ics)
    mean = float(ics.mean()) if n else np.nan
    sd = float(ics.std(ddof=1)) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd else np.nan
    return {"horizon": horizon, "n_sessions": n, "mean_ic": mean, "sd_ic": sd,
            "t_stat": float(t) if t == t else np.nan, "ic_series": ics}


def decile_table(scored: pd.DataFrame, horizon: int = 5,
                 score_col: str = "composite_z") -> pd.DataFrame:
    """Test 4. Mean forward return by score decile; want rank correlation > 0.6.

    A strong top bucket with a flat middle is usually curve-fit, not signal.
    """
    fwd = f"fwd_{horizon}d"
    if fwd not in scored.columns:
        scored = add_forward_returns(scored, horizons=(horizon,))
    elig = scored[scored.get("eligible", True)].dropna(subset=[score_col, fwd]).copy()
    if elig.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_fwd", "up_rate"])

    elig["decile"] = (
        elig.groupby("date", sort=False)[score_col]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    )
    tbl = (
        elig.groupby("decile", sort=True)
        .agg(n=(fwd, "size"), mean_fwd=(fwd, "mean"), up_rate=(fwd, lambda s: float((s > 0).mean())))
        .reset_index()
    )
    return tbl


def decile_monotonicity(tbl: pd.DataFrame) -> float:
    """Spearman between decile index and its mean forward return."""
    if len(tbl) < 3:
        return np.nan
    return _spearman(tbl["decile"].astype(float), tbl["mean_fwd"])


def empty_day_share(summary: pd.DataFrame) -> float:
    """Test 9. Want >= 0.20. Below that the threshold is too loose and the score
    has stopped meaning 'unusual' — tighten, never loosen."""
    return float(summary["empty_day"].mean()) if len(summary) else np.nan
