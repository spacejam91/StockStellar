"""Step 4 — pillars, sides, composite, and the re-standardisation that makes
composite_z mean what it says.

The non-obvious part is `_restandardise`. A weighted mean of k pillars has sd
about 1/sqrt(k), so thresholding the raw mean "in sigma" is wrong by a factor of
sqrt(k): with 9 pillars a nominal 2.0 threshold is really about 6 sigma. In
testing that produced 3 picks in 150 sessions before this step existed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scanner.config import FLIPPING_PILLARS, INPUTS, NEUTRAL_PILLARS, ScanConfig
from scanner.mathx import max_attainable_z, rank_to_z

log = logging.getLogger(__name__)

GROUP_KEYS = ["date", "market"]

# Fundamental pillars are optional -- members are supplied by whatever
# fundamentals source is wired up, and the pillar drops out when absent.
FUNDAMENTAL_MEMBERS: dict[str, tuple[str, ...]] = {
    "quality": ("gross_margin", "gross_margin_delta_yoy", "fcf_margin", "roic",
                "net_debt_ebitda", "interest_coverage"),
    "value": ("pe_forward", "ev_ebitda", "ev_sales", "pb", "fcf_yield"),
    "revisions": ("eps_rev_1m", "eps_rev_3m", "target_px_rev_1m", "net_upgrades_1m",
                  "last_surprise_pct"),
}

ALL_PILLARS = tuple(dict.fromkeys(list(NEUTRAL_PILLARS) + list(FLIPPING_PILLARS)))


def _members(pillar: str) -> list[str]:
    tech = [f"n_{c}" for c, (p, _) in INPUTS.items() if p == pillar]
    fund = [f"n_{c}" for c in FUNDAMENTAL_MEMBERS.get(pillar, ())]
    return tech + fund


def pillar_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each pillar's available normalised members.

    Missing members are skipped rather than treated as zero: a name with no
    revisions data should not be scored as "averagely revised".
    """
    out = df.copy()
    for pillar in ALL_PILLARS:
        cols = [c for c in _members(pillar) if c in out.columns]
        out[f"pillar_{pillar}"] = out[cols].mean(axis=1, skipna=True) if cols else np.nan
    return out


def _weighted(out: pd.DataFrame, cfg: ScanConfig, pillars) -> tuple[pd.Series, pd.Series]:
    num = pd.Series(0.0, index=out.index)
    den = pd.Series(0.0, index=out.index)
    for pillar in pillars:
        col = f"pillar_{pillar}"
        if col not in out.columns:
            continue
        w = float(cfg.weight_for(pillar))
        if w == 0.0:
            continue
        vals = out[col]
        ok = vals.notna()
        num = num.add((vals.fillna(0.0) * w).where(ok, 0.0))
        # Weight only counts where the pillar exists, so weights renormalise
        # instead of a missing pillar dragging the composite toward zero.
        den = den.add(pd.Series(np.where(ok, w, 0.0), index=out.index))
    return num, den


def _restandardise(raw: pd.Series, keys: list[pd.Series]) -> pd.Series:
    return raw.groupby(keys, sort=False).transform(lambda s: rank_to_z(s.to_numpy()))


def combine(df: pd.DataFrame, cfg: ScanConfig | None = None) -> pd.DataFrame:
    """Produce side, composite_raw, composite_z, score_pct and ceiling_z.

    Only eligible rows are scored -- ineligible names must not influence anyone
    else's rank.
    """
    cfg = cfg or ScanConfig()
    out = pillar_scores(df[df["eligible"]].copy()) if "eligible" in df.columns else pillar_scores(df.copy())
    if out.empty:
        log.warning("nothing eligible to score")
        return out

    num_n, den_n = _weighted(out, cfg, NEUTRAL_PILLARS)
    num_f, den_f = _weighted(out, cfg, FLIPPING_PILLARS)
    den = (den_n + den_f).replace(0, np.nan)

    raw_long = (num_n + num_f) / den
    raw_short = (num_n - num_f) / den

    take_long = raw_long >= raw_short
    out["side"] = np.where(take_long, "long", "short")
    out["composite_raw"] = raw_long.where(take_long, raw_short)

    # Pillar columns are re-oriented to the chosen side so the `drivers` string
    # reads in the direction the name was actually flagged in.
    flip = np.where(take_long, 1.0, -1.0)
    for pillar in FLIPPING_PILLARS:
        col = f"pillar_{pillar}"
        if col in out.columns:
            out[col] = out[col] * flip

    keys = [out[k] for k in GROUP_KEYS]
    out["composite_z"] = _restandardise(out["composite_raw"], keys)
    out["score_pct"] = (out["composite_z"].groupby(keys, sort=False)
                           .rank(pct=True) * 100).round(1)

    # A threshold above the ceiling returns nothing forever and looks exactly
    # like a quiet market, so carry the ceiling next to the score.
    out["ceiling_z"] = (out.groupby(GROUP_KEYS, sort=False)["composite_raw"]
                           .transform(lambda s: max_attainable_z(int(s.notna().sum()))))
    if (out["ceiling_z"] < cfg.composite_threshold).any():
        worst = float(out["ceiling_z"].min())
        log.warning(
            "composite_threshold %.2f exceeds the attainable ceiling %.2f for at least one "
            "(date, market) group -- those groups can never produce a pick. Widen the "
            "universe or lower the threshold.", cfg.composite_threshold, worst)
    return out


def drivers(row: pd.Series, k: int = 3) -> str:
    """The k pillars contributing most, signed, for the human reading the list."""
    vals = {c[len("pillar_"):]: row[c] for c in row.index
            if c.startswith("pillar_") and pd.notna(row[c])}
    if not vals:
        return ""
    top = sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:k]
    return ", ".join(f"{name} {val:+.1f}z" for name, val in top)
