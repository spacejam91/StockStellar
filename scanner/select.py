"""Step 5 — selection, and the per-session audit that includes empty days.

Threshold first, then rank, then cap. Never rank-then-slice: top-3-by-rank
returns 3 names by construction, which would make "unusual" meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanner.combine import drivers
from scanner.config import ScanConfig

PICK_FIELDS = ["date", "rank", "ticker", "market", "sector", "side", "composite_z",
               "score_pct", "n_qualifiers", "close", "atr14", "rvol", "ret_z",
               "range_ratio", "gap_atr", "drivers", "flags"]


def candidates(scored: pd.DataFrame, cfg: ScanConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ScanConfig()
    if scored.empty:
        return scored
    c = scored[scored["composite_z"] >= cfg.composite_threshold]
    c = c[c["n_qualifiers"] >= cfg.min_qualifiers]
    if cfg.require_confirmation and "pillar_confirmation" in c.columns:
        # A heavy up-day that closed on its low is not a long-side candidate,
        # however well it scored on volume and range.
        c = c[c["pillar_confirmation"].fillna(0.0) >= 0.0]
    return c.sort_values(["date", "composite_z"], ascending=[True, False])


def select(scored: pd.DataFrame, cfg: ScanConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ScanConfig()
    cand = candidates(scored, cfg)
    if cand.empty:
        return pd.DataFrame(columns=PICK_FIELDS)

    picks = []
    for dt, day in cand.groupby("date", sort=True):
        used: dict[str, int] = {}
        for _, row in day.iterrows():
            if len(used) and sum(used.values()) >= cfg.max_picks:
                break
            sector = row.get("sector")
            key = str(sector) if pd.notna(sector) else "UNKNOWN"
            # On a sector-wide move day an uncapped list returns three copies of
            # one bet, formatted to look like three independent signals.
            if cfg.one_per_sector and used.get(key, 0) >= 1:
                continue
            used[key] = used.get(key, 0) + 1
            flags = []
            if bool(row.get("suspect_unadjusted_split", False)):
                flags.append("suspect_unadjusted_split")
            picks.append({
                "date": dt, "rank": sum(used.values()), "ticker": row["ticker"],
                "market": row["market"], "sector": sector, "side": row["side"],
                "composite_z": round(float(row["composite_z"]), 3),
                "score_pct": float(row["score_pct"]),
                "n_qualifiers": int(row["n_qualifiers"]),
                "close": float(row["close"]), "atr14": _f(row.get("atr14")),
                "rvol": _f(row.get("rvol")), "ret_z": _f(row.get("ret_z")),
                "range_ratio": _f(row.get("range_ratio")), "gap_atr": _f(row.get("gap_atr")),
                "drivers": drivers(row), "flags": ",".join(flags),
            })
    return pd.DataFrame(picks, columns=PICK_FIELDS)


def day_summary(scored: pd.DataFrame, picks: pd.DataFrame, cfg: ScanConfig | None = None) -> pd.DataFrame:
    """One row per scanned session, including the ones that produced nothing.

    Empty days are the product, not a failure. Track the share: below
    cfg.empty_day_target the threshold is too loose and the score has stopped
    meaning "unusual".
    """
    cfg = cfg or ScanConfig()
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for dt, day in scored.groupby("date", sort=True):
        over = day["composite_z"] >= cfg.composite_threshold
        qual = over & (day["n_qualifiers"] >= cfg.min_qualifiers)
        n_picks = int((picks["date"] == dt).sum()) if len(picks) else 0
        rows.append({
            "date": dt,
            "n_eligible": int(len(day)),
            "n_over_threshold": int(over.sum()),
            "n_qualified": int(qual.sum()),
            "n_picks": n_picks,
            "max_composite_z": _r(day["composite_z"].max()),
            "ceiling_z": _r(day["ceiling_z"].max() if "ceiling_z" in day else np.nan),
            "empty_day": n_picks == 0,
        })
    return pd.DataFrame(rows)


def health(summary: pd.DataFrame, cfg: ScanConfig | None = None) -> dict:
    cfg = cfg or ScanConfig()
    if summary.empty:
        return {"sessions": 0}
    share = float(summary["empty_day"].mean())
    return {
        "sessions": int(len(summary)),
        "empty_day_share": round(share, 3),
        "empty_day_target": cfg.empty_day_target,
        "threshold_too_loose": bool(share < cfg.empty_day_target),
        "threshold_unreachable": bool((summary["ceiling_z"] < cfg.composite_threshold).any()),
        "total_picks": int(summary["n_picks"].sum()),
    }


def _f(v):
    return None if v is None or pd.isna(v) else round(float(v), 4)


def _r(v):
    return None if v is None or pd.isna(v) else round(float(v), 3)
