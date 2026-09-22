"""Step 5 — selection, and the per-session audit that includes empty days.

Threshold first, then rank, then cap. Never rank-then-slice: top-3-by-rank
returns 3 names by construction, which would make "unusual" meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanner.combine import drivers
from scanner.config import ScanConfig

# thr_* are the levels this session ACTUALLY applied. In scaled mode they are
# not the config's absolute numbers -- they are a trailing quantile that moves
# with the market -- and without carrying them a reader is shown a meter drawn
# against a limit the scan never used. The page was telling people "RVOL >= 3.0
# not met" for names that had cleared that session's real bar of 2.1.
PICK_FIELDS = ["date", "rank", "ticker", "market", "sector", "side", "composite_z",
               "score_pct", "n_qualifiers", "close", "atr14", "rvol", "ret_z",
               "range_ratio", "gap_atr", "thr_rvol", "thr_ret_z", "thr_range_ratio",
               "thr_gap_atr", "ceiling_z", "n_market", "drivers", "flags"]

THR_FIELDS = ("thr_rvol", "thr_ret_z", "thr_range_ratio", "thr_gap_atr")


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

    # How many names this pick was actually ranked against. `scored` holds only
    # eligible rows, and ranking is per (date, market) -- so the session-wide
    # eligible count is the wrong denominator for "#1 of N", by a factor of ten
    # for a Canadian name. The page was printing "#1 of 1076 eligible" for a
    # name that competed against 301.
    n_market = scored.groupby(["date", "market"], observed=True).size()

    picks = []
    # Per (date, MARKET), not per date.
    #
    # composite_z is a rank transform, so its attainable ceiling is set by group
    # size: max_attainable_z(2,784 US names) = 3.57, max_attainable_z(295 CA
    # names) = 2.93. Merging both into one sorted list therefore does not compare
    # strength, it compares how many names a country lists. Measured over the
    # 88 sessions logged here, CA cleared the threshold on 85 of them and
    # supplied 140 of 1,965 candidates -- and won 9 of 258 slots. The Canadian
    # half of the universe was being scanned and then discarded by arithmetic.
    #
    # So each market selects against its own candidates, with its own sector cap
    # and its own max_picks. CA can be empty while US is not, which is a truer
    # statement than either a merged list or a forced quota.
    for (dt, _mkt), day in cand.groupby(["date", "market"], sort=True, observed=True):
        used: dict[str, int] = {}
        for _, row in day.iterrows():
            if len(used) and sum(used.values()) >= cfg.max_picks:
                break
            sector = row.get("sector")
            known_sector = pd.notna(sector) and str(sector).strip() != ""
            # On a sector-wide move day an uncapped list returns three copies of
            # one bet, formatted to look like three independent signals.
            #
            # But the cap can only apply where the sector is KNOWN. Bucketing
            # every unclassified name under one "UNKNOWN" key treats "we have no
            # sector data" as "these are all the same sector", which silently
            # caps the whole list at one name. That is exactly what happened to
            # US names: nasdaqtraded.txt publishes no sector, so every US row
            # collided in one bucket and max_picks=3 could never return more
            # than 1. An unknown sector is an absence of evidence, not evidence
            # of sameness -- so those rows are not capped against each other.
            if known_sector:
                key = str(sector)
                if cfg.one_per_sector and used.get(key, 0) >= 1:
                    continue
                used[key] = used.get(key, 0) + 1
            else:
                # Counted toward max_picks, never toward a sector bucket.
                used[f"__unknown_{len(used)}"] = 1
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
                **{t: _f(row.get(t)) for t in THR_FIELDS},
                "ceiling_z": _f(row.get("ceiling_z")),
                "n_market": int(n_market.get((dt, row["market"]), 0)),
                "drivers": drivers(row), "flags": ",".join(flags),
            })
    out = pd.DataFrame(picks, columns=PICK_FIELDS)
    if out.empty:
        return out
    # Strongest first within a session, across both markets, for display. The
    # SELECTION above was per market; this only decides reading order.
    return out.sort_values(["date", "composite_z"], ascending=[True, False]) \
              .reset_index(drop=True)


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


def health(summary: pd.DataFrame, picks: pd.DataFrame | None = None,
           cfg: ScanConfig | None = None) -> dict:
    """Is the bar still tight enough that "unusual" means something?

    The >=20% empty-day floor is a statement about ONE list: below it, the
    threshold admits something every session and has stopped discriminating.
    Selection now runs per market, so there are two lists, and the combined
    figure is not the same quantity -- two independent lists go empty together
    far less often than either goes empty alone, so a combined 7.9% and a
    per-market 21.3% / 38.2% describe the same, unchanged bar. Judge each
    market against the floor; the combined number is reported but is not the
    test. (Measured over 89 sessions: US 21.3%, CA 38.2%.)
    """
    cfg = cfg or ScanConfig()
    if summary.empty:
        return {"sessions": 0}
    share = float(summary["empty_day"].mean())
    per_market: dict[str, float] = {}
    if picks is not None and len(picks) and "market" in picks.columns:
        dates = set(summary["date"])
        for m, grp in picks.groupby("market", observed=True):
            per_market[str(m)] = round(1 - len(set(grp["date"]) & dates) / max(len(dates), 1), 3)
    loose = ([v < cfg.empty_day_target for v in per_market.values()] if per_market
             else [share < cfg.empty_day_target])
    return {
        "sessions": int(len(summary)),
        "empty_day_share": round(share, 3),
        "empty_day_share_by_market": per_market,
        "empty_day_target": cfg.empty_day_target,
        "threshold_too_loose": bool(any(loose)),
        "threshold_unreachable": bool((summary["ceiling_z"] < cfg.composite_threshold).any()),
        "total_picks": int(summary["n_picks"].sum()),
    }


def _f(v):
    return None if v is None or pd.isna(v) else round(float(v), 4)


def _r(v):
    return None if v is None or pd.isna(v) else round(float(v), 3)
