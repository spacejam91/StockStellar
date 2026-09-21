"""Step 1 — hard gates, plus the absolute qualifier count.

Two different jobs live here and they must not be confused:

`eligible` is *membership*. It decides who gets ranked at all, and it is kept out
of the score entirely — otherwise illiquid junk ranks as "unusual" because nobody
trades it.

`n_qualifiers` is the *absolute* half of selection, in own-history units. It is
what makes an empty day possible: composite_z is a re-ranked cross-section, so
~2.3% of any universe clears 2 sigma every single session, and on 10k names that
is 230 candidates with a top-3 list that always fills. A quiet day simply does
not produce names with RVOL 2 *and* a 2-sigma move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanner.config import ScanConfig

GATES = ("liquidity", "price", "history", "continuity", "completeness")


def apply(df: pd.DataFrame, cfg: ScanConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ScanConfig()
    out = df.copy()

    min_dv = out["market"].map(lambda m: cfg.min_dollar_vol(m)).astype(float)
    out["gate_liquidity"] = out["dv_med20"] >= min_dv
    out["gate_price"] = out["close"] >= cfg.min_price
    out["gate_history"] = out["session_n"] >= cfg.min_sessions
    # First bar of a ticker has no previous bar, so no gap to measure -- that is
    # not a continuity failure, it is the start of the series.
    out["gate_continuity"] = out["gap_days"].isna() | (out["gap_days"] <= cfg.max_gap_days)
    out["gate_completeness"] = out[["atr14", "ret_std60", "vol_med20"]].notna().all(axis=1)

    gate_cols = [f"gate_{g}" for g in GATES]
    out[gate_cols] = out[gate_cols].fillna(False)
    out["eligible"] = out[gate_cols].all(axis=1)

    reason = pd.Series("", index=out.index, dtype=object)
    for g in GATES:
        reason = reason.where(out[f"gate_{g}"], reason + g + ",")
    out["gate_fail_reason"] = reason.str.rstrip(",")

    # Absolute, own-history qualifiers. NaN is not a fired condition.
    fired = (
        (out["rvol"] >= cfg.qual_rvol).fillna(False).astype(int)
        + (out["ret_z"].abs() >= cfg.qual_ret_z).fillna(False).astype(int)
        + (out["range_ratio"] >= cfg.qual_range_ratio).fillna(False).astype(int)
        + (out["gap_atr"].abs() >= cfg.qual_gap_atr).fillna(False).astype(int)
    )
    out["n_qualifiers"] = fired.astype(np.int8)
    return out


def gate_report(df: pd.DataFrame) -> pd.DataFrame:
    """How many names each gate is rejecting, on the latest session present.

    Worth looking at before blaming the criteria for a quiet list: one
    mis-specified gate can silently remove most of the universe.
    """
    if df.empty:
        return pd.DataFrame()
    last = df[df["date"] == df["date"].max()]
    rows = [{"gate": g, "passed": int(last[f"gate_{g}"].sum()),
             "failed": int((~last[f"gate_{g}"]).sum())} for g in GATES]
    rows.append({"gate": "ALL (eligible)", "passed": int(last["eligible"].sum()),
                 "failed": int((~last["eligible"]).sum())})
    return pd.DataFrame(rows)
