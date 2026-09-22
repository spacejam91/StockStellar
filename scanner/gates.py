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

GATES = ("liquidity", "price", "history", "continuity", "completeness", "volatility")


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
    # Keep the ATR denominator interpretable. See ScanConfig.min_atr_frac: three
    # scored inputs divide by ATR, so a near-zero one manufactures a signal from
    # noise and an absurdly large one is corrupt data.
    atr_frac = (out["atr14"] / out["close"].where(out["close"] > 0)).replace(
        [np.inf, -np.inf], np.nan)
    out["atr_frac"] = atr_frac
    out["gate_volatility"] = atr_frac.between(cfg.min_atr_frac, cfg.max_atr_frac)

    gate_cols = [f"gate_{g}" for g in GATES]
    out[gate_cols] = out[gate_cols].fillna(False)
    out["eligible"] = out[gate_cols].all(axis=1)

    reason = pd.Series("", index=out.index, dtype=object)
    for g in GATES:
        reason = reason.where(out[f"gate_{g}"], reason + g + ",")
    out["gate_fail_reason"] = reason.str.rstrip(",")

    out = add_qualifiers(out, cfg)
    return out


# metric -> (column, use |value|, the fixed fallback level on cfg)
QUALIFIERS = (
    ("rvol", "rvol", False, "qual_rvol"),
    ("ret_z", "ret_z", True, "qual_ret_z"),
    ("range_ratio", "range_ratio", False, "qual_range_ratio"),
    ("gap_atr", "gap_atr", True, "qual_gap_atr"),
)


def scaled_thresholds(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    """Per-session, PER-MARKET qualifier levels that hold the expected count steady.

    For each session: look back `qualifier_lookback` sessions of ELIGIBLE rows
    in the same market, and take the quantile of each metric that leaves
    `target_qualifiers` names expected to clear it out of today's eligible count.

    The window ends at the PREVIOUS session. Including today would make the bar
    a percentile of today's own names, which always admits the same proportion
    and can never produce an empty day -- the exact failure this replaces.

    Per market, because every other stage of this pipeline is: ranking, the
    composite, the ceiling and the liquidity gate all treat CA and US as
    separate regimes on the grounds that they are different liquidity worlds.
    Pooling them here quietly undid that. US supplies ~2,780 of ~3,080 eligible
    names, so a pooled top-0.2% tail is a US tail wearing both names, and the
    Canadian half was measured against a bar its own distribution never set.
    """
    rows = []
    for market, mdf in df.groupby("market", sort=True, observed=True):
        dates = sorted(mdf.loc[mdf["eligible"], "date"].unique())
        for i, d in enumerate(dates):
            lo = max(0, i - cfg.qualifier_lookback)
            past = mdf[mdf["eligible"] & mdf["date"].isin(dates[lo:i])] if i else None
            n_today = int((mdf["eligible"] & (mdf["date"] == d)).sum())
            rec = {"date": d, "market": market, "n_eligible": n_today}
            # q chosen so ~target_qualifiers names are expected above it.
            q = 1.0 - (cfg.target_qualifiers / max(n_today, 1))
            q = min(max(q, 0.5), 0.99999)
            for name, col, absolute, fallback in QUALIFIERS:
                fixed = float(getattr(cfg, fallback))
                if past is None or past.empty:
                    rec[name] = fixed
                    continue
                series = past[col].abs() if absolute else past[col]
                v = float(series.quantile(q)) if series.notna().any() else fixed
                # Bounded on BOTH sides, against the spec's absolute level.
                #
                # The floor was always here: a stretch of dead sessions would
                # otherwise drag the bar down to noise. The ceiling was not, and
                # it is the same failure in the other direction -- one violent
                # session contributes its whole cross-section to a 60-session
                # window, and at these quantiles (~99.8th) a single crash day
                # can BE the tail. The bar then sits far above anything the
                # market produces for the next three months, and every one of
                # those sessions reports an honest-looking empty day caused by
                # history rather than by today.
                rec[name] = min(max(v, fixed * cfg.qualifier_floor_frac),
                                fixed * cfg.qualifier_ceil_frac)
            rows.append(rec)
    return pd.DataFrame(rows)


def add_qualifiers(out: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    """n_qualifiers, by fixed level or by trailing-scaled level."""
    if cfg.qualifier_mode == "absolute":
        levels = {n: float(getattr(cfg, fb)) for n, _c, _a, fb in QUALIFIERS}
        fired = sum(
            ((out[col].abs() if absolute else out[col]) >= levels[name]).fillna(False).astype(int)
            for name, col, absolute, _fb in QUALIFIERS)
        for name in levels:
            out[f"thr_{name}"] = levels[name]
        out["n_qualifiers"] = fired.astype(np.int8)
        return out

    thr = scaled_thresholds(out, cfg)
    if thr.empty:
        # No eligible names in the whole frame, so there is no distribution to
        # take a quantile from. Fall back to the fixed levels and let selection
        # return nothing. Crashing here would turn "everything was gated out"
        # -- which is a legitimate, if alarming, empty day -- into a dead scan,
        # and an empty day is the one output this project must always be able
        # to produce.
        for name, _col, _absolute, fallback in QUALIFIERS:
            out[f"thr_{name}"] = float(getattr(cfg, fallback))
        out["n_qualifiers"] = np.int8(0)
        return out
    out = out.merge(thr.rename(columns={n: f"thr_{n}" for n, _c, _a, _f in QUALIFIERS})
                       .drop(columns=["n_eligible"]), on=["date", "market"], how="left")
    fired = sum(
        ((out[col].abs() if absolute else out[col]) >= out[f"thr_{name}"]).fillna(False).astype(int)
        for name, col, absolute, _fb in QUALIFIERS)
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
