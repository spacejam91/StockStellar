"""Step 3 — cross-sectional normalisation within (date, market).

CA and US are ranked separately on purpose: they are different liquidity
regimes, and a shared ranking lets US megacaps set the scale that TSXV juniors
are measured against.

Rank-based rather than raw z-scores, because one halted name with RVOL 90 would
otherwise hijack the composite, and daily equity data is full of them. Note there
is deliberately no winsorisation: clipping at the 1st/99th percentile preserves
order, so it cannot change a rank. It is pure cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scanner.config import INPUTS
from scanner.mathx import rank_to_z

GROUP_KEYS = ["date", "market"]


def normalize(df: pd.DataFrame, inputs: dict | None = None) -> pd.DataFrame:
    """Add an `n_<input>` column for every input present in `df`.

    Magnitude inputs are ranked on |x| -- a -3 sigma move is exactly as unusual
    as a +3 sigma one, and the direction is re-applied later by the side logic.
    """
    inputs = inputs or INPUTS
    out = df.copy()

    present = [c for c in inputs if c in out.columns]
    if not present:
        raise ValueError("no scoring inputs found -- run metrics.compute() first")

    grouped = out.groupby(GROUP_KEYS, sort=False, observed=True)
    for col in present:
        _, is_magnitude = inputs[col]
        src = out[col].abs() if is_magnitude else out[col]
        out[f"n_{col}"] = (src.groupby([out[k] for k in GROUP_KEYS], sort=False)
                              .transform(lambda s: rank_to_z(s.to_numpy())))
    del grouped
    return out


def normalize_fundamentals(df: pd.DataFrame, directions: dict[str, int]) -> pd.DataFrame:
    """Same treatment for fundamental inputs, then multiplied by their direction.

    Only columns actually present and not entirely null are normalised, so a
    pillar with no data for this market drops out of the composite instead of
    being scored as neutral. That distinction matters most for CA small caps,
    where analyst coverage is routinely absent rather than merely mediocre.
    """
    out = df.copy()
    for col, direction in directions.items():
        if col not in out.columns or not out[col].notna().any():
            continue
        out[f"n_{col}"] = (out[col].groupby([out[k] for k in GROUP_KEYS], sort=False)
                                   .transform(lambda s: rank_to_z(s.to_numpy()))) * float(direction)
    return out


def coverage(df: pd.DataFrame, inputs: dict | None = None) -> pd.DataFrame:
    """Null rate per normalised input on the latest session.

    Check this per market before comparing a CA score against a US one: if an
    input is 90% null for CA, its pillar is being carried by the other members.
    """
    inputs = inputs or INPUTS
    if df.empty:
        return pd.DataFrame()
    last = df[df["date"] == df["date"].max()]
    rows = []
    for col in inputs:
        n_col = f"n_{col}"
        if n_col not in last.columns:
            continue
        for market, grp in last.groupby("market", observed=True):
            rows.append({"input": col, "market": market, "n": len(grp),
                         "null_pct": round(100 * float(grp[n_col].isna().mean()), 1)})
    return pd.DataFrame(rows).sort_values(["input", "market"]).reset_index(drop=True)
