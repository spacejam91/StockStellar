"""Step 2 — per-name inputs, all in own-history units.

Input contract (from datasource): date, ticker, open, high, low, close, volume,
and optionally sector and market. Everything here is computed per ticker over
its own history; nothing is cross-sectional yet. That happens in rank.py.

The whole point of own-history units (own ATR, own sigma) is that a $3 TSXV
junior and a $400 megacap become comparable before they are ever ranked against
each other.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _g(df: pd.DataFrame, col: str):
    return df.groupby("ticker", sort=False)[col]


def _roll(df: pd.DataFrame, col: str, window: int, fn: str, min_periods: int | None = None):
    mp = window if min_periods is None else min_periods
    return _g(df, col).transform(lambda s: getattr(s.rolling(window, min_periods=mp), fn)())


def compute(
    bars: pd.DataFrame,
    benchmarks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add every Step 2 input column to a tidy bars frame.

    `benchmarks` is optional and shaped (date, market, bench_close) — the TSX
    Composite for CA and the S&P 500 for US. Without it the relstrength pillar
    is simply absent, which the combine step handles by renormalising weights
    rather than scoring a blank as neutral.
    """
    missing = [c for c in REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required column(s): {missing}")

    df = bars.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    if "market" not in df.columns:
        df["market"] = "US"
    if "sector" not in df.columns:
        df["sector"] = pd.NA

    # --- base series --------------------------------------------------------
    df["prev_close"] = _g(df, "close").shift(1)
    df["ret"] = df["close"] / df["prev_close"] - 1.0
    df["dollar_vol"] = df["close"] * df["volume"]

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["prev_close"]).abs()
    lc = (df["low"] - df["prev_close"]).abs()
    df["true_range"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    df["atr14"] = _roll(df, "true_range", 14, "mean")
    df["ret_std60"] = _roll(df, "ret", 60, "std")
    df["vol_med20"] = _roll(df, "volume", 20, "median")
    df["dv_med20"] = _roll(df, "dollar_vol", 20, "median")
    df["session_n"] = _g(df, "close").cumcount() + 1
    df["gap_days"] = _g(df, "date").diff().dt.days

    # --- volume pillar (side-neutral) --------------------------------------
    df["rvol"] = _safe_div(df["volume"], df["vol_med20"])
    dv_mean60 = _roll(df, "dollar_vol", 60, "mean")
    dv_std60 = _roll(df, "dollar_vol", 60, "std")
    df["dollar_vol_z"] = _safe_div(df["dollar_vol"] - dv_mean60, dv_std60)

    # --- displacement pillar (flips) ---------------------------------------
    df["ret_z"] = _safe_div(df["ret"], df["ret_std60"])
    df["gap_atr"] = _safe_div(df["open"] - df["prev_close"], df["atr14"])

    # --- range pillar (side-neutral) ---------------------------------------
    df["range_ratio"] = _safe_div(df["true_range"], df["atr14"])
    bb_width = _safe_div(4.0 * _roll(df, "close", 20, "std"), _roll(df, "close", 20, "mean"))
    df["_bb_width"] = bb_width
    # Percentile of today's Bollinger width against this name's own trailing
    # year — catches squeeze -> expansion, which raw width cannot.
    # Distance from the middle, NOT the raw percentile. The comment above says
    # this catches "squeeze -> expansion", but ranking the raw percentile scores
    # a squeeze LOWEST: a name at the tightest Bollinger width of its own year
    # sits at 0.004 and gets the least unusual z in a side-neutral pillar.
    # Half the stated purpose was simply not implemented. |pct - 0.5| makes
    # both extremes -- coiled and expanded -- score as unusual, which is what a
    # side-neutral range input is supposed to mean.
    _pct = _g(df, "_bb_width").transform(
        lambda s: s.rolling(252, min_periods=60).rank(pct=True)
    )
    df["bb_width_pct"] = (_pct - 0.5).abs()
    df = df.drop(columns=["_bb_width"])

    # --- structure pillar (flips) ------------------------------------------
    hi20 = _g(df, "high").transform(lambda s: s.rolling(20, min_periods=20).max().shift(1))
    hi55 = _g(df, "high").transform(lambda s: s.rolling(55, min_periods=55).max().shift(1))
    lo20 = _g(df, "low").transform(lambda s: s.rolling(20, min_periods=20).min().shift(1))
    lo55 = _g(df, "low").transform(lambda s: s.rolling(55, min_periods=55).min().shift(1))
    df["donchian_dir"] = (
        (df["close"] > hi20).astype(float)
        + (df["close"] > hi55).astype(float)
        - (df["close"] < lo20).astype(float)
        - (df["close"] < lo55).astype(float)
    )
    # A name with no 55d window yet has an unknowable breakout state, not a
    # neutral one. Keep it NaN so it drops out of the pillar mean.
    df.loc[hi55.isna() | lo55.isna(), "donchian_dir"] = np.nan

    # Shifted by one, like the donchian windows above. Including today puts
    # today's own high inside the range it is measured against, so a new
    # 52-week high can never exceed +0.5 and every breakout -- by a cent or by
    # 30% -- lands in the same razor-thin band. Measured before the fix: the
    # observed range over 200 names was -0.4949..0.4972, never reaching the
    # bound. Shifting restores the ability to rank breakouts by how far they
    # actually broke.
    hi252 = _g(df, "high").transform(
        lambda s: s.rolling(252, min_periods=252).max().shift(1))
    lo252 = _g(df, "low").transform(
        lambda s: s.rolling(252, min_periods=252).min().shift(1))
    df["pct_52w_centered"] = _safe_div(df["close"] - lo252, hi252 - lo252) - 0.5
    df["ma_dist_atr"] = _safe_div(df["close"] - _roll(df, "close", 50, "mean"), df["atr14"])

    # --- relstrength pillar (flips) ----------------------------------------
    df["ret_5d"] = _g(df, "close").transform(lambda s: s / s.shift(5) - 1.0)
    df["ret_20d"] = _g(df, "close").transform(lambda s: s / s.shift(20) - 1.0)
    if benchmarks is not None and not benchmarks.empty:
        df = _attach_benchmark(df, benchmarks)
    else:
        df["rs_5d"] = np.nan
        df["rs_20d"] = np.nan
        log.info("no benchmark supplied — relstrength pillar will be partially absent")

    # Sector-relative is the input that separates "the whole sector moved" from
    # "this name moved", which is most of the value in the pillar.
    if df["sector"].notna().any():
        med = df.groupby(["date", "market", "sector"], sort=False)["ret_20d"].transform("median")
        df["rs_sector_20d"] = df["ret_20d"] - med
    else:
        df["rs_sector_20d"] = np.nan

    # --- confirmation pillar (flips) ---------------------------------------
    # Signed in market direction, scaled by volume excess. This is what strips
    # out reversal-wick names that score beautifully on volume and range alone:
    # a heavy up-day closing on its low scores NEGATIVE for the long side.
    close_loc = _safe_div(df["close"] - df["low"], df["high"] - df["low"])
    df["vp_thrust"] = (close_loc - 0.5) * 2.0 * np.log1p(np.clip(df["rvol"] - 1.0, 0, None))

    # --- flags, carried but never scored ------------------------------------
    # An unadjusted split is a textbook 2-sigma move with no volume behind it,
    # and the scanner will "find" it every single time if nobody looks.
    df["suspect_unadjusted_split"] = (df["ret"].abs() > 0.35) & (df["rvol"] < 1.5)

    # Sanitise EVERY scoring input, not just the ones that happen to go through
    # _safe_div. That helper only guards denominators, so ret, ret_5d, ret_20d
    # and rs_sector_20d could all carry inf -- a single close of 0.0 anywhere in
    # a ticker's history produces inf in all four. An inf then takes the maximum
    # z in its group and the top of the shortlist with it. Done here, once, so a
    # new input cannot forget it.
    from scanner.config import INPUTS
    for col in list(INPUTS) + ["ret", "ret_5d", "ret_20d"]:
        if col in df.columns:
            df[col] = _finite(df[col])

    return df


def _attach_benchmark(df: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    b = benchmarks.sort_values(["market", "date"], kind="mergesort").copy()
    g = b.groupby("market", sort=False)["bench_close"]
    b["bench_5d"] = g.transform(lambda s: s / s.shift(5) - 1.0)
    b["bench_20d"] = g.transform(lambda s: s / s.shift(20) - 1.0)
    out = df.merge(b[["date", "market", "bench_5d", "bench_20d"]], on=["date", "market"], how="left")
    out["rs_5d"] = out["ret_5d"] - out["bench_5d"]
    out["rs_20d"] = out["ret_20d"] - out["bench_20d"]
    return out.drop(columns=["bench_5d", "bench_20d"])


def _finite(s: pd.Series) -> pd.Series:
    """inf -> NaN. An inf survives ranking as the single most extreme name in
    the universe and takes the top of the list with it; rank_to_z([1,2,3,inf])
    hands inf the maximum z. A NaN is excluded from the ranking instead, which
    is the correct treatment for a value that is not a number."""
    return s.replace([np.inf, -np.inf], np.nan)


def _safe_div(num, den):
    """Divide, mapping 0/0 and x/0 to NaN instead of inf.

    An inf here would survive ranking as the most extreme value in the universe
    and hijack the top-3 — exactly the failure the rank-based path exists to
    prevent, so it must not sneak in through arithmetic.
    """
    den = pd.Series(den).replace(0, np.nan) if not isinstance(den, pd.Series) else den.replace(0, np.nan)
    return pd.Series(num) / den
