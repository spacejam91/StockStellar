"""The market-data seam.

This is deliberately NOT the existing app/backend.py abstraction. That one is
broker-shaped — account, positions, place_order, with a per-symbol get_quote()
bolted on. Correct for a live dashboard watching a handful of names; fatal here,
because the scanner's primitive is "give me every name at once". Ten thousand
sequential get_quote() calls gets rate-limited into oblivion long before it
finishes a single sweep.

So the primitive is bulk by construction. A provider that can only answer one
symbol at a time does not belong behind this interface.

Selection mirrors the app's existing BACKEND pattern so there is one idiom in
the project, not two:

    SCANNER_PROVIDER=synthetic   # random-walk data, no network, for the null control
    SCANNER_PROVIDER=polygon     # free grouped-daily: one call, whole US market
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

# What every provider must return from daily_bars(). Extra columns are fine and
# are carried through; these are the ones the metrics layer requires.
BARS_CONTRACT = ["date", "ticker", "open", "high", "low", "close", "volume"]


@runtime_checkable
class MarketDataProvider(Protocol):
    """Read-only bulk market data. No trading, no account state, by design."""

    name: str

    def daily_bars(self, start: str, end: str, market: str = "US") -> pd.DataFrame:
        """Daily OHLCV for the whole universe over [start, end], tidy/long.

        Must return the BARS_CONTRACT columns. One row per (ticker, date).
        """
        ...

    def universe(self, market: str = "US") -> pd.DataFrame:
        """Current listed universe: ticker, name, exchange, market [, sector]."""
        ...


def validate_bars(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    """Contract check plus a fill-rate assertion.

    The fill-rate half is not paranoia. Several candidate upstreams return HTTP
    200 with silent nulls for columns they no longer recognise — TradingView's
    scanner endpoint does exactly this for any unknown column name. Combined
    with this project's absolute eligibility threshold and its honest empty-day
    output, a silently-dead column is indistinguishable from a quiet market:
    the scanner keeps running, reports nothing, and looks healthy while having
    been dead for weeks. Fail loudly instead.
    """
    missing = [c for c in BARS_CONTRACT if c not in df.columns]
    if missing:
        raise ValueError(f"provider {provider!r} returned bars missing {missing}")

    if df.empty:
        raise ValueError(f"provider {provider!r} returned zero rows")

    for col in ("close", "volume"):
        fill = df[col].notna().mean()
        if fill < 0.90:
            raise ValueError(
                f"provider {provider!r}: column {col!r} is only {fill:.1%} populated "
                f"(expected >=90%). Treat this as a dead upstream, not a quiet market."
            )

    dupes = df.duplicated(subset=["ticker", "date"]).sum()
    if dupes:
        raise ValueError(f"provider {provider!r} returned {dupes} duplicate (ticker, date) rows")

    return df
