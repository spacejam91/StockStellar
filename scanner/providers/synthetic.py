"""Random-walk provider — the harness for Test 10, the null-data control.

This exists to answer one question before any real data is trusted: does the
pipeline manufacture an edge out of noise? Prices here are a driftless geometric
random walk and volume is drawn independently of returns, so there is genuinely
nothing to find. If the scanner reports a meaningful information coefficient on
this data, the bug is in the scanner and every positive result on real data is
worthless until it is fixed.

Two properties are load-bearing and easy to break by accident:
  1. No drift. Any drift makes "went up" partially predictable from "is up".
  2. Volume independent of |return|. Real markets correlate them; correlating
     them HERE would hand the volume pillar a genuine relationship with the
     displacement pillar and produce a non-zero IC that looks like signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["Energy", "Materials", "Industrials", "Financials", "Technology",
           "Healthcare", "Utilities", "Consumer", "RealEstate", "Communication"]


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, n_tickers: int = 400, n_days: int = 500, seed: int = 7,
                 annual_vol: float = 0.45, market: str = "US"):
        self.n_tickers = n_tickers
        self.n_days = n_days
        self.seed = seed
        self.annual_vol = annual_vol
        self.market = market

    def universe(self, market: str = "US") -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        tickers = [f"SYN{i:04d}" for i in range(self.n_tickers)]
        return pd.DataFrame({
            "ticker": tickers,
            "name": [f"Synthetic {t}" for t in tickers],
            "exchange": "SYNTH",
            "market": market,
            "sector": rng.choice(SECTORS, size=self.n_tickers),
        })

    def daily_bars(self, start: str | None = None, end: str | None = None,
                   market: str | None = None) -> pd.DataFrame:
        market = market or self.market
        rng = np.random.default_rng(self.seed)
        n, d = self.n_tickers, self.n_days

        dates = pd.bdate_range(end=pd.Timestamp("2026-09-18"), periods=d)
        sig = self.annual_vol / np.sqrt(252.0)

        # Driftless GBM. The -sig^2/2 keeps E[price] flat rather than drifting
        # up through Jensen's inequality, which would be a real (if small)
        # predictable component.
        shocks = rng.normal(0.0, sig, size=(d, n)) - 0.5 * sig**2
        logp = np.log(rng.uniform(3.0, 120.0, size=n)) + np.cumsum(shocks, axis=0)
        close = np.exp(logp)

        prev = np.vstack([close[0:1], close[:-1]])
        # Intraday geometry drawn independently of the close-to-close move.
        gap = np.exp(rng.normal(0, sig * 0.4, size=(d, n)))
        open_ = prev * gap
        span = np.abs(rng.normal(0, sig * 1.2, size=(d, n)))
        hi_raw = np.maximum(open_, close) * (1 + span)
        lo_raw = np.minimum(open_, close) * (1 - span)

        # Volume: lognormal per name, independent of |return| on purpose.
        base = rng.uniform(11.5, 15.5, size=n)
        volume = np.exp(base + rng.normal(0, 0.55, size=(d, n)))

        df = pd.DataFrame({
            "date": np.repeat(dates.to_numpy(), n),
            "ticker": np.tile(np.array([f"SYN{i:04d}" for i in range(n)]), d),
            "open": open_.ravel(),
            "high": hi_raw.ravel(),
            "low": lo_raw.ravel(),
            "close": close.ravel(),
            "volume": np.round(volume.ravel()),
        })
        df["market"] = market
        df = df.merge(self.universe(market)[["ticker", "sector"]], on="ticker", how="left")

        # Guarantee OHLC coherence after the independent draws.
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def benchmark(self, market: str | None = None) -> pd.DataFrame:
        """An equal-weight index of the synthetic universe, for the relstrength pillar."""
        bars = self.daily_bars(market=market)
        b = bars.groupby("date", as_index=False)["close"].mean()
        b = b.rename(columns={"close": "bench_close"})
        b["market"] = market or self.market
        return b
