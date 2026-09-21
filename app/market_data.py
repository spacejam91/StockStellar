"""Bulk market-data selector — READ ONLY, separate from the broker backend.

`app.backend.client` is a *broker* abstraction (account, positions, place_order)
with a per-symbol `get_quote()` bolted on. It cannot do universe-scale work: at
~10k symbols, per-symbol REST polling on a retail tier (60 req/min) is ~2.8
hours per sweep. The scanner needs whole-universe daily bars in a handful of
calls, so it gets its own seam.

Toggle with the MARKET_DATA env var:
    MARKET_DATA=mock    → deterministic synthetic bars, no network (default)
    MARKET_DATA=yahoo   → real daily bars via yfinance, free, unofficial

Everything in `scanner/` imports `provider` from here and doesn't care which
implementation it gets. A paid bulk provider drops in as a third branch.

Contracts
---------
universe()     -> DataFrame[ticker, market, sector]
daily_bars()   -> DataFrame[date, ticker, open, high, low, close, volume]
benchmarks()   -> DataFrame[date, market, bench_close]
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

BENCH_TICKERS = {"CA": "^GSPTSE", "US": "^GSPC"}

SECTORS = ["Energy", "Materials", "Financials", "Industrials", "Technology",
           "Health Care", "Consumer", "Utilities", "Real Estate", "Communication"]


# ---------------------------------------------------------------------------
# mock: deterministic, offline, and the only provider the tests should use
# ---------------------------------------------------------------------------

class MockMarketData:
    """Synthetic universe with a fixed seed.

    Deliberately a random walk with a few injected volume/gap anomalies: a
    random walk is the correct null case. If the scanner's evaluation shows an
    edge on this provider, the harness has a bug -- that is what it is for.
    """

    name = "mock"

    def __init__(self, n_tickers: int = 120, n_sessions: int = 420, seed: int = 20260921):
        self.n_tickers = n_tickers
        self.n_sessions = n_sessions
        self.seed = seed
        self._bars: pd.DataFrame | None = None
        self._bench: pd.DataFrame | None = None

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(end=pd.Timestamp(date.today()), periods=self.n_sessions)
        rows = []
        for i in range(self.n_tickers):
            market = "CA" if i % 2 == 0 else "US"
            ticker = f"{'CA' if market == 'CA' else 'US'}{i:03d}" + (".TO" if market == "CA" else "")
            px = float(rng.uniform(4, 180)) * np.exp(np.cumsum(rng.normal(0.0002, rng.uniform(0.012, 0.035), self.n_sessions)))
            vol = rng.lognormal(rng.uniform(11.5, 14.0), 0.5, self.n_sessions)
            for d in rng.choice(np.arange(60, self.n_sessions), size=rng.integers(2, 6), replace=False):
                px[d:] *= 1 + rng.choice([-1, 1]) * rng.uniform(0.05, 0.16)
                vol[d] *= rng.uniform(3.5, 9.0)
            o = px * (1 + rng.normal(0, 0.006, self.n_sessions))
            rows.append(pd.DataFrame({
                "date": dates, "ticker": ticker, "market": market,
                "sector": SECTORS[i % len(SECTORS)],
                "open": o,
                "high": np.maximum(o, px) * (1 + np.abs(rng.normal(0, 0.008, self.n_sessions))),
                "low": np.minimum(o, px) * (1 - np.abs(rng.normal(0, 0.008, self.n_sessions))),
                "close": px, "volume": vol,
            }))
        self._bars = pd.concat(rows, ignore_index=True)
        self._bench = pd.concat([
            pd.DataFrame({
                "date": dates, "market": m,
                "bench_close": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, self.n_sessions))),
            }) for m in ("CA", "US")
        ], ignore_index=True)

    def _ensure(self) -> None:
        if self._bars is None:
            self._build()

    def universe(self) -> pd.DataFrame:
        self._ensure()
        return (self._bars[["ticker", "market", "sector"]]
                .drop_duplicates("ticker").reset_index(drop=True))

    def daily_bars(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        self._ensure()
        df = self._bars
        if tickers is not None:
            df = df[df["ticker"].isin(list(tickers))]
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True).copy()

    def benchmarks(self, start=None, end=None) -> pd.DataFrame:
        self._ensure()
        df = self._bench
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True).copy()


# ---------------------------------------------------------------------------
# yahoo: free and real, with two honest caveats
# ---------------------------------------------------------------------------

class YahooMarketData:
    """Daily bars via yfinance.

    Caveats that matter for this use, not boilerplate:

    1. It BACK-ADJUSTS history for splits and dividends, so bars fetched today
       are not what was printed on the day. A calibration log built from it is
       therefore NOT point-in-time and will overstate any measured edge. Fine
       for building and for a first look; not what you trust a base rate on.
    2. It cannot enumerate an exchange. `universe()` reads ticker lists from
       data/universe_ca.csv and data/universe_us.csv (columns: ticker[,sector]);
       both are free downloads -- Nasdaq publishes nasdaqtraded.txt and TMX
       publishes its listings file. Without them this falls back to the app
       watchlist, which is a handful of names, not a universe.
    """

    name = "yahoo"
    BATCH = 150

    def universe(self) -> pd.DataFrame:
        frames = []
        for market, fname in (("CA", "universe_ca.csv"), ("US", "universe_us.csv")):
            p = DATA_DIR / fname
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d.columns = [c.strip().lower() for c in d.columns]
            if "ticker" not in d.columns:
                log.warning("%s has no 'ticker' column -- skipping", p.name)
                continue
            d["market"] = market
            if "sector" not in d.columns:
                d["sector"] = pd.NA
            frames.append(d[["ticker", "market", "sector"]])

        if frames:
            u = pd.concat(frames, ignore_index=True)
            u["ticker"] = u["ticker"].astype(str).str.strip().str.upper()
            return u.drop_duplicates("ticker").reset_index(drop=True)

        from app import store as app_store
        syms = app_store.list_symbols()
        log.warning(
            "no universe files in %s -- falling back to the %d-symbol app watchlist. "
            "A %d-name 'universe' cannot support cross-sectional ranking; drop "
            "universe_ca.csv / universe_us.csv in place.", DATA_DIR, len(syms), len(syms))
        return pd.DataFrame({"ticker": syms, "market": ["US"] * len(syms), "sector": pd.NA})

    def daily_bars(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        import yfinance as yf

        if tickers is None:
            tickers = self.universe()["ticker"].tolist()
        tickers = [str(t).upper() for t in tickers]
        start = start or (date.today() - timedelta(days=500))

        frames = []
        for i in range(0, len(tickers), self.BATCH):
            chunk = tickers[i:i + self.BATCH]
            try:
                raw = yf.download(chunk, start=str(start), end=str(end) if end else None,
                                  auto_adjust=True, progress=False, group_by="ticker",
                                  threads=True)
            except Exception as e:
                log.warning("yfinance batch %d failed: %s", i // self.BATCH, e)
                continue
            for t in chunk:
                try:
                    d = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
                except KeyError:
                    continue
                if d.empty:
                    continue
                d = d.reset_index()
                d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
                d = d.rename(columns={"adj_close": "close"})
                d["ticker"] = t
                frames.append(d[["date", "ticker", "open", "high", "low", "close", "volume"]])

        if not frames:
            raise RuntimeError("yfinance returned no bars for any ticker")
        bars = pd.concat(frames, ignore_index=True)
        bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None).dt.normalize()

        u = self.universe()
        bars = bars.merge(u, on="ticker", how="left")
        bars["market"] = bars["market"].fillna(
            np.where(bars["ticker"].str.endswith((".TO", ".V", ".CN", ".NE")), "CA", "US"))
        return bars

    def benchmarks(self, start=None, end=None) -> pd.DataFrame:
        import yfinance as yf

        start = start or (date.today() - timedelta(days=500))
        frames = []
        for market, tkr in BENCH_TICKERS.items():
            try:
                d = yf.download(tkr, start=str(start), end=str(end) if end else None,
                                auto_adjust=True, progress=False)
            except Exception as e:
                log.warning("benchmark %s failed: %s", tkr, e)
                continue
            if d.empty:
                continue
            d = d.reset_index()
            d.columns = [str(c[0] if isinstance(c, tuple) else c).strip().lower() for c in d.columns]
            frames.append(pd.DataFrame({
                "date": pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize(),
                "market": market, "bench_close": d["close"].astype(float)}))
        if not frames:
            log.warning("no benchmarks fetched -- relstrength will fall back to universe median")
            return pd.DataFrame(columns=["date", "market", "bench_close"])
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("MARKET_DATA", "mock").lower()

if _PROVIDER == "yahoo":
    provider = YahooMarketData()
    log.info("Market data: YAHOO (free daily bars, back-adjusted, not point-in-time)")
else:
    provider = MockMarketData()
    log.info("Market data: MOCK (deterministic synthetic bars, offline)")


def get_provider(name: str | None = None):
    """Explicit provider lookup, for tests and for the CLI's --provider flag."""
    if name is None:
        return provider
    name = name.lower()
    if name == "yahoo":
        return YahooMarketData()
    if name == "mock":
        return MockMarketData()
    raise ValueError(f"unknown market data provider: {name!r} (mock|yahoo)")
