"""Bulk market-data selector — READ ONLY, separate from the broker backend.

The scanner's primitive is "give me every name at once". At ~10k symbols,
per-symbol REST polling on a retail tier (60 req/min) is ~2.8 hours for a
single sweep, so a provider that can only answer one symbol at a time does not
belong behind this interface. Bulk by construction.

Toggle with the MARKET_DATA env var:
    MARKET_DATA=mock    → deterministic synthetic bars, no network (default).
                          Plants events: development data AND a smoke test,
                          NOT a null case.
    MARKET_DATA=null    → driftless random walk, volume independent of returns.
                          Nothing to find. The Test 10 harness.
    MARKET_DATA=signal  → null plus a deliberately strong planted signal.
                          The positive control: proves the measuring apparatus
                          can detect an edge, so a clean null means something.
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
    """Offline development data, and the POSITIVE control.

    This is NOT a null case, despite being mostly a random walk. It plants real
    structure on purpose: a permanent 5-16% level shift and a 3.5-9x volume
    spike on the SAME session, two to five times per name, on top of a small
    positive drift. That is precisely the volume/displacement relationship the
    scanner is built to detect.

    So the scanner SHOULD show an edge here, and a near-zero IC on this provider
    means the measurement apparatus is broken, not that the scanner is honest.

    For the null control -- no drift, volume independent of returns, nothing to
    find -- use NullMarketData below. Do not confuse the two: running Test 10
    against this provider would "validate" the harness by rediscovering events
    that were planted for it, which is the exact failure the test exists to
    catch.
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
# null: the Test 10 harness -- provably nothing to find
# ---------------------------------------------------------------------------

class NullMarketData:
    """Driftless random walk with volume independent of returns.

    The whole point is that there is no signal here, so the scanner must score
    ~zero on it. If it does not, the pipeline manufactures an edge out of noise
    and every positive result on real data is worthless until that is fixed.

    Three properties are load-bearing and easy to break by accident:

    1. NO DRIFT. The -sigma^2/2 term keeps E[price] flat rather than drifting
       up through Jensen's inequality. Any drift makes "went up" partially
       predictable from "is up", which is a real (if small) edge.
    2. VOLUME INDEPENDENT OF |RETURN|. Real markets correlate them; correlating
       them here would hand the volume pillar a genuine relationship with the
       displacement pillar and produce an IC that looks like signal.
    3. NO INJECTED EVENTS. No level shifts, no volume spikes. Contrast
       MockMarketData, which plants both on purpose.

    If you edit this class, re-read those three before you commit.
    """

    name = "null"

    def __init__(self, n_tickers: int = 120, n_sessions: int = 420, seed: int = 11,
                 annual_vol: float = 0.40, planted_drift: float = 0.0,
                 planted_horizon: int = 10):
        self.n_tickers = n_tickers
        self.n_sessions = n_sessions
        self.seed = seed
        self.annual_vol = annual_vol
        # planted_drift > 0 turns this into the POSITIVE control: see
        # SignalMarketData. Left at 0.0 this is a pure null.
        self.planted_drift = planted_drift
        self.planted_horizon = planted_horizon
        self._bars: pd.DataFrame | None = None
        self._bench: pd.DataFrame | None = None

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        n, d = self.n_tickers, self.n_sessions
        dates = pd.bdate_range(end=pd.Timestamp(date.today()), periods=d)
        sig = self.annual_vol / np.sqrt(252.0)

        # Driftless GBM in PRICE: the -sig^2/2 makes E[S_t+1/S_t] = 1, so simple
        # returns -- the thing evaluate.py correlates against -- have zero mean.
        # Mean LOG return is then -sig^2/2 by construction. That is volatility
        # drag, an identity, not drift; do not "fix" it.
        shocks = rng.normal(0.0, sig, size=(d, n)) - 0.5 * sig**2

        # Volume drawn from its own state, never a function of the returns.
        volume = np.exp(rng.uniform(11.5, 14.0, size=n) + rng.normal(0, 0.5, size=(d, n)))

        if self.planted_drift > 0:
            # Positive control. A volume spike on day t is followed by drift in
            # a consistent direction over the next `planted_horizon` sessions,
            # so the spike genuinely PREDICTS forward return. This is the thing
            # mock does not have: mock's level shift is fully realised on the
            # event day, leaving nothing ahead to detect.
            # The drift starts on day t ITSELF, not t+1. That matters: the
            # scanner scores day t, so the direction has to be observable in
            # day t's own bar or there is nothing for it to key on. Drift
            # beginning at t+1 plants a move whose direction is unknowable at
            # scan time, which no scanner could detect and which therefore
            # tests nothing.
            h = self.planted_horizon
            for j in range(n):
                for t in rng.choice(np.arange(60, d - h), size=rng.integers(3, 7), replace=False):
                    volume[t, j] *= rng.uniform(4.0, 9.0)
                    direction = rng.choice([-1.0, 1.0])
                    shocks[t, j] += direction * self.planted_drift * 4.0   # visible same-day move
                    shocks[t + 1:t + 1 + h, j] += direction * self.planted_drift

        close = np.exp(np.log(rng.uniform(4.0, 180.0, size=n)) + np.cumsum(shocks, axis=0))

        prev = np.vstack([close[0:1], close[:-1]])
        open_ = prev * np.exp(rng.normal(0, sig * 0.4, size=(d, n)))
        span = np.abs(rng.normal(0, sig * 1.1, size=(d, n)))
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)

        idx = np.arange(n)
        self._bars = pd.DataFrame({
            "date": np.repeat(dates.to_numpy(), n),
            "ticker": np.tile(np.array([f"{'CA' if i % 2 == 0 else 'US'}{i:03d}"
                                        + (".TO" if i % 2 == 0 else "") for i in idx]), d),
            "market": np.tile(np.where(idx % 2 == 0, "CA", "US"), d),
            "sector": np.tile(np.array([SECTORS[i % len(SECTORS)] for i in idx]), d),
            "open": open_.ravel(), "high": high.ravel(),
            "low": low.ravel(), "close": close.ravel(), "volume": volume.ravel(),
        })
        # Keep OHLC coherent after the independent draws.
        self._bars["high"] = self._bars[["open", "high", "low", "close"]].max(axis=1)
        self._bars["low"] = self._bars[["open", "high", "low", "close"]].min(axis=1)
        self._bars = self._bars.sort_values(["ticker", "date"]).reset_index(drop=True)

        bench_shocks = rng.normal(0.0, 0.009, size=d) - 0.5 * 0.009**2
        self._bench = pd.concat([
            pd.DataFrame({"date": dates, "market": m,
                          "bench_close": 100 * np.exp(np.cumsum(bench_shocks))})
            for m in ("CA", "US")
        ], ignore_index=True)

    _ensure = MockMarketData._ensure
    universe = MockMarketData.universe
    daily_bars = MockMarketData.daily_bars
    benchmarks = MockMarketData.benchmarks


class SignalMarketData(NullMarketData):
    """The POSITIVE control: identical to the null except signal is planted.

    A volume spike on session t is followed by consistent drift over the next
    10 sessions, so the spike genuinely predicts forward return. A scanner that
    cannot find THIS cannot find anything, which is what makes a clean null
    control meaningful rather than vacuous.

    Note this is a deliberately unrealistic, unusually strong relationship. It
    exists to prove the measuring apparatus works, and says nothing whatsoever
    about whether real markets contain anything comparable. They do not.
    """

    name = "signal"

    def __init__(self, **kw):
        kw.setdefault("planted_drift", 0.015)
        kw.setdefault("seed", 23)
        super().__init__(**kw)


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
                # yfinance is inconsistent about the index name across versions:
                # 1.4.0 leaves it UNNAMED, so reset_index() yields "index", not
                # "date". Older builds give "Date"; intraday gives "Datetime".
                # Normalise all three rather than assuming one.
                d = d.rename(columns={"adj_close": "close", "index": "date", "datetime": "date"})
                if "date" not in d.columns:
                    log.warning("yfinance frame for %s has no date column (%s) — skipping",
                                t, list(d.columns))
                    continue
                d["ticker"] = t
                frames.append(d[["date", "ticker", "open", "high", "low", "close", "volume"]])

        if not frames:
            raise RuntimeError("yfinance returned no bars for any ticker")
        bars = pd.concat(frames, ignore_index=True)
        bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None).dt.normalize()

        u = self.universe()
        bars = bars.merge(u, on="ticker", how="left")
        # pandas 3.0 rejects a bare ndarray as a fillna value, so wrap it in a
        # Series aligned to the frame.
        inferred = pd.Series(
            np.where(bars["ticker"].str.endswith((".TO", ".V", ".CN", ".NE")), "CA", "US"),
            index=bars.index,
        )
        bars["market"] = bars["market"].fillna(inferred) if "market" in bars.columns else inferred
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
            # Same unnamed-index trap as daily_bars(): yfinance 1.4.0 leaves the
            # index unnamed, so reset_index() gives "index", not "date".
            d = d.rename(columns={"index": "date", "datetime": "date"})
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
    if name == "null":
        return NullMarketData()
    if name == "signal":
        return SignalMarketData()
    raise ValueError(f"unknown market data provider: {name!r} (mock|null|signal|yahoo)")
