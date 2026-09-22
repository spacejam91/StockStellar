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
    MARKET_DATA=yahoo   → real daily bars via yfinance. Free and unofficial,
                          and it CANNOT serve a full 8,700-name universe: a
                          sweep that size rate-limits even from a residential
                          IP. Fine for the ~2,800-name Canadian half.
    MARKET_DATA=polygon → the entire US market per request, end-of-day, free
                          tier, works from a datacenter IP. Needs
                          POLYGON_API_KEY. US only; Canada is unavailable at
                          any Polygon tier.

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
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

BENCH_TICKERS = {"CA": "^GSPTSE", "US": "^GSPC"}

# Providers whose bars are generated, not observed. Any measurement that is
# meant to describe the real world MUST exclude these. Adding a synthetic
# provider without adding it here is the single most damaging mistake
# available in this file: `signal` plants a deliberately strong artificial
# edge, so leaking it into the base-rate table would report a manufactured
# edge as measured fact -- and the base rate is the ONLY probability-shaped
# number the scanner produces.
SYNTHETIC_PROVIDERS = frozenset({"mock", "null", "signal"})

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
            # Events are planted after a 60-session warm-up so the rolling
            # inputs exist. With <=60 sessions there is no room, and
            # rng.choice on an empty range raises -- so plant nothing.
            room = np.arange(60, self.n_sessions)
            n_ev = int(min(rng.integers(2, 6), len(room)))
            for d in (rng.choice(room, size=n_ev, replace=False) if n_ev else []):
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
    # 150 back-to-back threaded batches trips Yahoo's limiter even from a
    # residential IP: a full 8,700-name sweep produced 13 YFRateLimitErrors,
    # and a rate-limited chunk is DROPPED, so the scan quietly scores a
    # universe with holes in it. Smaller batches, a pause between them, and
    # explicit backoff-and-retry on a limit error.
    BATCH = 80
    PAUSE = 1.2          # seconds between batches
    MAX_RETRIES = 4

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

        frames, dropped = [], 0
        for i in range(0, len(tickers), self.BATCH):
            chunk = tickers[i:i + self.BATCH]
            raw = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    raw = yf.download(chunk, start=str(start), end=str(end) if end else None,
                                      auto_adjust=True, progress=False, group_by="ticker",
                                      threads=True)
                    break
                except Exception as e:                            # noqa: BLE001
                    limited = "rate" in str(e).lower() or "too many" in str(e).lower()
                    if attempt == self.MAX_RETRIES - 1:
                        log.warning("yfinance batch %d gave up after %d tries: %s",
                                    i // self.BATCH, self.MAX_RETRIES, e)
                        break
                    wait = (4 ** attempt if limited else 2 ** attempt) + 1
                    log.warning("yfinance batch %d %s, retrying in %ds",
                                i // self.BATCH, "rate-limited" if limited else "failed", wait)
                    time.sleep(wait)
            if raw is None:
                dropped += len(chunk)
                continue
            if i + self.BATCH < len(tickers):
                time.sleep(self.PAUSE)
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

        returned = {f["ticker"].iloc[0] for f in frames if len(f)}
        coverage = len(returned) / max(len(tickers), 1)
        if coverage < 0.95:
            # THE failure mode this provider actually has. yf.download never
            # raises on a per-ticker failure: yfinance catches it internally,
            # stores an empty frame and only logs "N Failed downloads". So the
            # retry loop above never fires, `dropped` stays 0, and two thirds of
            # the universe can vanish with every counter reading zero. That is
            # exactly what happened -- a 22:30 UTC run right after the US close
            # returned ~1/3 of the names and reported nothing wrong, while the
            # same code off-peak returned all 8,701.
            log.warning(
                "COVERAGE %.1f%%: only %d of %d requested tickers returned bars. "
                "Cross-sectional ranks are percentiles, so a short universe moves "
                "EVERY score. Treat this as a degraded run, not a quiet market.",
                100 * coverage, len(returned), len(tickers))
        if dropped:
            # Loud on purpose. A silently shortened universe changes every
            # cross-sectional rank in the scan, and the output still looks fine.
            log.warning("DROPPED %d of %d tickers to fetch failures (%.1f%%) -- "
                        "cross-sectional ranks are computed on the survivors",
                        dropped, len(tickers), 100 * dropped / max(len(tickers), 1))
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
        bars.attrs["coverage"] = coverage
        bars.attrs["requested"] = len(tickers)
        bars.attrs["returned"] = len(returned)
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



BARS_CONTRACT = ["date", "ticker", "open", "high", "low", "close", "volume"]

# Below this share of requested tickers the run is refused outright. Set from
# observation: a healthy sweep returns ~100%, the degraded CI run returned ~35%.
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.80"))


def validate_bars(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    """Contract + fill-rate check on every fetch.

    The fill-rate half is the point. Several upstreams return HTTP 200 with
    silent nulls for a column they no longer recognise -- TradingView's scanner
    does exactly this for any unknown column name. In a scanner whose correct
    output on a quiet day is NOTHING, a dead column is indistinguishable from a
    quiet market: the scan runs, reports no names, and looks healthy while
    having been dead for weeks. Fail loudly instead.
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
                f"provider {provider!r}: column {col!r} only {fill:.1%} populated "
                f"(expected >=90%). Treat this as a dead upstream, not a quiet market.")
    dupes = int(df.duplicated(subset=["ticker", "date"]).sum())
    if dupes:
        raise ValueError(f"provider {provider!r} returned {dupes} duplicate (ticker, date) rows")

    # Coverage is the check that matters most and the one nothing had. A
    # provider that returns 30% of the universe produces a perfectly
    # well-formed frame: every column present, every value populated, no
    # duplicates. It just describes a different, smaller market -- and since
    # composite_z is a percentile, every score in the run is wrong.
    cov = df.attrs.get("coverage")
    if cov is not None and cov < MIN_COVERAGE:
        raise ValueError(
            f"provider {provider!r}: only {cov:.1%} of requested tickers returned bars "
            f"({df.attrs.get('returned')} of {df.attrs.get('requested')}, floor "
            f"{MIN_COVERAGE:.0%}). A short universe moves every cross-sectional score; "
            f"this is a degraded upstream, not a quiet market.")
    return df


class _Validated:
    """Wraps a provider so daily_bars() cannot bypass the guard.

    Structural rather than conventional: a new provider cannot forget to call
    it, because it never gets the chance.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def daily_bars(self, *a, **kw):
        return validate_bars(self._inner.daily_bars(*a, **kw), getattr(self._inner, "name", "?"))


# pandas' default na_values include "NA", "NULL", "NaN", "None" and "nan" --
# every one of which is a plausible ticker. NA.TO is National Bank of Canada.
# Reading a universe file without keep_default_na=False silently turns such a
# ticker into a float NaN, and .astype(str).str.upper() then resurrects it as
# the literal string "NAN" -- which is exactly how a phantom "NAN" ticker with
# 343 duplicate rows got into the bars.
def read_universe(path, **kw):
    import pandas as _pd
    return _pd.read_csv(path, keep_default_na=False, na_values=[""], **kw)


def get_provider(name: str | None = None):
    """Explicit provider lookup, for tests and for the CLI's --provider flag."""
    if name is None:
        return provider
    name = name.lower()
    impls = {"yahoo": YahooMarketData, "mock": MockMarketData,
             "null": NullMarketData, "signal": SignalMarketData}
    if name == "polygon":
        from app.polygon_data import PolygonMarketData
        return _Validated(PolygonMarketData())
    known = sorted(set(impls) | {"polygon"})
    if name not in impls:
        raise ValueError(
            f"unknown market data provider: {name!r} (have: {'|'.join(known)})")
    return _Validated(impls[name]())


# Built through get_provider() so the module-level singleton is WRAPPED. It was
# not, and that mattered: run_scan() defaults to this object, so every scheduled
# scan and every default local run bypassed validate_bars completely. The guard
# existed, was documented as "structural rather than conventional", and was
# inert on the one path production actually uses. It caught nothing -- including
# 343 duplicate (ticker, date) rows from a phantom "NAN" ticker.
# No silent fallback. An unrecognised MARKET_DATA used to land on MockMarketData,
# which plants a 5-16% level shift and a 3.5-9x volume spike by design -- so a
# typo in the workflow would publish picks found in FABRICATED events, labelled
# with whatever the env var said. get_provider raises on an unknown name.
provider = get_provider(_PROVIDER)
log.info("Market data: %s", getattr(provider, "name", _PROVIDER).upper())
