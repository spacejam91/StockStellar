"""Polygon (now Massive) grouped-daily provider — the whole US market per call.

Why this exists: Yahoo cannot serve a full universe. Sweeping 8,706 names
produced 33 YFRateLimitErrors in one run even from a residential IP and after
backoff, and a rate-limited batch that gets dropped silently shortens the
universe -- which moves every cross-sectional rank, because composite_z is a
percentile. Polygon's grouped-daily endpoint returns EVERY US ticker for a
session in a single request, so there is no per-symbol polling to rate-limit.

    GET /v2/aggs/grouped/locale/us/market/stocks/{date}

Free tier: end-of-day only, 2 years of history, and FIVE REQUESTS PER MINUTE.

That last number is the one that matters, and an earlier version of this file
asserted the opposite -- "unlimited calls to this endpoint" -- which was wrong
and was measured to be wrong: the API returns 429 "exceeded the maximum
requests per minute", and after enough of them it escalates to 401 "Unknown API
Key" on a key that worked seconds earlier.

One request returns one SESSION for the whole US market, so history costs one
request per trading day. At 5/min a cold build of 400 sessions is 80 minutes of
wall clock and ~0.6 GB of cached JSON. That is fine to do once on a laptop and
does not fit a CI job, so this provider is not a drop-in replacement for the
daily scheduled run until the cache is persisted between runs. It IS the right
source once warm: a daily run needs exactly one new request.

Each session is cached on disk forever, so the cost is paid once.

Set POLYGON_API_KEY. Get a free key at polygon.io (no card). US only -- Canada
is not available from Polygon at any tier, so the Canadian half stays on Yahoo,
where the smaller 2,784-name universe is inside what it will serve.

Bars are SPLIT-ADJUSTED but NOT dividend-adjusted (adjusted=true default). That
is the correct choice here: dividend adjustment rewrites history every time a
dividend is paid, which is exactly the "not point-in-time" problem that
overstates a measured edge.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Polygon rebranded to Massive in October 2025. The official clients now default
# to api.massive.com and api.polygon.io is kept alive "for an extended period" --
# which is a promise with no date on it, so the new host leads and the old one is
# the fallback. Both answered identically when checked (12,592 US tickers for the
# same session), so this is about outliving a deprecation, not about behaviour.
HOSTS = ("https://api.massive.com", "https://api.polygon.io")
PATH = "/v2/aggs/grouped/locale/us/market/stocks/{d}"
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE = DATA_DIR / ".polygon_cache"


class PolygonMarketData:
    """Grouped daily bars for the entire US market."""

    name = "polygon"

    # Free tier: 5 requests per minute. Sleeping 12.5s between calls keeps us
    # inside it by construction rather than discovering it as a 429 -- which
    # matters because the penalty escalates to a 401 on a valid key, and a
    # burst spent on a rate limit is a burst of history not fetched.
    RATE_LIMIT_PER_MIN = 5

    def __init__(self, api_key: str | None = None, sessions: int = 400,
                 adjusted: bool = True, pause: float | None = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.sessions = sessions
        self.adjusted = adjusted
        self.pause = (60.0 / self.RATE_LIMIT_PER_MIN + 0.5) if pause is None else pause
        self._bars: pd.DataFrame | None = None

    # -- universe -----------------------------------------------------------

    def universe(self) -> pd.DataFrame:
        p = DATA_DIR / "universe_us.csv"
        if not p.exists():
            raise SystemExit("run scripts/build_universe.py first")
        u = pd.read_csv(p)
        u["market"] = "US"
        cols = ["ticker", "market"] + [c for c in ("sector",) if c in u.columns]
        return u[cols]

    # -- bars ---------------------------------------------------------------

    def _fetch_day(self, d: date) -> list[dict] | None:
        cache = CACHE / f"{d.isoformat()}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text()).get("results")
            except (json.JSONDecodeError, OSError):
                pass
        query = (PATH.format(d=d.isoformat())
                 + f"?adjusted={'true' if self.adjusted else 'false'}&apiKey={self.api_key}")
        payload, last_err = None, None
        for host in HOSTS:
            try:
                with urllib.request.urlopen(host + query, timeout=45) as r:
                    payload = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # The limit is per MINUTE, so sleeping 15s just spends the
                    # next attempt on the same wall. Wait the window out.
                    log.warning("%s rate limit on %s; waiting out the 60s window", host, d)
                    time.sleep(62)
                    return self._fetch_day(d)
                if e.code in (401, 403):
                    # Try the other host before giving up: a key issued under
                    # one brand may not be registered against the other's key
                    # store. A dead key fails on both and still lands here.
                    last_err = e
                    continue
                log.info("%s %s: HTTP %d", host, d, e.code)
                last_err = e
                continue
            except (urllib.error.URLError, TimeoutError) as e:
                log.info("%s %s: %s", host, d, e)
                last_err = e
                continue
        if payload is None:
            if isinstance(last_err, urllib.error.HTTPError) and last_err.code in (401, 403):
                raise SystemExit(
                    f"the key was rejected by every host ({', '.join(HOSTS)}) with HTTP "
                    f"{last_err.code}. Set POLYGON_API_KEY to a valid key from massive.com "
                    f"(formerly polygon.io) -- and check it has not been rotated away.")
            return None

        results = payload.get("results")
        if results:                       # weekends/holidays return no results
            CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"results": results}))
        return results

    def daily_bars(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        if not self.api_key:
            raise SystemExit(
                "POLYGON_API_KEY is not set. Get a free key at polygon.io, then:\n"
                "    export POLYGON_API_KEY=...        (or add it to .env)")
        if self._bars is None:
            self._bars = self._build()
        df = self._bars
        if tickers is not None:
            df = df[df["ticker"].isin(list(tickers))]
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True).copy()

    def _build(self) -> pd.DataFrame:
        # Walk back calendar days, skipping the ones with no results (weekends
        # and holidays), until `sessions` trading days are collected.
        cached = len(list(CACHE.glob("*.json"))) if CACHE.exists() else 0
        need = max(self.sessions - cached, 0)
        if need > 20:
            log.warning(
                "polygon: %d of %d sessions are not cached. At %d requests/minute that "
                "is about %d minutes. Each session is cached permanently, so this is a "
                "one-time cost -- but it will not fit inside a CI job.",
                need, self.sessions, self.RATE_LIMIT_PER_MIN,
                round(need / self.RATE_LIMIT_PER_MIN))
        rows, d, got, tried = [], date.today(), 0, 0
        while got < self.sessions and tried < self.sessions * 2:
            tried += 1
            res = self._fetch_day(d)
            d -= timedelta(days=1)
            if not res:
                continue
            got += 1
            stamp = d + timedelta(days=1)
            for r in res:
                rows.append((stamp, r.get("T"), r.get("o"), r.get("h"),
                             r.get("l"), r.get("c"), r.get("v")))
            if self.pause:
                time.sleep(self.pause)

        if not rows:
            raise RuntimeError("polygon returned no sessions")
        df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high",
                                         "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()

        # Restrict to our own universe: grouped-daily returns everything that
        # traded, including warrants, units and preferreds already excluded
        # deliberately in build_universe.py.
        u = self.universe()
        df = df.merge(u, on="ticker", how="inner")
        log.info("polygon: %d sessions, %d rows, %d tickers",
                 got, len(df), df["ticker"].nunique())
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def benchmarks(self, start=None, end=None) -> pd.DataFrame:
        """Equal-weight index of the fetched universe.

        Polygon's free tier has no index data, and SPY is an ETF that
        build_universe.py deliberately excludes, so it is not in these bars.
        An equal-weight mean of the universe is a defensible stand-in for a
        relative-strength denominator and needs no extra request.
        """
        if self._bars is None:
            self._bars = self._build()
        b = self._bars.groupby("date", as_index=False)["close"].mean()
        b = b.rename(columns={"close": "bench_close"})
        b["market"] = "US"
        if start is not None:
            b = b[b["date"] >= pd.Timestamp(start)]
        if end is not None:
            b = b[b["date"] <= pd.Timestamp(end)]
        return b.reset_index(drop=True)
