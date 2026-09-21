"""Yahoo Finance backend — real (~15 min delayed) quotes, simulated trading.

Extends MockClient so account/positions/orders/safeguards all keep working;
only the price source is swapped from random-walk to yfinance.

Caveats:
  - Prices from Yahoo are usually 15-min delayed for US stocks.
  - yfinance is an unofficial scraper — Yahoo can change endpoints anytime.
  - Calls are sync, so we run them in a thread pool via asyncio.to_thread().
  - We cache each symbol's price for QUOTE_CACHE_TTL seconds to stay polite.
"""

from __future__ import annotations

import asyncio
import logging
import time

import yfinance as yf

from app.mock import MockClient

log = logging.getLogger(__name__)


class YahooClient(MockClient):
    QUOTE_CACHE_TTL = 1.0  # seconds — matches our tick interval

    def __init__(self) -> None:
        super().__init__()
        self.host = "yahoo"
        self.port = 0
        self._quote_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, fetched_ts)

    async def connect(self) -> None:
        await super().connect()
        # Warm the cache for seeded positions so the first dashboard render
        # shows real prices, not random-walked drift from the seed values.
        symbols = list({p.symbol for p in self._positions})
        if symbols:
            log.info("Yahoo: warming quote cache for %s", symbols)
            await asyncio.gather(*(self.get_quote(s) for s in symbols))

    # --- price source override ------------------------------------------------

    def _drift_prices(self) -> None:
        """No-op: real prices are pulled on demand by get_quote()."""
        return

    @staticmethod
    def _fetch_sync(symbol: str) -> float | None:
        try:
            t = yf.Ticker(symbol)
            price = getattr(t.fast_info, "last_price", None)
            return float(price) if price else None
        except Exception as e:
            log.warning("Yahoo fetch failed for %s: %s", symbol, e)
            return None

    async def get_quote(self, symbol: str) -> float:
        symbol = symbol.upper()
        cached = self._quote_cache.get(symbol)
        now = time.time()
        if cached and now - cached[1] < self.QUOTE_CACHE_TTL:
            return cached[0]

        price = await asyncio.to_thread(self._fetch_sync, symbol)
        if price is None:
            # Unknown symbol or transient failure — fall back to last known
            # value (or a seeded random one for first-time symbols).
            price = self._seed_quote(symbol)
            log.info("Yahoo: no data for %s, using fallback %.2f", symbol, price)

        self._quote_cache[symbol] = (price, now)
        self._watchlist_prices[symbol] = price
        return price

    async def tick_all(self, symbols: list[str]) -> list[dict]:
        """Fetch all symbols concurrently (thread pool) and emit tick payloads."""
        prices = await asyncio.gather(*(self.get_quote(s) for s in symbols))
        ts = time.time()
        return [
            {"symbol": s, "price": round(p, 2), "ts": ts}
            for s, p in zip(symbols, prices)
        ]

    # --- ensure positions/account/orders see fresh marks ----------------------

    async def _refresh_position_prices(self) -> None:
        symbols = [p.symbol for p in self._positions if p.quantity > 0]
        if symbols:
            await asyncio.gather(*(self.get_quote(s) for s in symbols))

    async def account_summary(self):
        await self._refresh_position_prices()
        return await super().account_summary()

    async def positions(self):
        await self._refresh_position_prices()
        return await super().positions()

    async def place_order(self, req):
        # Make sure the fill price is based on a fresh quote, not a stale cache.
        await self.get_quote(req.symbol)
        return await super().place_order(req)
