"""Mock backend — simulates an IBKR connection with fake account, positions,
drifting prices, and instant-fill orders. Lets us build and demo every phase
without IB Gateway.

Has the same public interface as IBKRClient in ibkr.py, so the rest of the
app doesn't care which is plugged in.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from app.ibkr import AccountSummary, Position
from app.orders import FilledOrder, OrderRequest


# Realistic-ish seed portfolio. Quantity + avg cost are fixed; current market
# price is read from _watchlist_prices (the single source of truth, so a
# BUY filled at the displayed tick price actually matches what you see).
_SEED_POSITIONS = [
    # symbol, currency, quantity, avg_cost, starting_price
    ("AAPL", "USD", 50,  175.20, 232.50),
    ("MSFT", "USD", 25,  310.00, 428.10),
    ("NVDA", "USD", 40,   95.40, 138.25),
    ("TSLA", "USD", 15,  220.00, 198.40),
    ("SHOP", "CAD", 100,  78.50, 142.80),
]

_SEED_ACCOUNT = {
    "account_id": "DU1234567",  # IBKR paper accounts start with DU
    "starting_cash": 38_400.00,
    "buying_power": 200_000.00,
    "currency": "USD",
}


@dataclass
class _PositionState:
    symbol: str
    currency: str
    quantity: float
    avg_cost: float
    realized_pnl: float = 0.0


class MockClient:
    """In-memory fake of IBKRClient. Same public API."""

    def __init__(self) -> None:
        self._connected = False
        self.host = "mock"
        self.port = 0
        self.client_id = 1
        # Stable randomness so the dashboard is the same across reloads
        # within one process — but each call still drifts prices.
        self._rng = random.Random(42)
        self._positions = [
            _PositionState(s, c, q, ac) for (s, c, q, ac, _) in _SEED_POSITIONS
        ]
        # Single source of truth for any symbol's current price. Positions
        # and watchlist both read from here.
        self._watchlist_prices: dict[str, float] = {
            s: p for (s, _, _, _, p) in _SEED_POSITIONS
        }
        self._cash: float = _SEED_ACCOUNT["starting_cash"]
        self._next_order_id = 1
        self._last_tick = time.time()

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def _drift_prices(self) -> None:
        """Random walk every call. Roughly +/- 0.3% per tick.
        Only positions held; watchlist drift is handled by tick_all().
        """
        for p in self._positions:
            old = self._watchlist_prices.get(p.symbol, p.avg_cost)
            shock = self._rng.gauss(0, 0.003)
            self._watchlist_prices[p.symbol] = max(0.01, old * (1 + shock))
        self._last_tick = time.time()

    async def account_summary(self) -> list[AccountSummary]:
        if not self._connected:
            await self.connect()
        self._drift_prices()
        positions_value = sum(
            p.quantity * self._watchlist_prices.get(p.symbol, p.avg_cost)
            for p in self._positions
        )
        return [
            AccountSummary(
                account_id=_SEED_ACCOUNT["account_id"],
                net_liquidation=self._cash + positions_value,
                total_cash=self._cash,
                buying_power=_SEED_ACCOUNT["buying_power"],
                currency=_SEED_ACCOUNT["currency"],
            )
        ]

    async def positions(self) -> list[Position]:
        if not self._connected:
            await self.connect()
        self._drift_prices()
        out: list[Position] = []
        for p in self._positions:
            if p.quantity == 0:
                continue
            mark = self._watchlist_prices.get(p.symbol, p.avg_cost)
            out.append(
                Position(
                    symbol=p.symbol,
                    exchange="SMART",
                    currency=p.currency,
                    quantity=p.quantity,
                    avg_cost=p.avg_cost,
                    market_price=mark,
                    market_value=p.quantity * mark,
                    unrealized_pnl=(mark - p.avg_cost) * p.quantity,
                    realized_pnl=p.realized_pnl,
                )
            )
        return out

    # ---- Quote streaming (Phase 2) -------------------------------------------------

    def _seed_quote(self, symbol: str) -> float:
        """Fake a plausible starting price for any symbol we've never seen."""
        if symbol not in self._watchlist_prices:
            # Seed using position price if we hold it, else pick a random plausible price.
            existing = next((p.price for p in self._positions if p.symbol == symbol), None)
            self._watchlist_prices[symbol] = (
                existing if existing else round(self._rng.uniform(20, 500), 2)
            )
        return self._watchlist_prices[symbol]

    async def get_quote(self, symbol: str) -> float:
        return self._seed_quote(symbol)

    async def tick_all(self, symbols: list[str]) -> list[dict]:
        """Drift each watchlist symbol's price by ~0.3% and return tick payloads."""
        ticks: list[dict] = []
        now = time.time()
        for s in symbols:
            old = self._seed_quote(s)
            shock = self._rng.gauss(0, 0.003)
            new = max(0.01, old * (1 + shock))
            self._watchlist_prices[s] = new
            ticks.append({"symbol": s, "price": round(new, 2), "ts": now})
        return ticks

    # ---- Order placement (Phase 3) -------------------------------------------------

    def _get_or_create_position(self, symbol: str, currency: str = "USD") -> _PositionState:
        for p in self._positions:
            if p.symbol == symbol:
                return p
        p = _PositionState(symbol=symbol, currency=currency, quantity=0, avg_cost=0)
        self._positions.append(p)
        return p

    async def place_order(self, req: OrderRequest) -> FilledOrder:
        """Instant-fill simulation.

        MKT: fills at current tick price (with a tiny ~0.05% slippage against you).
        LMT: fills immediately at limit_price if marketable, else rejects.
             (Real broker would leave it open; MVP keeps it simple.)
        """
        if not self._connected:
            await self.connect()

        order_id = self._next_order_id
        self._next_order_id += 1

        mark = self._seed_quote(req.symbol)
        slip_factor = 1 + (0.0005 if req.side == "BUY" else -0.0005)

        if req.order_type == "MKT":
            fill_price = round(mark * slip_factor, 2)
        else:  # LMT
            marketable = (
                (req.side == "BUY"  and (req.limit_price or 0) >= mark) or
                (req.side == "SELL" and (req.limit_price or 0) <= mark)
            )
            if not marketable:
                return FilledOrder(
                    id=order_id, timestamp=time.time(),
                    symbol=req.symbol, side=req.side, quantity=req.quantity,
                    order_type=req.order_type, limit_price=req.limit_price,
                    status="rejected", fill_price=None,
                    rejection_reason=f"LMT not marketable (last={mark:.2f})",
                )
            fill_price = req.limit_price  # type: ignore[assignment]

        pos = self._get_or_create_position(req.symbol)
        realized = 0.0

        if req.side == "BUY":
            new_qty = pos.quantity + req.quantity
            # Weighted-average cost basis (only when increasing the long; a buy
            # against a short would partially cover — we don't simulate shorts).
            if pos.quantity >= 0:
                pos.avg_cost = (
                    (pos.quantity * pos.avg_cost + req.quantity * fill_price) / new_qty
                    if new_qty > 0 else 0
                )
            pos.quantity = new_qty
            self._cash -= req.quantity * fill_price
        else:  # SELL
            if pos.quantity < req.quantity:
                return FilledOrder(
                    id=order_id, timestamp=time.time(),
                    symbol=req.symbol, side=req.side, quantity=req.quantity,
                    order_type=req.order_type, limit_price=req.limit_price,
                    status="rejected", fill_price=None,
                    rejection_reason=f"Insufficient position ({pos.quantity} held)",
                )
            realized = (fill_price - pos.avg_cost) * req.quantity
            pos.quantity -= req.quantity
            pos.realized_pnl += realized
            self._cash += req.quantity * fill_price

        return FilledOrder(
            id=order_id, timestamp=time.time(),
            symbol=req.symbol, side=req.side, quantity=req.quantity,
            order_type=req.order_type, limit_price=req.limit_price,
            status="filled", fill_price=fill_price,
            rejection_reason=None, realized_pnl=realized,
        )

    async def cancel_all_orders(self) -> int:
        """No open orders in this mock (everything fills instantly), so this
        is a no-op that returns 0. Real IBKR will need to call ib.reqGlobalCancel().
        """
        return 0
