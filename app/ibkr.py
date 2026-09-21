"""IBKR connection manager.

Wraps ib_async so the FastAPI app can share a single persistent connection
to IB Gateway. The IB API is stateful — opening/closing a connection per
request would be slow and would burn through client IDs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ib_async import IB


@dataclass
class Position:
    symbol: str
    exchange: str
    currency: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class AccountSummary:
    account_id: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    currency: str


class IBKRClient:
    """Thin async wrapper around ib_async.IB for the FastAPI app."""

    def __init__(self) -> None:
        self._ib = IB()
        self.host = os.getenv("IB_HOST", "127.0.0.1")
        self.port = int(os.getenv("IB_PORT", "4002"))
        self.client_id = int(os.getenv("IB_CLIENT_ID", "1"))

    @property
    def connected(self) -> bool:
        return self._ib.isConnected()

    async def connect(self) -> None:
        if self.connected:
            return
        await self._ib.connectAsync(
            host=self.host,
            port=self.port,
            clientId=self.client_id,
            readonly=False,  # set True if you want a hard guarantee no orders go out
        )

    async def disconnect(self) -> None:
        if self.connected:
            self._ib.disconnect()

    async def account_summary(self) -> list[AccountSummary]:
        """One row per account (most users have one)."""
        if not self.connected:
            await self.connect()
        rows = await self._ib.accountSummaryAsync()

        # accountSummaryAsync returns flat tag/value rows; pivot per account.
        by_account: dict[str, dict[str, Any]] = {}
        for r in rows:
            by_account.setdefault(r.account, {"account_id": r.account})[r.tag] = r.value
            by_account[r.account]["currency"] = r.currency

        out: list[AccountSummary] = []
        for acct in by_account.values():
            out.append(
                AccountSummary(
                    account_id=acct["account_id"],
                    net_liquidation=float(acct.get("NetLiquidation", 0) or 0),
                    total_cash=float(acct.get("TotalCashValue", 0) or 0),
                    buying_power=float(acct.get("BuyingPower", 0) or 0),
                    currency=acct.get("currency", "USD"),
                )
            )
        return out

    async def get_quote(self, symbol: str) -> float:
        # TODO Phase 2 (real IBKR): call self._ib.reqMktData(Stock(symbol, 'SMART', 'USD'))
        # and read the last/close tick. Need to manage subscription lifecycle.
        raise NotImplementedError("Real IBKR quote streaming not implemented yet — use BACKEND=mock")

    async def tick_all(self, symbols: list[str]) -> list[dict]:
        raise NotImplementedError("Real IBKR quote streaming not implemented yet — use BACKEND=mock")

    async def place_order(self, req):  # type: ignore[no-untyped-def]
        # TODO Phase 3 (real IBKR):
        #   contract = Stock(req.symbol, 'SMART', 'USD')
        #   order = MarketOrder(req.side, req.quantity) or LimitOrder(...)
        #   trade = self._ib.placeOrder(contract, order)
        #   await trade.filledEvent...
        raise NotImplementedError("Real IBKR order placement not implemented yet — use BACKEND=mock")

    async def cancel_all_orders(self) -> int:
        # TODO Phase 3 (real IBKR): self._ib.reqGlobalCancel(); return count
        raise NotImplementedError("Real IBKR order cancellation not implemented yet — use BACKEND=mock")

    async def positions(self) -> list[Position]:
        if not self.connected:
            await self.connect()
        # portfolio() gives positions WITH live market price + P&L.
        # positions() is leaner but lacks marks.
        items = self._ib.portfolio()
        out: list[Position] = []
        for p in items:
            c = p.contract
            out.append(
                Position(
                    symbol=c.symbol,
                    exchange=c.exchange or c.primaryExchange or "",
                    currency=c.currency,
                    quantity=float(p.position),
                    avg_cost=float(p.averageCost),
                    market_price=float(p.marketPrice),
                    market_value=float(p.marketValue),
                    unrealized_pnl=float(p.unrealizedPNL),
                    realized_pnl=float(p.realizedPNL),
                )
            )
        return out


# Module-level singleton — FastAPI imports this directly.
client = IBKRClient()
