"""Order types, validation, and safeguards.

Two layers of protection before any order leaves the app:
  1. Schema validation (Pydantic) — rejects malformed input at the API edge.
  2. Pre-trade risk checks — caps on order value, halt switch, daily loss.

Real orders go through the same gauntlet whether they're placed manually
from the UI or by an algo (Phase 4). Don't bypass these checks anywhere.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Risk caps — read once at import time. Override via env if you want.
MAX_ORDER_VALUE = float(os.getenv("MAX_ORDER_VALUE", "25000"))   # USD/CAD per order
MAX_DAILY_LOSS  = float(os.getenv("MAX_DAILY_LOSS",  "1000"))    # absolute, e.g. -1000 trips it


Side = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT"]
TIF = Literal["DAY", "GTC"]
OrderStatus = Literal["filled", "rejected", "open", "cancelled"]


class OrderRequest(BaseModel):
    """What the UI / algo submits."""
    symbol: str = Field(..., min_length=1, max_length=12)
    side: Side
    quantity: float = Field(..., gt=0)
    order_type: OrderType = "MKT"
    limit_price: float | None = Field(None, gt=0)
    tif: TIF = "DAY"

    @field_validator("symbol")
    @classmethod
    def upcase(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("limit_price")
    @classmethod
    def limit_required_for_lmt(cls, v, info):
        if info.data.get("order_type") == "LMT" and v is None:
            raise ValueError("limit_price required for LMT orders")
        return v


@dataclass
class FilledOrder:
    """Result of a place_order() call (filled OR rejected by the backend)."""
    id: int
    timestamp: float
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    limit_price: float | None
    status: OrderStatus
    fill_price: float | None
    rejection_reason: str | None
    realized_pnl: float = 0.0  # Non-zero only on closing SELLs


@dataclass
class RiskCheckResult:
    ok: bool
    reason: str = ""


def check_pretrade_risk(
    req: OrderRequest,
    estimated_price: float,
    daily_realized_pnl: float,
    halted: bool,
) -> RiskCheckResult:
    """All pre-trade checks in one place. Call this BEFORE any order goes out.

    Args:
        req: the validated order request
        estimated_price: best estimate of fill price (last tick for MKT, limit for LMT)
        daily_realized_pnl: sum of realized P&L since start-of-day (negative = loss)
        halted: kill-switch state
    """
    if halted:
        return RiskCheckResult(False, "Kill switch is active — trading halted")

    notional = req.quantity * estimated_price
    if notional > MAX_ORDER_VALUE:
        return RiskCheckResult(
            False,
            f"Order value ${notional:,.2f} exceeds per-order cap ${MAX_ORDER_VALUE:,.2f}",
        )

    if daily_realized_pnl <= -abs(MAX_DAILY_LOSS):
        return RiskCheckResult(
            False,
            f"Daily realized loss ${abs(daily_realized_pnl):,.2f} hit cap ${MAX_DAILY_LOSS:,.2f}",
        )

    return RiskCheckResult(True)


def today_start_ts() -> float:
    """Unix ts for local midnight. Used to scope 'daily' realized P&L."""
    import datetime
    d = datetime.date.today()
    return time.mktime(d.timetuple())
