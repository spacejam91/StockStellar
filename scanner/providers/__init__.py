"""Provider registry. Mirrors the app's BACKEND env-var idiom.

    SCANNER_PROVIDER=synthetic   random-walk data, no network (the null control)
    SCANNER_PROVIDER=polygon     free grouped-daily: one call, whole US market
"""

from __future__ import annotations

import logging
import os

from scanner.providers.base import BARS_CONTRACT, MarketDataProvider, validate_bars

log = logging.getLogger(__name__)

__all__ = ["BARS_CONTRACT", "MarketDataProvider", "validate_bars", "get_provider"]


def get_provider(name: str | None = None, **kwargs) -> MarketDataProvider:
    name = (name or os.getenv("SCANNER_PROVIDER", "synthetic")).lower()
    if name == "synthetic":
        from scanner.providers.synthetic import SyntheticProvider
        return SyntheticProvider(**kwargs)
    if name == "polygon":
        from scanner.providers.polygon import PolygonProvider
        return PolygonProvider(**kwargs)
    raise ValueError(f"unknown SCANNER_PROVIDER {name!r} (have: synthetic, polygon)")
