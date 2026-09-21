"""Backend selector.

Toggle with the BACKEND env var:
    BACKEND=mock    → fake data + simulated trades (default, no setup)
    BACKEND=yahoo   → REAL (~15min delayed) quotes from Yahoo Finance,
                       trades still simulated. No signup.
    BACKEND=ibkr    → real ib_async connection to IB Gateway (Phase 2/3/4 of
                       the real-broker path; needs Gateway running).

Everything else in the app imports `client` from here and doesn't care
which implementation it gets.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_BACKEND = os.getenv("BACKEND", "mock").lower()

if _BACKEND == "ibkr":
    from app.ibkr import client  # noqa: F401  (re-exported)
    log.info("Backend: REAL IBKR (ib_async)")
elif _BACKEND == "yahoo":
    from app.yahoo import YahooClient
    client = YahooClient()
    log.info("Backend: YAHOO (real ~15min delayed quotes, simulated trading)")
else:
    from app.mock import MockClient
    client = MockClient()
    log.info("Backend: MOCK (simulated data)")
