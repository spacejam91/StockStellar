"""WebSocket fan-out hub for streaming quote ticks to browsers.

One background producer task drifts/pulls prices for watchlist symbols and
calls hub.broadcast(...). Each browser holds one /ws/quotes connection and
receives every broadcast.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from fastapi import WebSocket

log = logging.getLogger(__name__)


class QuoteHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._producer: asyncio.Task | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info("WS client connected (%d total)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("WS client disconnected (%d total)", len(self._clients))

    async def broadcast(self, msg: dict[str, Any]) -> None:
        payload = json.dumps(msg)
        dead: list[WebSocket] = []
        async with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                await c.send_text(payload)
            except Exception:
                dead.append(c)
        if dead:
            async with self._lock:
                for c in dead:
                    self._clients.discard(c)

    def start_producer(self, coro: Callable[["QuoteHub"], Awaitable[None]]) -> None:
        if self._producer and not self._producer.done():
            return
        self._producer = asyncio.create_task(coro(self))

    async def stop_producer(self) -> None:
        if self._producer and not self._producer.done():
            self._producer.cancel()
            try:
                await self._producer
            except asyncio.CancelledError:
                pass


hub = QuoteHub()
