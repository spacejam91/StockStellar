"""FastAPI app.

Phase 1: read-only portfolio dashboard.
Phase 2: watchlist + live quote streaming over WebSocket.

Run:
    cd ~/Documents/Claude/Projects/stockstellar
    uv run uvicorn app.main:app --reload

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import store
from app.backend import client
from app.orders import (
    MAX_DAILY_LOSS,
    MAX_ORDER_VALUE,
    OrderRequest,
    check_pretrade_risk,
    today_start_ts,
)
from app.quotes import hub

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stockstellar")

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# How often we push watchlist tick updates. 1s is plenty for an MVP.
TICK_INTERVAL_SECONDS = 1.0


async def quote_producer(hub_) -> None:
    """Background loop: pull/drift watchlist prices and broadcast as ticks."""
    while True:
        try:
            symbols = store.list_symbols()
            if symbols and hasattr(client, "tick_all"):
                ticks = await client.tick_all(symbols)
                for t in ticks:
                    await hub_.broadcast({"type": "tick", **t})
        except Exception as e:
            log.warning("quote_producer error: %s", e)
        await asyncio.sleep(TICK_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await client.connect()
        log.info("Connected to backend %s:%s (clientId=%s)", client.host, client.port, client.client_id)
    except Exception as e:
        log.warning("Could not connect to backend on startup: %s", e)
    hub.start_producer(quote_producer)
    yield
    await hub.stop_producer()
    await client.disconnect()


app = FastAPI(title="StockStellar", lifespan=lifespan)


# ---- Health + account + positions ----------------------------------------------

@app.get("/health")
async def health():
    return {"connected": client.connected, "host": client.host, "port": client.port}


@app.get("/api/account")
async def api_account():
    try:
        summary = await client.account_summary()
        return [s.__dict__ for s in summary]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/positions")
async def api_positions():
    try:
        positions = await client.positions()
        return [p.__dict__ for p in positions]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---- Watchlist (Phase 2) -------------------------------------------------------

class AddSymbol(BaseModel):
    symbol: str


@app.get("/api/watchlist")
async def api_watchlist():
    symbols = store.list_symbols()
    out = []
    for s in symbols:
        try:
            price = await client.get_quote(s)
        except NotImplementedError:
            price = None
        out.append({"symbol": s, "price": price})
    return out


@app.post("/api/watchlist")
async def api_watchlist_add(payload: AddSymbol):
    sym = payload.symbol.strip().upper()
    if not sym:
        return JSONResponse({"error": "symbol required"}, status_code=400)
    store.add_symbol(sym)
    try:
        price = await client.get_quote(sym)
    except NotImplementedError:
        price = None
    return {"symbol": sym, "price": price}


@app.delete("/api/watchlist/{symbol}")
async def api_watchlist_remove(symbol: str):
    store.remove_symbol(symbol)
    return {"ok": True}


# ---- Orders + safeguards (Phase 3) ---------------------------------------------

@app.get("/api/orders")
async def api_orders(limit: int = 50):
    return store.list_orders(limit=limit)


@app.get("/api/risk")
async def api_risk():
    return {
        "halted": store.halted(),
        "max_order_value": MAX_ORDER_VALUE,
        "max_daily_loss": MAX_DAILY_LOSS,
        "daily_realized_pnl": store.daily_realized_pnl(today_start_ts()),
    }


@app.post("/api/orders")
async def api_place_order(req: OrderRequest):
    # Estimate price for the risk check.
    try:
        last_price = await client.get_quote(req.symbol)
    except NotImplementedError:
        return JSONResponse({"error": "Quotes not available on this backend"}, status_code=400)

    estimated = req.limit_price if (req.order_type == "LMT" and req.limit_price) else last_price
    pnl_today = store.daily_realized_pnl(today_start_ts())
    risk = check_pretrade_risk(req, estimated, pnl_today, store.halted())

    if not risk.ok:
        order_id = store.log_order(
            symbol=req.symbol, side=req.side, quantity=req.quantity,
            order_type=req.order_type, limit_price=req.limit_price,
            status="rejected", fill_price=None, rejection_reason=risk.reason,
        )
        return JSONResponse(
            {"status": "rejected", "order_id": order_id, "reason": risk.reason},
            status_code=400,
        )

    # Send to backend.
    filled = await client.place_order(req)
    order_id = store.log_order(
        symbol=filled.symbol, side=filled.side, quantity=filled.quantity,
        order_type=filled.order_type, limit_price=filled.limit_price,
        status=filled.status, fill_price=filled.fill_price,
        realized_pnl=filled.realized_pnl,
        rejection_reason=filled.rejection_reason,
    )
    return {
        "status": filled.status,
        "order_id": order_id,
        "fill_price": filled.fill_price,
        "reason": filled.rejection_reason,
    }


@app.post("/api/halt")
async def api_halt(payload: dict):
    """Toggle kill switch. Body: {\"halted\": true|false}. Also cancels open orders."""
    new_state = bool(payload.get("halted"))
    store.set_halted(new_state)
    if new_state:
        try:
            await client.cancel_all_orders()
        except NotImplementedError:
            pass
        await hub.broadcast({"type": "halt", "halted": True})
    else:
        await hub.broadcast({"type": "halt", "halted": False})
    return {"halted": new_state}


@app.websocket("/ws/quotes")
async def ws_quotes(ws: WebSocket):
    await hub.connect(ws)
    try:
        # We don't process inbound messages — receive_text() blocks until
        # the client disconnects, at which point WebSocketDisconnect fires.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)


# ---- Dashboard -----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    error = None
    accounts: list = []
    positions: list = []
    watchlist: list = []
    orders: list = []
    try:
        if not client.connected:
            await client.connect()
        accounts = await client.account_summary()
        positions = await client.positions()
        for s in store.list_symbols():
            try:
                price = await client.get_quote(s)
            except NotImplementedError:
                price = None
            watchlist.append({"symbol": s, "price": price})
        orders = store.list_orders(limit=20)
    except Exception as e:
        error = str(e)

    return TEMPLATES.TemplateResponse(
        request,
        "dashboard.html",
        {
            "connected": client.connected,
            "host": client.host,
            "port": client.port,
            "accounts": accounts,
            "positions": positions,
            "watchlist": watchlist,
            "orders": orders,
            "halted": store.halted(),
            "max_order_value": MAX_ORDER_VALUE,
            "max_daily_loss": MAX_DAILY_LOSS,
            "daily_pnl": store.daily_realized_pnl(today_start_ts()),
            "error": error,
        },
    )


# ---- Scanner -------------------------------------------------------------------
# The scan is a read-only screen over bulk daily bars (app.market_data), entirely
# separate from the broker client above. It never places an order, and its output
# is a description of which names are statistically unusual today -- not a
# forecast and not a recommendation.

@app.get("/api/scan")
async def api_scan(as_of: str | None = None):
    """Latest logged watchlist. Cheap: reads the log, does not run a scan."""
    from scanner import store as scan_store

    day = scan_store.day_for(as_of)
    if day is None:
        return JSONResponse({
            "as_of": None, "picks": [], "day": None,
            "message": "no scan logged yet -- POST /api/scan/run or: uv run python -m scanner",
        })
    return JSONResponse({
        "as_of": day["as_of_date"],
        "day": day,
        "picks": scan_store.picks_for(day["as_of_date"]),
        "score_means": "score_pct is a 0-100 percentile of criteria strength within "
                       "that session's eligible universe. Not a probability of gain.",
    })


@app.get("/api/scan/history")
async def api_scan_history(limit: int = 60):
    """Per-session audit including empty days, newest first."""
    from scanner import store as scan_store

    hist = scan_store.day_history(limit=limit)
    empty = [r for r in hist if r["empty_day"]]
    return JSONResponse({
        "sessions": len(hist),
        "empty_days": len(empty),
        "empty_day_share": round(len(empty) / len(hist), 3) if hist else None,
        "history": hist,
    })


@app.get("/api/scan/base-rates")
async def api_scan_base_rates(horizon: int = 5, include_mock: bool = False):
    """What actually followed each score band, measured from the log.

    The only probability-shaped output the scanner has, and it is a measurement:
    read `se_pp` before anything else, since a band with few observations is not
    a finding.
    """
    from scanner import store as scan_store

    df = scan_store.score_band_base_rates(horizon=horizon, include_mock=include_mock)
    return JSONResponse({
        "horizon_days": horizon,
        "include_mock": include_mock,
        "providers_logged": scan_store.provider_breakdown(),
        "bands": [] if df.empty else df.to_dict(orient="records"),
        "note": "measured base rates from the point-in-time log, not predictions. "
                "Mock-provider rows are excluded by default: mock bars are a random "
                "walk and would dilute a real measurement toward 50%.",
    })


class ScanRunRequest(BaseModel):
    history_days: int = 0
    threshold: float | None = None
    persist: bool = True


@app.post("/api/scan/run")
async def api_scan_run(req: ScanRunRequest):
    """Run a scan now. Synchronous, and slow on a real provider over a big
    universe -- the scheduled path should call `python -m scanner` instead."""
    from scanner import run_scan
    from scanner.config import ScanConfig

    cfg = ScanConfig()
    if req.threshold is not None:
        cfg.composite_threshold = req.threshold
    try:
        result = await asyncio.to_thread(
            run_scan, cfg=cfg, history_days=req.history_days, persist=req.persist)
    except Exception as e:
        log.warning("scan failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

    today = result.today()
    return JSONResponse({
        "as_of": str(result.as_of.date()),
        "provider": result.provider,
        "empty_day": result.is_empty_day,
        "picks": [] if today.empty else today.to_dict(orient="records"),
        "health": result.health,
    })


@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request, as_of: str | None = None):
    """Latest logged session, or a past one with ?as_of=YYYY-MM-DD."""
    from scanner import store as scan_store

    day = scan_store.day_for(as_of)
    picks = scan_store.picks_for(day["as_of_date"]) if day else []
    hist = scan_store.day_history(limit=30)
    empty_share = (sum(1 for r in hist if r["empty_day"]) / len(hist)) if hist else None
    return TEMPLATES.TemplateResponse(
        request,
        "scan.html",
        {
            "day": day,
            "picks": picks,
            "history": hist,
            "empty_share": empty_share,
            "provider": os.getenv("MARKET_DATA", "mock"),
        },
    )
