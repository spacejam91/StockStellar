"""StockStellar web interface.

A read-only view over the scanner: today's watchlist (at most three names, or
none), the per-session history including empty days, and measured base rates
from the point-in-time log.

There is no trading here. The app places no orders, holds no broker connection,
and knows nothing about an account. It reads bulk daily bars, ranks them, and
shows the result.

Run:
    cd ~/Documents/Claude/Projects/stockstellar
    uv run uvicorn app.main:app --reload

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import store

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stockstellar")

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="StockStellar")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/health")
async def health():
    return {"ok": True, "provider": os.getenv("MARKET_DATA", "mock")}


# ---- Watchlist -----------------------------------------------------------------
# Names you follow by hand, independent of what the scanner surfaces.

class AddSymbol(BaseModel):
    symbol: str


@app.get("/api/watchlist")
async def api_watchlist():
    return JSONResponse({"symbols": store.list_symbols()})


@app.post("/api/watchlist")
async def api_watchlist_add(payload: AddSymbol):
    try:
        store.add_symbol(payload.symbol)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"symbols": store.list_symbols()})


@app.delete("/api/watchlist/{symbol}")
async def api_watchlist_remove(symbol: str):
    store.remove_symbol(symbol)
    return JSONResponse({"symbols": store.list_symbols()})


# ---- Scanner -------------------------------------------------------------------
# A read-only screen over bulk daily bars (app.market_data). Its output describes
# which names are statistically unusual today. It is not a forecast and not a
# recommendation, and nothing here can place an order.

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


# ---- Pages ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
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
            # From the log, not the env var: the env says how this process
            # is configured now, the log says what produced this session.
            "provider": (day or {}).get("provider") or os.getenv("MARKET_DATA", "mock"),
        },
    )
