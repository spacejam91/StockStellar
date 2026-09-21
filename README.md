# StockStellar

Personal Interactive Brokers trading app — portfolio dashboard, watchlist with
alerts, manual trade entry with safeguards, and a simple algo strategy. Talks
to IB Gateway via `ib_async`.

## Setup

### 1. Install IB Gateway (one-time)

1. Download **IB Gateway — Latest** from
   <https://www.interactivebrokers.com/en/trading/ibgateway-latest.php>
   (the "Latest" build, not "Stable" — stable is older).
2. Install and launch. **Log in with your PAPER credentials**, not live.
3. After login, open **Configure → Settings → API → Settings**:
   - Enable ActiveX and Socket Clients (check)
   - Read-Only API: uncheck (we need to place orders later)
   - Socket port: **4002** (paper-trading port)
   - Master API client ID: leave blank
   - Trusted IPs: add `127.0.0.1`
4. Apply, then OK. Leave Gateway running.

### 2. Run the app

```bash
cd ~/Documents/Claude/Projects/stockstellar
cp .env.example .env       # tweak if you used a non-default port
uv run uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

## Safety notes

- IB Gateway logs **paper** and **live** completely separately. The port
  number is your only safety rail: **4002 = paper, 4001 = live**. Double-check
  `.env` before every run until Phase 3 adds explicit guards.
- IB Gateway auto-logs-out around **midnight ET**. Just log back in.
- One `clientId` per connected app. If you connect twice with id=1, the
  second connection kicks the first one off.

## Roadmap

- **Phase 1** Read-only portfolio dashboard (current)
- **Phase 2** Watchlist + live quotes + alerts
- **Phase 3** Manual trade entry with safeguards (kill switch, position caps)
- **Phase 4** Simple MA crossover algo, paper only

## Scanner

Daily cross-sectional screen over the CA + US universe. Produces at most 3 names,
and honestly produces none when nothing qualifies.

```bash
uv run python -m scanner                          # today, offline mock provider
uv run python -m scanner --diagnose               # + gate rejections, input coverage
uv run python -m scanner --history-days 120 --backfill --report
MARKET_DATA=yahoo uv run python -m scanner        # real bars
```

Web: `/scan` (phone-friendly), `/api/scan`, `/api/scan/history`,
`/api/scan/base-rates`, `POST /api/scan/run`.

**Pipeline** (`scanner/`, in spec order): `metrics` per-name inputs in own-history
units → `gates` eligibility + absolute qualifier count → `rank` cross-sectional
rank→z within (date, market) → `combine` pillars → side → `composite_z`/`score_pct`
→ `select` threshold, then rank, then cap at 3 → `store` point-in-time log in
`stockstellar.db` (`scan_*` tables).

**Bulk data** is `app/market_data.py` (`MARKET_DATA=mock|yahoo`) — read-only and
deliberately separate from `app/backend.py`, which is the broker. At ~10k symbols
per-symbol polling is ~2.8h per sweep on a retail tier, so the scanner only ever
asks for bulk daily bars.

**What the score means.** `score_pct` is a 0–100 percentile of criteria strength
within that session's eligible universe — 92 means only 8% of the universe fired
this hard on these criteria. It is not a probability of gain, not a forecast, and
not a recommendation. The only probability-shaped output is
`/api/scan/base-rates`, which measures what actually followed each score band from
the log, with n and standard errors.

Two thresholds do different jobs: `composite_z >= 2.0` is *relative* (~2.3% of any
universe clears it every session, so on its own it can never produce an empty day),
and `n_qualifiers >= 2` is *absolute*, in own-history units (RVOL ≥ 2, |ret_z| ≥ 2,
range ≥ 1.5×ATR, |gap| ≥ 0.75×ATR) — that half is what makes a quiet day return
nothing. Watch the empty-day share: below ~20% the threshold is too loose.

Full criteria spec, including the validation tests:
`~/Documents/Claude/Outputs/market-scanner/PARAMETERS.md`
