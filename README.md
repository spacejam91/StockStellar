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
