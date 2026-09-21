# StockStellar

A daily cross-sectional screen over a CA + US equity universe. It scores every
eligible name on how **statistically unusual** it is versus the rest of that
session's universe, and returns **at most three** — or none.

Empty days are the point, not a bug. Roughly a fifth of sessions should return
nothing: if fewer do, the bar is too loose and "unusual" has stopped meaning
anything. Tighten it, never loosen it.

## What the output means

`score_pct` is a 0–100 percentile of criteria strength within the same
session's eligible universe. `score_pct = 92` means only 8% of the universe
fired this hard on these criteria.

It is **not** a probability of gain, a forecast, or a recommendation. The only
legitimate probability-shaped number here is a measured base rate from the
point-in-time log, reported with n and a standard error — see
`/api/scan/base-rates`.

## Run it

```bash
uv run python -m scanner              # today's watchlist
uv run python -m scanner --report     # what the log says so far
uv run python -m scanner --diagnose   # gate rejections + input coverage
uv run python -m uvicorn app.main:app --reload   # web view at http://localhost:8000
```

## Market data

Set `MARKET_DATA` (see `.env.example`):

| value | what it is |
|---|---|
| `mock` | Deterministic offline bars. **Plants events** — dev data and a smoke test, *not* a null case. |
| `null` | Driftless random walk, volume independent of returns. Nothing to find. The Test 10 harness. |
| `signal` | Null plus a strong planted signal. The positive control. |
| `yahoo` | Real daily bars via yfinance. Free, unofficial, back-adjusted. |

Back-adjusted history is not point-in-time and will overstate any measured
edge. Fine for building; not for a calibration log you intend to trust.

## Before trusting any output

```bash
uv run python -m tests.test_null_control
```

Test 10 asks whether the pipeline manufactures an edge out of noise. It runs a
driftless random walk through the whole pipeline and requires mean IC ≈ 0, and
separately requires that a *planted* signal IS found — because a null control
passes trivially whenever the measuring apparatus is broken.

Known caveat: overlapping forward windows autocorrelate the IC series and
understate its standard error, inflating |t|. Add a Newey–West correction
before quoting a t-stat from real data.

## Layout

```
app/           web interface (read-only) + the bulk market-data seam
scanner/       gates, metrics, ranking, selection, evaluation, the log
tests/         Test 10 and its supporting controls
```

## Scope

StockStellar surfaces information. It has no broker connection, places no
orders, and knows nothing about an account.
