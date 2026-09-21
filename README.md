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
| `yahoo` | Real daily bars via yfinance. Free, unofficial, back-adjusted. **Cannot serve a full 8,700-name universe** — rate-limits even from a residential IP. Fine for the ~2,800-name Canadian half. |
| `polygon` | Whole US market per request, end-of-day, free tier, works from a datacenter IP. Needs `POLYGON_API_KEY`. US only. |

Back-adjusted history is not point-in-time and will overstate any measured
edge. Fine for building; not for a calibration log you intend to trust.

## What the back-test says

Measured on 85 sessions of real data, 258,651 logged rows, 3,318 names:

| horizon | signed IC | \|t\| | decile monotonicity | edge vs universe |
|---|---|---|---|---|
| 1d | −0.0110 | 1.32 | −0.53 | −1.36 pp (inside 2 SE) |
| 5d | −0.0231 | 3.04 | −0.85 | **−2.70 pp (outside 2 SE)** |
| 20d | −0.0309 | 4.25 | −0.88 | **−3.75 pp (outside 2 SE)** |

The selected names **underperform** the same sessions' universe in the direction
`side` calls, and more so the longer the horizon. The decile table is monotone
across all 258k rows. Measured base rates say the same thing: the 90–100 score
band closed higher 47.4% of the time (n=26,156, SE 0.3pp) against 49.6% for the
0–10 band.

This is not a harness bug — the null control returns IC ≈ 0 on random walks
across five seeds, and the positive control detects a planted signal at t=+7.4.
The likeliest reading is that the criteria select the population the literature
says underperforms (extreme move, extreme volume, high attention) and `side`
then follows the move into its reversal.

**Read `side` as which tail the move was in, not as a direction to trade.** And
do not invert it on this evidence: one 85-session window, back-adjusted data
that is not point-in-time, and a |t| inflated by overlapping forward windows.

    uv run python scripts/backtest.py --horizon 5

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
