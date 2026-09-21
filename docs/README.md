# StockStellar — scanner documentation

Drop this folder into the repo as `docs/`:

```bash
cp -R ~/Documents/Claude/Outputs/stockstellar-docs ~/Documents/Claude/Projects/stockstellar/docs
```

| File | What it is |
|---|---|
| `PARAMETERS.md` | The criteria spec. Every gate, input formula, normalisation step, pillar, threshold and validation test. Self-contained — paste it into a fresh session and it can implement the scanner with no other context. |
| `PARAMETERS.json` | The same spec, machine-readable: 29 inputs, 9 pillars, 5 gates, 7 unscored flags, 10 tests. For feeding to code or another tool rather than a reader. |
| `HANDOFF.md` | Verified defects in the current code, with fixes. Each one was checked against the files on disk and survived an adversarial review pass. |

## The one thing to get right

`score_pct` is a 0–100 percentile of criteria strength within that session's eligible
universe. 92 means only 8% of the universe fired this hard on these criteria.

It is **not** a probability of gain, not a forecast, and not a recommendation. The only
probability-shaped output this system is allowed to produce is a **measured base rate
from the point-in-time log**, reported with its sample size and standard error — what
actually followed each score band, never what is expected to follow.

Two consequences that are easy to break by accident:

1. **Empty days are a real output.** Selection needs both floors — the relative one
   (`composite_z >= 2.0`) *and* the absolute own-history one (`>= 2` of: RVOL ≥ 2,
   |ret_z| ≥ 2, range ≥ 1.5×ATR, |gap| ≥ 0.75×ATR). The relative floor alone can never
   produce an empty day, because ~2.3% of *any* universe clears 2σ every session.
2. **Measurements must not mix data providers.** Synthetic providers generate random
   walks. Mixing their rows into a base rate drags it toward 50% while shrinking the
   standard error — a confident-looking null that is an artifact of the test data.
