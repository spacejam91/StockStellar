# Scanner Parameters — portable brief

Paste this whole file into a fresh conversation or feed it to another app. It is
self-contained: no prior context needed. Machine-readable twin: `PARAMETERS.json`.

---

## Task

Daily cross-sectional screen over a CA + US equity universe. For each session, score
every eligible name on how **statistically unusual** it is versus the rest of that
day's universe, and return at most 3 names — or none.

**Output vocabulary.** `score_pct` is a 0–100 percentile of criteria strength within
the same session's eligible universe. `score_pct = 92` means only 8% of the universe
fired this hard on these criteria. It is not a probability of gain, not a forecast,
and not a recommendation. Do not label output "picks", "signals", "expected movers",
or "buys". The only legitimate probability-shaped number is a **measured base rate
from a point-in-time log, reported with n and standard error** (see Test 5).

---

## Step 1 — Hard gates (eligibility, never scored)

Score only names passing all of these. Keep them out of the score, or illiquid junk
ranks as "unusual" because nobody trades it.

| Gate | Rule |
|---|---|
| Liquidity | `median(close × volume, 20d) ≥ $2M (CA) / $5M (US)` |
| Price | `close ≥ 1.00` |
| History | `≥ 250 sessions` |
| Continuity | no calendar gap `> 3 days` between consecutive bars |
| Completeness | `atr14`, `stdev(ret,60d)`, `median(volume,20d)` all non-null |

---

## Step 2 — Inputs, by pillar

`atr14` = 14d mean of true range. `true_range = max(high−low, |high−prev_close|, |low−prev_close|)`.
Everything is in **own-history units** (own ATR, own σ) before any cross-sectional step.

**volume** (side-neutral)
- `rvol = volume / median(volume, 20d)`
- `dollar_vol_z = (close×volume − mean(close×volume,60d)) / stdev(close×volume,60d)`

**displacement** (flips by side)
- `ret_z = (close/prev_close − 1) / stdev(daily_return, 60d)`
- `gap_atr = (open − prev_close) / atr14`

**range** (side-neutral)
- `range_ratio = true_range / atr14`
- `bb_width_pct = percentile_rank(4×stdev(close,20d)/mean(close,20d), trailing 252d)`

**structure** (flips)
- `donchian_dir = 1[close > max(high,20d)₋₁] + 1[close > max(high,55d)₋₁] − 1[close < min(low,20d)₋₁] − 1[close < min(low,55d)₋₁]`
- `pct_52w_centered = (close − min(low,252d))/(max(high,252d) − min(low,252d)) − 0.5`
- `ma_dist_atr = (close − mean(close,50d)) / atr14`  ← volatility units, not percent

**relstrength** (flips)
- `rs_5d`, `rs_20d` = own return over that window minus benchmark return (TSX Composite for CA, S&P 500 for US)
- `rs_sector_20d` = own 20d return − median 20d return of same sector/date/market
  *This one separates "the sector moved" from "this name moved". Highest-value single input here.*

**confirmation** (flips)
- `vp_thrust = ((close−low)/(high−low) − 0.5) × 2 × log1p(max(rvol−1, 0))`
  *Signed in market direction, scaled by volume excess. This is what strips out
  reversal-wick names that score beautifully on volume and range alone: a heavy
  up-day closing on its low scores **negative** for the long side.*

**quality** (side-neutral) — direction in parentheses
`gross_margin (+)`, `gross_margin_delta_yoy (+)`, `fcf_margin (+)`, `roic (+)`, `net_debt_ebitda (−)`, `interest_coverage (+)`

**value** (side-neutral) — *direction is a reading convention, not a claim that cheap outperforms*
`pe_forward (−)`, `ev_ebitda (−)`, `ev_sales (−)`, `pb (−)`, `fcf_yield (+)`

**revisions** (flips)
`eps_rev_1m (+)`, `eps_rev_3m (+)`, `target_px_rev_1m (+)`, `net_upgrades_1m (+)`, `last_surprise_pct (+)`

### Flags — carried, never scored
`earnings_in_2d` · `macro_print_day` (CPI, payrolls, FOMC, BoC, StatCan) · `short_pct_float` · `days_to_cover` · `borrow_fee_pct` · `recent_filing` (8-K/6-K, Form 4/SEDI, 13D/G, index change, lockup) · `suspect_unadjusted_split` = `|ret| > 0.35 AND rvol < 1.5`

Short interest and borrow are deliberately unscored: **crowding cuts both ways**, so a
signed weight on them would be a directional claim rather than a measurement.
`macro_print_day` matters more than it looks — without it the screen "discovers" that
everything is unusual on CPI day.

---

## Step 3 — Normalize (cross-sectional, rank-based)

Within each `(date, market)` group — CA and US ranked **separately**, different
liquidity regimes:

1. Magnitude inputs rank on `|x|`; signed inputs rank on `x`.
2. `percentile = (rank − 0.5) / n` — never 0 or 1, so never ±∞.
3. `value = inverse_standard_normal_cdf(percentile)`.
4. Fundamental inputs: multiply by their direction (+1/−1) afterwards.

**No winsorization in the rank path** — clipping at the 1st/99th percentile preserves
order, so it cannot change a rank. (It cost 8× the runtime for zero effect.)
Rank-based on purpose: one halted name with RVOL 90 hijacks a raw-z composite.

---

## Step 4 — Combine

```
pillar_score  = mean of that pillar's available normalized members (skip missing)
raw_long      = weighted mean of all pillars                    (weights: all 1.0)
raw_short     = same, with the FLIPPING pillars sign-inverted    (neutral ones unchanged)
side          = 'long' if raw_long >= raw_short else 'short'
raw           = max(raw_long, raw_short)
```

Missing pillars drop out of **both** numerator and denominator so weights renormalize
— never score a blank as neutral.

**Then re-standardize. This step is mandatory:**

```
composite_z = inverse_normal((rank(raw within date,market) − 0.5)/n)
score_pct   = that percentile × 100
```

A weighted mean of k pillars has sd ≈ 1/√k, so thresholding the raw mean "in σ" is
wrong by a factor of √k. With 9 pillars a nominal 2.0 threshold is really ~6σ — in
testing that produced **3 picks in 150 sessions** before this step was added.

**Ceiling check.** `max_attainable_composite_z = inverse_normal((n−0.5)/n)`:

| universe n | 20 | 40 | 100 | 500 | 1000 | 5000 |
|---|---|---|---|---|---|---|
| ceiling | 1.96 | 2.16 | 2.58 | 3.09 | 3.24 | 3.66 |

A threshold above the ceiling returns nothing **forever** and looks identical to a
quiet market. Verify feasibility against universe size before blaming the criteria.

---

## Step 5 — Select

```
composite_z         >= 2.0          # relative: unusual vs today's universe
n_qualifiers        >= 2            # absolute: own-history conditions
    rvol >= 2.0 | |ret_z| >= 2.0 | range_ratio >= 1.5 | |gap_atr| >= 0.75
pillar_confirmation >= 0
then: top 3 by composite_z, max 1 per sector
```

**Why both floors.** `composite_z` is a re-ranked cross-section, so ~2.3% of *any*
universe clears 2.0σ every session — on 10,000 names that's 230 candidates and a
top-3 list always fills, making "unusual" meaningless. The absolute qualifiers are in
own-history units: on a quiet session nothing has RVOL 2 *and* a 2σ move, and the
honest output is **"no names cleared the threshold today."**

Target **≥ 20% empty sessions**. Below that, tighten — never loosen. The sector cap
matters: on a sector-wide move day an uncapped list hands back three copies of one
bet, formatted to look like three independent signals.

---

## Step 6 — Weights

**Equal across all pillars, by default.** Not laziness:

- Dawes (1979), *improper linear models* — unit weights routinely beat
  regression-fitted weights out of sample, because fitted weights absorb estimation error.
- DeMiguel, Garlappi & Uppal (2009) — naive 1/N beat optimized mean-variance out of
  sample across 14 datasets.

With ~250 sessions of log, any fitted weight vector is mostly noise wearing a decimal
point. Before tuning anything, run Test 7; if the top-3 is unstable, the fix is
**fewer pillars, not tuned weights**.

---

## Step 7 — The tests that decide whether any of this ranks anything

Requires a **point-in-time log**: one row per name per session, written once, never
edited. Recomputing history with today's universe, listings, or restated fundamentals
is survivorship + lookahead bias and will manufacture an edge that isn't there.

| # | Test | Pass criteria / how to read it |
|---|---|---|
| 1 | **IC** — Spearman(composite, forward return) per session, full cross-section, horizons 1/5/20d | Real cross-sectional equity signals sit at **0.02–0.05**. ~0.30 means a lookahead bug — check that forward returns start the session *after* the score. |
| 2 | **IC t-stat** = mean(IC)/(sd(IC)/√n) | **\|t\| ≥ 3.0** (Harvey, Liu & Zhu 2016 — 2.0 is not enough given how many predictors have been tested). Raise the bar for every variant tried, and count them. |
| 3 | **Hit rate − universe base rate**, same sessions | Edge must exceed 2 SE. Absolute hit rate is meaningless: in an up year everything hits. |
| 4 | **Decile monotonicity** — mean forward return by score decile | Rank correlation > 0.6. Top-bucket-only with a flat middle is usually curve-fit. |
| 5 | **Score-band base rates** — up-rate + mean return per 10-point band, with n and binomial SE | The *only* legitimate probability-shaped output. e.g. "95–100 band closed higher 5d later on 53.4% of occasions vs 50.0% universe base rate, n=876, SE 1.7pp". |
| 6 | **Cost drag** — gross minus round trip (spread/2 + slippage, in and out) | Default 35bps. In testing it ate **80% of the gross** on a daily-turnover 3-name list. |
| 7 | **Weight sensitivity** — perturb weights ±25%, 200 trials | Mean Jaccard overlap of top-3 > 0.8, else the ranking is noise. |
| 8 | **Walk-forward** — expanding window, report only on sessions after each window | Never quote an in-sample number. |
| 9 | **Empty-day share** | ≥ 20% of sessions. |
| 10 | **Null-data control** — run the whole pipeline on synthetic random-walk prices | Must yield mean IC ≈ 0, \|t\| < 2, edge inside 2 SE, non-monotone deciles. **Run this before trusting any positive result** — it proves the harness doesn't manufacture an edge. |

---

## Known limits — state these alongside any output

- **Horizon mismatch.** Documented factor premia (value, momentum, quality, low-vol)
  are measured on diversified, long-short, monthly-rebalanced portfolios over decades,
  and McLean & Pontiff (2016) found published anomaly returns decay materially
  post-publication. A 3-name daily list is a different statistical object —
  idiosyncratic variance dominates, so even a genuine premium is invisible at that
  concentration and horizon. Factor literature is **not** support for this tool.
- **Small universes** can't produce large `composite_z` (see the ceiling table). In a
  40-name test every pick printed the same ~2z because the top rank maps to a fixed percentile.
- **Regime dependence.** `bb_width_pct` and `ret_z` both use trailing vol, so they
  misread for ~60 sessions after a volatility regime shift.
- **CA data gaps are real**, not a wiring bug: effectively no usable options data
  (Montréal Exchange too thin for most names), thin-to-absent analyst coverage below
  mid-cap, lagged short interest (CIRO twice-monthly; FINRA bi-monthly for US). Check
  pillar null rates per market before comparing a CA score to a US one.
- **Unadjusted corporate actions** produce textbook 2σ moves with volume spikes. Audit first.
- **Back-adjusted history** (e.g. yfinance) is not point-in-time and will overstate
  any measured edge. Fine for building, not for the calibration log you'll trust.
