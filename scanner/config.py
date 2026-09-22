"""Scan configuration — every threshold from docs/PARAMETERS.md in one place.

Defaults here are the spec's defaults. Change them deliberately; several have
non-obvious interactions documented inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Pillars whose sign flips when scoring the short side. The rest are magnitude
# measures where "unusual" means the same thing in both directions.
FLIPPING_PILLARS = ("displacement", "structure", "relstrength", "confirmation", "revisions")
NEUTRAL_PILLARS = ("volume", "range", "quality", "value")

# input -> (pillar, is_magnitude). is_magnitude means rank on |x|, because a
# -3 sigma move is exactly as unusual as a +3 sigma one.
INPUTS: dict[str, tuple[str, bool]] = {
    "rvol":              ("volume", True),
    "dollar_vol_z":      ("volume", True),
    "ret_z":             ("displacement", False),
    "gap_atr":           ("displacement", False),
    "range_ratio":       ("range", True),
    "bb_width_pct":      ("range", True),
    "donchian_dir":      ("structure", False),
    "pct_52w_centered":  ("structure", False),
    "ma_dist_atr":       ("structure", False),
    "rs_5d":             ("relstrength", False),
    "rs_20d":            ("relstrength", False),
    "rs_sector_20d":     ("relstrength", False),
    "vp_thrust":         ("confirmation", False),
}


@dataclass
class ScanConfig:
    # --- Step 1: hard gates (eligibility, never scored) ---------------------
    min_dollar_vol_ca: float = 2_000_000.0
    min_dollar_vol_us: float = 5_000_000.0
    min_price: float = 1.00
    min_sessions: int = 250
    max_gap_days: int = 3

    # Three of the scored inputs -- gap_atr, range_ratio, ma_dist_atr -- divide
    # by ATR, so a name whose ATR is near zero produces enormous "unusualness"
    # out of a rounding error. LB.TO was published on 2026-09-21 with ATR14 of
    # $0.08 on a $40.66 close: a 17-cent gap read as +2.13 ATR and a 26-cent
    # range as 3.25x ATR, and it took a top slot. Nothing happened to that
    # stock; the denominator was broken.
    #
    # The far end is data corruption rather than quiet trading: an average true
    # range larger than half the share price is an unadjusted split or a bad
    # bar, not a market.
    #
    # Measured on 276,679 eligible rows: the floor removes 0.35% and the cap
    # 0.09%. The 5th percentile of atr14/close is 1.74%, so neither bound is
    # anywhere near an ordinary low-volatility large cap.
    min_atr_frac: float = 0.005     # ATR14 >= 0.5% of price
    max_atr_frac: float = 0.50      # ATR14 <= 50% of price

    # --- Step 5: selection --------------------------------------------------
    composite_threshold: float = 2.0
    min_qualifiers: int = 2
    max_picks: int = 3
    one_per_sector: bool = True
    require_confirmation: bool = True   # pillar_confirmation >= 0

    # Absolute, own-history qualifier conditions. These are what make empty days
    # possible at all: composite_z is a re-ranked cross-section, so ~2.3% of ANY
    # universe clears 2.0 sigma every single session. On 10,000 names that is 230
    # candidates and a top-3 list always fills, which would make "unusual"
    # meaningless. The qualifiers are in own-history units and genuinely do not
    # fire on a quiet day.
    # Tightened 2026-09-21 from the spec's 2.0 / 2.0 / 1.5 / 0.75. Those
    # defaults produced ~0-3% empty sessions on real data against the spec's
    # own >=20% target, and docs/PARAMETERS.md is explicit that below 20% the
    # bar is too loose and the score has stopped meaning "unusual" -- tighten,
    # never loosen. min_qualifiers stays at 2, so the "2 of 4" structure is
    # unchanged; only the levels moved.
    qual_rvol: float = 3.0
    qual_ret_z: float = 2.5
    qual_range_ratio: float = 2.0
    qual_gap_atr: float = 1.00

    # --- Qualifier scaling ---------------------------------------------------
    # A FIXED absolute threshold does not survive a change of universe size,
    # and this was measured, not assumed:
    #
    #     universe      eligible/session   qualifying/session   empty days
    #     400 tickers        ~190                ~2.4             25.4%
    #     full CA+US        2,877               21.6              2.2%
    #
    # Same thresholds, fifteen times the names, fifteen times as many clearing
    # any fixed bar -- so 21.6 names qualify for 3 slots and an empty day
    # becomes impossible. Raising the levels by hand just re-fits them to one
    # universe size and breaks again when the universe changes.
    #
    # In "scaled" mode each threshold is instead the quantile of that metric's
    # own TRAILING distribution that leaves `target_qualifiers` names expected
    # to clear it, whatever the universe size:
    #
    #     threshold = trailing_quantile(metric, 1 - target_qualifiers/n_eligible)
    #
    # Trailing, not today's cross-section. That distinction is the whole point:
    # a bar set from today's own names is a percentile and can never be empty,
    # while a bar set from the past 60 sessions is one today may simply fail to
    # reach. Quiet day, nothing clears it, empty list.
    qualifier_mode: str = "scaled"          # "scaled" | "absolute"
    # 5, not 6. Measured across three universe sizes (2,882 / 743 / 128
    # eligible), 6 lands at 17.7% / 15.6% / 30.2% empty and misses the spec's
    # >=20% floor at two of them; 5 gives 21.9% / 20.8% / 31.2% and clears it
    # everywhere while keeping the most names (1.07-1.57 picks/session).
    # 4 also clears but costs picks for no benefit.
    target_qualifiers: int = 5              # expected names clearing the bar
    qualifier_lookback: int = 60            # trailing sessions for the baseline
    # Bounds on the scaled level, both relative to the spec's absolute one.
    # The floor stops a stretch of dead sessions dragging the bar down to noise.
    # The ceiling stops the opposite: at these quantiles one violent session
    # supplies the entire tail of the 60-session window, and without a cap the
    # bar stays above anything the market produces for the next three months --
    # three months of empty days caused by history rather than by today.
    qualifier_floor_frac: float = 0.5       # never below this x the absolute level
    qualifier_ceil_frac: float = 2.0        # never above this x the absolute level

    # --- Step 6: weights ----------------------------------------------------
    # Equal by default, and that is a considered choice, not laziness. Dawes
    # (1979) on improper linear models and DeMiguel/Garlappi/Uppal (2009) on 1/N
    # both find unit weights beat fitted ones out of sample, because fitted
    # weights absorb estimation error. With ~250 sessions of log any fitted
    # vector is mostly noise. If the top-3 is unstable, the fix is fewer
    # pillars, not tuned weights.
    weights: dict[str, float] = field(default_factory=dict)

    # --- Health -------------------------------------------------------------
    # Below this share of empty sessions the threshold is too loose and the
    # score has stopped meaning "unusual". Tighten, never loosen.
    empty_day_target: float = 0.20

    def weight_for(self, pillar: str) -> float:
        return self.weights.get(pillar, 1.0)

    def min_dollar_vol(self, market: str) -> float:
        return self.min_dollar_vol_ca if str(market).upper() == "CA" else self.min_dollar_vol_us
