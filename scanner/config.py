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
    qual_rvol: float = 2.0
    qual_ret_z: float = 2.0
    qual_range_ratio: float = 1.5
    qual_gap_atr: float = 0.75

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
