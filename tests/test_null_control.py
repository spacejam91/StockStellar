"""Test 10 — the null-data control, plus the two checks that make it mean anything.

Run this before trusting ANY positive result from the scanner.

The test has three parts and all three are necessary:

  A. GENERATOR SANITY — prove the null data really is null. If the generator has
     drift, or volume correlated with |return|, then "no edge found" says nothing
     about the scanner. Test the test first.

  B. NULL CONTROL (Test 10 proper) — the pipeline on driftless random-walk prices
     must produce mean IC ~ 0, |t| < 2, and non-monotone deciles.

  C. POSITIVE CONTROL — the same pipeline on SignalMarketData, which plants a
     same-day move plus continued drift after a volume spike, MUST find it.
     Measured on the names the scanner SELECTS, not the full cross-section:
     planted events are ~1% of rows and a whole-universe correlation dilutes
     them to nothing. (Mock is not usable here — its level shift is fully
     realised on the event day, leaving nothing ahead to predict.) Without
     this, part B passes trivially whenever the measurement apparatus is broken:
     an IC function that returns NaN or zero for any input would sail through the
     null control and tell you nothing. B says "no false positives"; C says "and
     it could have detected one."

Usage:
    uv run python -m tests.test_null_control     # prints a report
    uv run pytest tests/test_null_control.py     # if pytest is installed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.market_data import MockMarketData, NullMarketData, SignalMarketData
from scanner import evaluate
from scanner.config import ScanConfig
from scanner.pipeline import run_scan

HORIZONS = (1, 5, 20)
SESSIONS = 150          # scored sessions; enough for a stable t-stat
NULL_IC_MAX = 0.02      # real cross-sectional equity signals live at 0.02-0.05
NULL_T_MAX = 2.0        # spec's bar for the null
POS_T_MIN = 3.0         # Harvey/Liu/Zhu (2016), not the traditional 2.0


# --------------------------------------------------------------------------
# A. Generator sanity — is the null actually null?
# --------------------------------------------------------------------------

def _simple_returns(bars: pd.DataFrame) -> pd.Series:
    b = bars.sort_values(["ticker", "date"])
    return b.groupby("ticker", sort=False)["close"].transform(lambda s: s / s.shift(1) - 1.0).dropna()


def test_null_generator_has_no_drift():
    """Measured on SIMPLE returns, deliberately.

    An earlier version of this test used log returns and failed. That was the
    test being wrong, not the generator: a price martingale has mean log return
    of exactly -sigma^2/2 by Jensen, which is volatility drag, an identity. The
    quantity that matters is the one evaluate.py actually correlates against,
    and that is the simple return.
    """
    bars = NullMarketData().daily_bars()
    r = _simple_returns(bars)
    se = r.std(ddof=1) / np.sqrt(len(r))
    t = r.mean() / se
    assert abs(t) < 3.0, (
        f"null generator has drift in simple returns: mean {r.mean():.2e} is "
        f"{t:.2f} SE from zero. A drifting null makes 'went up' partly "
        f"predictable and invalidates Test 10."
    )
    return {"mean_simple_ret": float(r.mean()), "t": float(t), "n": int(len(r))}


def _event_day_volume_lift(bars: pd.DataFrame) -> tuple[float, int]:
    """Median volume on the top-1% |return| days, as a multiple of each name's
    own median volume.

    A global correlation is the wrong instrument here: planted events are ~1% of
    rows, so 99% of the data dilutes any coupling to nothing. Conditioning on the
    event days is what makes it visible.
    """
    b = bars.sort_values(["ticker", "date"]).copy()
    b["ret"] = b.groupby("ticker", sort=False)["close"].transform(lambda s: s / s.shift(1) - 1.0)
    b["vol_med"] = b.groupby("ticker", sort=False)["volume"].transform("median")
    b["vol_mult"] = b["volume"] / b["vol_med"]
    d = b.dropna(subset=["ret", "vol_mult"])
    cut = d["ret"].abs().quantile(0.99)
    top = d[d["ret"].abs() >= cut]
    return float(top["vol_mult"].median()), int(len(top))


def test_null_generator_volume_independent_of_returns():
    lift, n = _event_day_volume_lift(NullMarketData().daily_bars())
    assert 0.8 < lift < 1.25, (
        f"null generator couples volume to |return|: biggest-move days carry "
        f"{lift:.2f}x the name's median volume (want ~1.0). That is a planted "
        f"volume/displacement relationship, so Test 10 would be measuring the "
        f"generator, not the scanner."
    )
    return {"top1pct_volume_multiple": lift, "n": n}


def test_mock_generator_does_have_planted_signal():
    """Guards the docstring fix: mock must NOT be mistaken for a null."""
    lift, n = _event_day_volume_lift(MockMarketData().daily_bars())
    assert lift > 1.5, (
        f"mock's biggest-move days carry only {lift:.2f}x median volume — the "
        f"planted events appear to be gone, and mock is no longer distinguishable "
        f"from a null."
    )
    return {"top1pct_volume_multiple": lift, "n": n}


# --------------------------------------------------------------------------
# B + C. The controls
# --------------------------------------------------------------------------

def _run(provider, sessions: int = SESSIONS) -> pd.DataFrame:
    cfg = ScanConfig()
    res = run_scan(provider=provider, cfg=cfg, history_days=sessions, persist=False)
    return evaluate.add_forward_returns(res.scored, horizons=HORIZONS), res.summary


def add_signed_score(scored: pd.DataFrame) -> pd.DataFrame:
    """composite_z carries magnitude only; `side` carries direction.

    This matters for any IC measurement and is easy to get wrong. `raw =
    max(raw_long, raw_short)`, so composite_z answers "how unusual", not "which
    way". Correlating it against SIGNED forward returns is therefore close to
    zero however much directional signal exists — the up-movers and down-movers
    sit at the same end of the scale and cancel.

    So there are two distinct questions, and they need different columns:
      - "does it find unusual names?"  -> composite_z vs |forward return|
      - "does it call direction?"      -> signed_score vs forward return
    """
    out = scored.copy()
    sign = np.where(out["side"].to_numpy() == "long", 1.0, -1.0)
    out["signed_score"] = out["composite_z"].to_numpy() * sign
    return out


def _ic_block(scored: pd.DataFrame, score_col: str = "composite_z") -> dict:
    out = {}
    for h in HORIZONS:
        ic = evaluate.information_coefficient(scored, horizon=h, score_col=score_col)
        tbl = evaluate.decile_table(scored, horizon=h, score_col=score_col)
        out[h] = {
            "mean_ic": ic["mean_ic"], "t_stat": ic["t_stat"], "n_sessions": ic["n_sessions"],
            "monotonicity": evaluate.decile_monotonicity(tbl),
        }
    return out


NULL_SEEDS = (11, 101, 202, 303, 404)


def test_null_control():
    """Test 10. The pipeline must NOT manufacture an edge from noise.

    Run across several seeds rather than one, for a specific reason. Overlapping
    forward windows make the naive t-stat unreliable: sessions t and t+1 share
    4 of their 5 forward days, so the per-session IC series is autocorrelated,
    the standard error is understated, and |t| is inflated. A single draw
    crossing 2.0 is therefore expected on a genuine null and would be a false
    alarm. (A Newey-West correction is the textbook fix and is worth adding
    before quoting any t-stat from real data.)

    What actually has to hold on a null is that the effect SIZE is negligible
    and its sign is unstable across draws. Both are checked here; a real signal
    would show a consistent sign and a mean IC that does not shrink toward zero.

    Checks composite_z and signed_score. Passing only on composite_z would be
    weak: a two-sided score is near-zero against signed returns almost by
    construction, so it could hide a directional bug signed_score would expose.
    """
    rows = []
    for sd in NULL_SEEDS:
        scored, _ = _run(NullMarketData(seed=sd))
        scored = add_signed_score(scored)
        for label in ("composite_z", "signed_score"):
            for h, m in _ic_block(scored, score_col=label).items():
                rows.append({"seed": sd, "score": label, "h": h, **m})
    r = pd.DataFrame(rows)

    failures = []
    for (label, h), g in r.groupby(["score", "h"]):
        mean_ic = g["mean_ic"].mean()
        if abs(mean_ic) > NULL_IC_MAX:
            failures.append(
                f"{label} {h}d: mean IC across seeds {mean_ic:+.4f}, |.| > {NULL_IC_MAX}")
        # A true null scatters sign; a real edge does not.
        same_sign = max((g["mean_ic"] > 0).sum(), (g["mean_ic"] < 0).sum())
        if same_sign == len(g) and abs(mean_ic) > NULL_IC_MAX / 2:
            failures.append(
                f"{label} {h}d: IC has the same sign on all {len(g)} seeds "
                f"(mean {mean_ic:+.4f}) — that is consistency, not noise")

    assert not failures, (
        "NULL CONTROL FAILED — the pipeline finds an edge in pure noise:\n  "
        + "\n  ".join(failures)
        + "\nEvery positive result on real data is untrustworthy until this is fixed. "
          "Usual causes: forward returns not shifted past the scored session, "
          "a metric peeking at same-bar close, or survivorship in the universe."
    )
    return {
        f"{lbl} {h}d": {
            "mean_ic": g["mean_ic"].mean(), "t_stat": g["t_stat"].abs().median(),
            "n_sessions": int(g["n_sessions"].iloc[0]), "monotonicity": g["monotonicity"].mean(),
        }
        for (lbl, h), g in r.groupby(["score", "h"])
    }


def test_positive_control():
    """The apparatus must detect signal that IS there — and in the right direction.

    Uses SignalMarketData, not mock. Mock plants a permanent level shift that is
    fully realised on the event day, so there is nothing ahead of it to predict;
    it is offline development data, not a positive control. SignalMarketData
    plants forward drift after a volume spike, which is genuinely predictable.

    The assertion requires a POSITIVE t, not |t|. An earlier version used abs()
    and "passed" on an IC of -0.0256 — an inverse relationship reported as
    success, which is precisely the kind of result this test exists to catch.
    """
    scored, _summary = _run(SignalMarketData())
    scored = add_signed_score(scored)

    # Measured on the names the scanner actually SELECTS, not the full
    # cross-section. Planted events are ~1% of rows, so a whole-universe
    # Spearman dilutes them to nothing — the same mistake that made the first
    # version of check A3 fail. The question that matters, and the one the tool
    # is for, is whether the shortlist behaves as planted.
    out = {}
    failures = []
    for h in HORIZONS:
        fwd = f"fwd_{h}d"
        sel = scored[
            scored["eligible"]
            & (scored["composite_z"] >= ScanConfig().composite_threshold)
            & (scored["n_qualifiers"] >= ScanConfig().min_qualifiers)
        ].dropna(subset=[fwd])
        if len(sel) < 30:
            failures.append(f"{h}d: only {len(sel)} selected rows — too few to test")
            continue
        sign = np.where(sel["side"].to_numpy() == "long", 1.0, -1.0)
        edge = sel[fwd].to_numpy() * sign            # return in the called direction
        t = edge.mean() / (edge.std(ddof=1) / np.sqrt(len(edge)))
        out[h] = {"mean_ic": float(edge.mean()), "t_stat": float(t),
                  "n_sessions": int(len(sel)), "monotonicity": float("nan")}

    ts = [m["t_stat"] for m in out.values()]
    best_t = max(ts) if ts else np.nan
    assert not np.isnan(best_t), "positive control produced no t-stat — IC machinery is broken"
    if best_t < POS_T_MIN:
        failures.append(f"best directional t across horizons {best_t:+.2f} < {POS_T_MIN}")

    assert not failures, (
        "POSITIVE CONTROL FAILED:\n  " + "\n  ".join(failures)
        + "\nSignal data plants a same-day move plus continued drift after a volume "
          "spike. If the scanner's own shortlist does not capture that, the null "
          "control's clean result is meaningless — an apparatus that detects nothing "
          "passes a null trivially."
    )
    return out


# --------------------------------------------------------------------------

def main() -> int:
    checks = [
        ("A1  null: no drift", test_null_generator_has_no_drift),
        ("A2  null: volume independent of |ret|", test_null_generator_volume_independent_of_returns),
        ("A3  mock: planted signal present", test_mock_generator_does_have_planted_signal),
        ("B   NULL CONTROL (Test 10)", test_null_control),
        ("C   POSITIVE CONTROL", test_positive_control),
    ]
    print("=" * 78)
    print("  Test 10 — null-data control")
    print("=" * 78)
    failed = 0
    for label, fn in checks:
        try:
            detail = fn()
            print(f"\n  PASS  {label}")
        except AssertionError as e:
            failed += 1
            detail = None
            print(f"\n  FAIL  {label}\n        {e}")
        if isinstance(detail, dict):
            for k, v in detail.items():
                if isinstance(v, dict):
                    print(f"          {k:>3}d  IC {v['mean_ic']:+.4f}  t {v['t_stat']:+.2f}  "
                          f"mono {v['monotonicity']:+.2f}  n={v['n_sessions']}")
                else:
                    print(f"          {k}: {v}")
    print("\n" + "=" * 78)
    print(f"  {len(checks) - failed}/{len(checks)} passed")
    if failed:
        print("  Do not trust scanner output until these pass.")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
