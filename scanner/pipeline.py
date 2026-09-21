"""Orchestration: provider -> metrics -> gates -> rank -> combine -> select -> log.

One entry point, `run_scan()`. The FastAPI routes, the CLI and any future
scheduled job all call this, so there is exactly one definition of what a scan is.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from scanner import combine, gates, metrics, rank, select, store
from scanner.config import ScanConfig

log = logging.getLogger(__name__)

# Indicators need a year of history before the first scorable session
# (52-week range, 252d Bollinger percentile), plus the 250-session gate.
WARMUP_SESSIONS = 300


@dataclass
class ScanResult:
    as_of: pd.Timestamp
    provider: str
    picks: pd.DataFrame
    summary: pd.DataFrame
    scored: pd.DataFrame
    health: dict
    run_ids: list[int]

    @property
    def is_empty_day(self) -> bool:
        today = self.picks[self.picks["date"] == self.as_of] if len(self.picks) else self.picks
        return len(today) == 0

    def today(self) -> pd.DataFrame:
        return self.picks[self.picks["date"] == self.as_of] if len(self.picks) else self.picks


def run_scan(*, provider=None, cfg: ScanConfig | None = None, as_of=None,
             history_days: int = 0, persist: bool = True,
             fundamentals: pd.DataFrame | None = None,
             fundamental_directions: dict[str, int] | None = None) -> ScanResult:
    """Score one session (or `history_days` extra sessions, to build the log).

    Scoring is restricted to the target sessions on purpose: indicators need the
    full history, but a live daily run only needs today's cross-section, which
    is the difference between seconds and minutes on a 10k-name universe.
    """
    cfg = cfg or ScanConfig()
    if provider is None:
        from app.market_data import provider as default_provider
        provider = default_provider

    bars = provider.daily_bars()
    if bars.empty:
        raise RuntimeError(f"provider {provider.name!r} returned no bars")
    bars["date"] = pd.to_datetime(bars["date"])
    universe_size = int(bars["ticker"].nunique())

    try:
        bench = provider.benchmarks()
    except Exception as e:  # a missing benchmark must not kill the scan
        log.warning("benchmark fetch failed (%s) -- relstrength will be partial", e)
        bench = None
    if bench is not None and not bench.empty:
        bench["date"] = pd.to_datetime(bench["date"])

    df = metrics.compute(bars, benchmarks=bench)
    df = gates.apply(df, cfg)

    sessions = pd.DatetimeIndex(sorted(df["date"].unique()))
    target = pd.Timestamp(as_of) if as_of is not None else sessions.max()
    if target not in sessions:
        earlier = sessions[sessions <= target]
        if len(earlier) == 0:
            raise ValueError(f"no sessions on or before {target.date()}")
        log.info("%s is not a session; scoring %s instead", target.date(), earlier.max().date())
        target = earlier.max()

    upto = sessions[sessions <= target]
    score_dates = upto[-(history_days + 1):]
    if len(upto) < WARMUP_SESSIONS:
        log.warning("only %d sessions of history; the %d-session gate and the 52-week "
                    "inputs need ~%d before scores mean anything",
                    len(upto), cfg.min_sessions, WARMUP_SESSIONS)

    window = df[df["date"].isin(score_dates)]
    window = rank.normalize(window)
    if fundamentals is not None and len(fundamentals) and fundamental_directions:
        window = window.merge(fundamentals, on=["ticker", "date"], how="left")
        window = rank.normalize_fundamentals(window, fundamental_directions)

    scored = combine.combine(window, cfg)
    picks = select.select(scored, cfg)
    summary = select.day_summary(scored, picks, cfg)
    hlth = select.health(summary, cfg)

    run_ids = []
    if persist and not scored.empty:
        cfg_dict = dataclasses.asdict(cfg)
        for d in pd.DatetimeIndex(sorted(scored["date"].unique())):
            run_ids.append(store.write_scan(
                as_of=d, scored=scored, picks=picks, summary=summary,
                config=cfg_dict, universe_size=universe_size, provider=provider.name))

    return ScanResult(as_of=target, provider=provider.name, picks=picks, summary=summary,
                      scored=scored, health=hlth, run_ids=run_ids)


def format_watchlist(result: ScanResult) -> str:
    """Plain-text watchlist, for the CLI and for notification bodies."""
    lines = ["=" * 68, f"  WATCHLIST {result.as_of.date()}   [{result.provider}]", "=" * 68]
    day = result.summary[result.summary["date"] == result.as_of]
    if len(day):
        r = day.iloc[0]
        lines.append(f"  eligible {int(r['n_eligible']):,}   over threshold "
                     f"{int(r['n_over_threshold'])}   qualified {int(r['n_qualified'])}"
                     f"   ceiling {r['ceiling_z']}")
    today = result.today()
    if today.empty:
        lines += ["", "  No names cleared the threshold today.", "",
                  "  That is an output, not a failure: nothing in the universe was",
                  "  statistically unusual enough on the configured criteria."]
    else:
        for r in today.itertuples(index=False):
            lines.append("")
            lines.append(f"  {r.rank}. {r.ticker}  [{r.market}/{r.sector}]  {r.side.upper()}")
            lines.append(f"     score {r.score_pct:.0f}/100   composite {r.composite_z:+.2f}z"
                         f"   qualifiers {r.n_qualifiers}/4")
            lines.append(f"     close {r.close:.2f}   ATR14 {r.atr14}   RVOL {r.rvol}")
            lines.append(f"     drivers: {r.drivers}")
            if r.flags:
                lines.append(f"     FLAGS: {r.flags}")
        lines += ["", "  score = percentile of criteria strength in today's universe.",
                  "  Not a probability, not a forecast, not a recommendation."]
    lines.append("=" * 68)
    return "\n".join(lines)
