"""StockStellar scanner — daily cross-sectional screen for unusual names.

Pipeline, in spec order:
    metrics.compute   Step 2  per-name inputs in own-history units
    gates.apply       Step 1  eligibility + absolute qualifier count
    rank.normalize    Step 3  cross-sectional rank -> z within (date, market)
    combine.combine   Step 4  pillars -> side -> composite_z / score_pct
    select.select     Step 5  threshold, then rank, then cap at 3
    store.write_scan          point-in-time log in stockstellar.db

Full criteria spec: ~/Documents/Claude/Outputs/market-scanner/PARAMETERS.md

What the output means: score_pct is a 0-100 percentile of criteria strength
within the same session's eligible universe. It is not a probability, not a
forecast and not a recommendation. The only probability-shaped number here is
store.score_band_base_rates(), which measures what actually followed each score
band from the log, with n and standard errors.
"""

from scanner.config import ScanConfig
from scanner.pipeline import ScanResult, format_watchlist, run_scan

__all__ = ["ScanConfig", "ScanResult", "run_scan", "format_watchlist"]
