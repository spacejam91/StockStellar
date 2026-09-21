"""SIC code -> sector label.

Sectors are used for exactly two things here, and that shapes the design:

  1. rs_sector_20d -- a name's 20d return minus the MEDIAN 20d return of its own
     sector that session. Buckets must hold enough names for a median to mean
     something.
  2. A max-one-pick-per-sector cap, whose job is to stop the daily shortlist
     being three companies that are really the same bet.

Neither needs GICS fidelity, and neither survives the wrong granularity. SIC's
own ten divisions are far too coarse: "Manufacturing" (2000-3999) would hold
pharma, semiconductors and automakers in one bucket, which makes the median
meaningless and lets the cap reject genuinely independent names. Four-digit SIC
is the opposite failure -- hundreds of buckets with three members each.

So the rules below sit mostly at the 2-digit major-group level and drop to 3-
or 4-digit only where a major group straddles a real sector boundary. Measured
counts in our 6,226-name universe made those splits obvious:

    SIC 28  724 names   chemicals AND pharma/biotech  -> must split
    SIC 73  576 names   business services, mostly software
    SIC 67  487 names   holding/investment offices, mostly REITs and funds
    SIC 60  365 names   banks
    SIC 38  257 names   instruments, incl. medical devices -> must split
    SIC 36  255 names   electronics, incl. semiconductors -> must split

RULES ARE ORDERED AND FIRST MATCH WINS. Specific ranges come before the general
major group they sit inside; reordering them silently refiles whole industries.

Known to be wrong, because every SIC crosswalk is:
  * SIC is SELF-REPORTED and often stale -- a company files under what it was
    at registration, not what it is now.
  * Conglomerates get one code for a diversified business.
  * Modern business models fit badly. Retailers that are really logistics or
    advertising businesses, and "business services" (73) absorbing companies
    with little in common beyond being hard to classify.
  * 6770 "blank checks" holds SPACs, which are not an industry.
Treat a sector here as a rough grouping key, not a statement about the company.
"""

from __future__ import annotations

import math

# (low, high, sector) inclusive, FIRST MATCH WINS.
RULES: list[tuple[int, int, str]] = [
    # --- carved out of major groups that straddle a sector boundary ---
    (2833, 2836, "Health Care"),        # medicinal, pharma prep, biologicals
    (3826, 3826, "Health Care"),        # lab analytical instruments
    (3841, 3851, "Health Care"),        # surgical, medical, dental, ophthalmic
    (8731, 8731, "Health Care"),        # commercial physical & biological research
    (3570, 3579, "Technology"),         # computer & office equipment
    (3674, 3674, "Technology"),         # semiconductors
    (3661, 3669, "Technology"),         # communications equipment
    (3827, 3827, "Technology"),         # optical instruments
    (7370, 7379, "Technology"),         # software, data processing, IT services
    (6798, 6798, "Real Estate"),        # REITs
    (6770, 6770, "Financials"),         # blank checks / SPACs
    (6722, 6726, "Financials"),         # investment offices, closed-end funds

    # --- major groups ---
    (100, 999, "Consumer Staples"),     # agriculture production
    (1000, 1119, "Materials"),          # metal mining
    (1200, 1299, "Energy"),             # coal
    (1300, 1399, "Energy"),             # oil & gas extraction
    (1400, 1499, "Materials"),          # nonmetallic minerals
    (1500, 1799, "Industrials"),        # construction
    (2000, 2199, "Consumer Staples"),   # food, tobacco
    (2200, 2399, "Consumer Discretionary"),   # textiles, apparel
    (2400, 2599, "Industrials"),        # lumber, furniture
    (2600, 2699, "Materials"),          # paper
    (2700, 2799, "Communication"),      # printing & publishing
    (2800, 2899, "Materials"),          # chemicals (pharma carved out above)
    (2900, 2999, "Energy"),             # petroleum refining
    (3000, 3299, "Materials"),          # rubber, plastics, leather, stone/glass
    (3300, 3399, "Materials"),          # primary metals
    (3400, 3499, "Industrials"),        # fabricated metal
    (3500, 3599, "Industrials"),        # industrial machinery
    (3600, 3699, "Industrials"),        # electrical equipment
    (3700, 3799, "Industrials"),        # transportation equipment
    (3800, 3899, "Industrials"),        # instruments
    (3900, 3999, "Consumer Discretionary"),   # misc manufacturing
    (4000, 4499, "Transport"),          # rail, transit, trucking, water
    (4500, 4599, "Transport"),          # air
    (4600, 4699, "Energy"),             # pipelines
    (4700, 4799, "Transport"),          # transportation services
    (4800, 4899, "Communication"),      # telecom, broadcasting, cable
    (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"),        # wholesale
    (5200, 5399, "Consumer Discretionary"),
    (5400, 5499, "Consumer Staples"),   # food stores
    (5500, 5799, "Consumer Discretionary"),
    (5800, 5899, "Consumer Discretionary"),   # eating & drinking
    (5900, 5999, "Consumer Discretionary"),
    (6000, 6199, "Financials"),         # banks, credit
    (6200, 6299, "Financials"),         # brokers
    (6300, 6499, "Insurance"),
    (6500, 6599, "Real Estate"),
    (6700, 6799, "Financials"),         # holding & other investment offices
    (7000, 7099, "Consumer Discretionary"),   # hotels
    (7200, 7299, "Consumer Discretionary"),   # personal services
    (7300, 7399, "Industrials"),        # business services not caught above
    (7500, 7699, "Consumer Discretionary"),   # auto & repair services
    (7800, 7999, "Communication"),      # motion pictures, amusement
    (8000, 8099, "Health Care"),        # health services
    (8100, 8399, "Consumer Discretionary"),   # legal, education, social
    (8400, 8999, "Industrials"),        # engineering, accounting, research
]

UNCLASSIFIED = {9995, 9997, 9999, 6199}


def sic_to_sector(sic) -> str | None:
    """Map an SIC code to a sector label, or None when unknown.

    None is deliberate and meaningful: selection treats an unknown sector as an
    ABSENCE of information rather than a shared bucket, so unknown-sector names
    are not capped against each other. Returning a catch-all string here would
    silently recreate the bug where every unclassified US name collided in one
    bucket and the max-3 list could only ever return one name.
    """
    if sic is None:
        return None
    try:
        if isinstance(sic, float) and math.isnan(sic):
            return None
        code = int(sic)
    except (TypeError, ValueError):
        return None
    if code <= 0 or code in UNCLASSIFIED:
        return None
    for lo, hi, sector in RULES:
        if lo <= code <= hi:
            return sector
    return None


def sectors() -> list[str]:
    return sorted({s for _, _, s in RULES})
