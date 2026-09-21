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

So the rules sit mostly at the 2-digit major-group level and drop to 3- or
4-digit only where a major group straddles a real sector boundary. The measured
counts that forced each split are noted inline.

RULES ARE ORDERED AND FIRST MATCH WINS. Specific ranges come before the general
major group they sit inside; reordering them silently refiles whole industries.
Three that will bite if moved: 3559 sits inside the 35xx machinery range, 3827
inside 38xx instruments, and 6199/6722-6726 inside the 61xx/67xx finance ranges.

Also: SIC codes under 1000 are stored WITHOUT a leading zero (agriculture is
100, not 0100), so parse and compare as int. A zero-padded string comparison
misfiles every agricultural name.

Known to be wrong, because every SIC crosswalk is. Specific cases you will
notice:
  * SIC 7389 "Business Services, NEC" is a junk drawer holding Visa, Mastercard,
    PayPal, Uber, DoorDash and eBay. They land in Commercial & Professional
    Services. Visa is not a financial here, and that is visibly odd -- but 7389
    also holds genuine business-services companies, so the code cannot be
    reassigned wholesale.
  * Airbnb is SIC 7340, "Services-To Dwellings & Other Buildings" -- literally a
    building-cleaning code. The single most absurd entry in the universe.
  * Netflix is SIC 7841, "Video Tape Rental". It lands in Media & Telecom, which
    is the right answer for a thoroughly obsolete reason.
  * Amazon is 5961 "Catalog & Mail-Order Houses". AWS is invisible to SIC.
  * Booking Holdings is 4700 "Transportation Services", so it sits with railroads.
  * SIC is SELF-REPORTED and often stale: a registrant files under what it was at
    registration. Conglomerates get one code for a diversified business.
Treat a sector here as a rough grouping key, not a statement about the company.
"""

from __future__ import annotations

import math

# (low, high, sector) inclusive, FIRST MATCH WINS.
#
# Granularity is chosen for the two jobs above, not for fidelity to GICS. SIC's
# own ten divisions are useless here -- "Manufacturing" (2000-3999) would hold
# pharma, semiconductors and automakers together. Fama-French 12 is the tempting
# free standard and is also wrong for job (b): measured on this universe its
# "Money" bucket holds 1,333 names, mixing banks, insurers, REITs and SPACs, so
# a shortlist could carry a regional bank and an apartment REIT and call them
# independent bets. On a rate-move day they are the same bet.
#
# So: FF12's skeleton, with Money split four ways, health split two, tech split
# three, and FF12's large "Other" dissolved into Transport, Media, Industrials
# and Commercial Services.
RULES: list[tuple[int, int, str]] = [
    # --- carved out FIRST, because each sits inside a wider major group ---
    (6770, 6770, "Shell & SPAC"),            # blank checks: not an industry at all,
                                             # and 246 of them would otherwise flood
                                             # Financials and cap it to one pick
    (2833, 2836, "Pharma & Biotech"),        # medicinal, pharma prep, biologicals
    (8731, 8731, "Pharma & Biotech"),        # commercial physical & biological research
    (3674, 3674, "Semiconductors"),
    (3559, 3559, "Semiconductors"),          # semiconductor production machinery --
                                             # inside the 35xx machinery range
    (3826, 3826, "Healthcare Equipment"),    # lab analytical instruments
    (3841, 3851, "Healthcare Equipment"),    # surgical, medical, dental, ophthalmic
    (8000, 8099, "Healthcare Equipment"),    # health services
    (3570, 3579, "Tech Hardware"),           # computer & office equipment
    (3661, 3673, "Tech Hardware"),           # communications & electronic equipment
    (3675, 3679, "Tech Hardware"),
    (3827, 3827, "Tech Hardware"),           # optical instruments, inside 38xx
    (7370, 7379, "Software & IT Services"),
    (6798, 6798, "Real Estate & REITs"),
    (6199, 6199, "Capital Markets"),         # finance services NEC: AmEx, Coinbase
    (6200, 6299, "Capital Markets"),         # brokers & exchanges
    (6722, 6726, "Capital Markets"),         # investment offices, closed-end funds

    # --- major groups ---
    (100, 999, "Consumer Staples"),
    (1000, 1119, "Mining & Metals"),
    (1200, 1299, "Oil, Gas & Coal"),
    (1300, 1399, "Oil, Gas & Coal"),
    (1400, 1499, "Mining & Metals"),
    (1500, 1799, "Industrials & Machinery"),
    (2000, 2199, "Consumer Staples"),
    (2200, 2399, "Consumer Staples"),        # textiles & apparel
    (2400, 2599, "Industrials & Machinery"),
    (2600, 2699, "Materials & Chemicals"),
    (2700, 2799, "Media & Telecom"),
    (2800, 2899, "Materials & Chemicals"),
    (2900, 2999, "Oil, Gas & Coal"),
    (3000, 3299, "Materials & Chemicals"),
    (3300, 3399, "Mining & Metals"),
    (3400, 3599, "Industrials & Machinery"),
    (3600, 3699, "Industrials & Machinery"),
    (3700, 3799, "Industrials & Machinery"),
    (3800, 3899, "Industrials & Machinery"),
    (3900, 3999, "Retail & Consumer Services"),
    (4000, 4499, "Transportation & Logistics"),
    (4500, 4599, "Transportation & Logistics"),
    (4600, 4699, "Oil, Gas & Coal"),         # pipelines
    (4700, 4799, "Transportation & Logistics"),
    (4800, 4899, "Media & Telecom"),
    (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials & Machinery"), # wholesale
    (5200, 5399, "Retail & Consumer Services"),
    (5400, 5499, "Consumer Staples"),
    (5500, 5999, "Retail & Consumer Services"),
    (6000, 6099, "Banks & Consumer Finance"),
    (6100, 6198, "Banks & Consumer Finance"),
    (6300, 6499, "Insurance"),
    (6500, 6599, "Real Estate & REITs"),
    (6700, 6799, "Capital Markets"),
    (7000, 7099, "Retail & Consumer Services"),
    (7200, 7299, "Retail & Consumer Services"),
    (7300, 7399, "Commercial & Professional Services"),
    (7500, 7699, "Retail & Consumer Services"),
    (7800, 7999, "Media & Telecom"),
    (8100, 8399, "Commercial & Professional Services"),
    (8400, 8999, "Commercial & Professional Services"),
]

UNCLASSIFIED = {9995, 9997, 9999}


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
