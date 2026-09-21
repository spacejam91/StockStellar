"""Build data/universe_us.csv and data/universe_ca.csv from free sources.

    uv run python scripts/build_universe.py

US  — nasdaqtrader.com's nasdaqtraded.txt. One file covering every US listing
      venue, updated intraday, no key, no auth. It is the union of
      nasdaqlisted.txt and otherlisted.txt under one 12-column schema, so it
      also carries a `Listing Exchange` column that the two-file split cannot
      express.

CA  — the CSE's own public web API. Two endpoints: one lists TSX + TSXV + NEO,
      the other CSE's own issuers. This route matters because TMX's Terms of
      Use grant only a personal view-and-print licence and prohibit automated
      extraction, and TMX sells the machine-readable issuer list by
      subscription. The CSE endpoints carry the same TSX/TSXV data and are
      openly served.

Two traps worth knowing, both hit during verification:

  * nasdaqtraded.txt ends with a `File Creation Time: ...` trailer line that
    will corrupt a naive parse. Drop it.
  * nasdaqtrader.com returns HTTP 200 with a ~43KB "Page Not Available" HTML
    body for any bad URL, so the status code proves nothing. Validate the
    header row instead.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
UA = {"User-Agent": "curl/8.4.0", "Accept": "*/*"}

NASDAQ_TRADED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
CSE_OTHER = "https://thecse.com/api/webapi/other-companies/"     # TSX, TSXV, NEO
CSE_OWN = "https://thecse.com/api/webapi/listed-companies/"      # CSE

# Yahoo's suffix per Canadian venue. Symbol format is the single largest source
# of silent bugs here: every source spells these differently (Yahoo .TO/.V/.CN,
# Bloomberg "CN", Alpha Vantage .TRT/.TRV, IBKR TSE/VENTURE). Translate in
# exactly one place, which is this file.
# NB: the CSE API spells TSX Venture "TSX.V", not "TSXV". Getting this wrong
# silently drops all ~1,750 Venture names and the totals still look plausible.
YAHOO_SUFFIX = {"TSX": ".TO", "TSX.V": ".V", "CSE": ".CN", "NEO": ".NE"}


def _get(url: str, timeout: int = 60) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def build_us() -> pd.DataFrame:
    raw = _get(NASDAQ_TRADED).decode("utf-8", "replace")
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("File Creation Time")]
    if not lines or not lines[0].startswith("Nasdaq Traded"):
        raise SystemExit("nasdaqtraded.txt: unexpected header — the URL may be serving an error page")

    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")
    before = len(df)

    df = df[(df["Test Issue"] != "Y") & (df["ETF"] != "Y")]
    df = df[df["Listing Exchange"].isin(["Q", "N", "A"])]

    # Drop non-common-stock security types. Nasdaq encodes these in a 5th
    # character (W warrant, R right, U unit); NYSE/AMEX use dot suffixes and
    # '$' for preferred series. Do NOT classify off the security name: the live
    # file spells the same concept "Warrant", "Warrants" and "warrants", so
    # name matching is a guaranteed silent miss.
    # Check the `Symbol` column, not `NASDAQ Symbol`: on NYSE/AMEX rows Symbol
    # carries 543 suffixed tickers where NASDAQ Symbol carries only 24, so the
    # wrong column lets ~500 preferreds and warrants through and the total
    # still looks reasonable.
    sym = df["Symbol"].str.strip()
    is_nasdaq = df["Listing Exchange"] == "Q"
    suffix5 = is_nasdaq & (sym.str.len() == 5) & sym.str[-1].isin(list("WRU"))
    dotted = (~is_nasdaq) & sym.str.contains(r"[.$]", regex=True)
    # Debt and preferred instruments listed under a plain symbol slip past the
    # suffix rules (e.g. "…8.50% Notes due 2031", "First Mortgage Bonds"). The
    # security name is unreliable for classification generally, but these
    # instrument words are unambiguous.
    # Instrument words only. Do NOT filter on "Depositary": American
    # Depositary Shares are how BABA, BIDU, BHP, NatWest and ~88 other real
    # common-equity names are listed, and excluding them silently removes some
    # of the most liquid tickers on the tape. A depositary share standing in
    # for preferred stock always says "Preferred" and is still caught below.
    debt = df["Security Name"].str.contains(
        r"\b(?:Notes?|Bonds?|Debentures?|Preferred)\b",
        case=False, regex=True, na=False)
    df = df[~(suffix5 | dotted | debt)]

    out = pd.DataFrame({
        "ticker": sym[df.index],
        "name": df["Security Name"].str.strip(),
        "exchange": df["Listing Exchange"].map({"Q": "NASDAQ", "N": "NYSE", "A": "AMEX"}),
        # Financial Status is a risk flag, not a filter: D deficient,
        # E delinquent, Q bankrupt, N normal.
        "financial_status": df["Financial Status"],
    })
    out = out.drop_duplicates("ticker")
    print(f"  US: {before:,} rows -> {len(out):,} common equities "
          f"({out.exchange.value_counts().to_dict()})")
    return out.reset_index(drop=True)


def build_ca() -> pd.DataFrame:
    frames = []
    for label, url in (("TSX/TSXV/NEO", CSE_OTHER), ("CSE", CSE_OWN)):
        d = json.loads(_get(url))
        recs = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
        f = pd.DataFrame(recs)
        f = f[(f["security_type"] == "Equity") & (f["status"] == "Active")]
        print(f"  CA {label}: {len(recs):,} records -> {len(f):,} active equities")
        frames.append(f)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["listing_market"].isin(YAHOO_SUFFIX)]

    # Yahoo spells a Canadian sub-suffix with a HYPHEN, not a dot: BIP-UN.TO,
    # CTC-A.TO, AGF-B.TO. Emitting BIP.UN.TO returns "possibly delisted; no
    # timezone found" -- so every .UN REIT unit and every .A/.B share class,
    # exactly the ones kept deliberately above, silently fetched nothing.
    sym = (df["symbol"].str.strip().str.upper().str.replace(".", "-", regex=False))
    out = pd.DataFrame({
        "ticker": sym + df["listing_market"].map(YAHOO_SUFFIX),
        "name": df["security_name"].str.strip(),
        "exchange": df["listing_market"],
        "sector": df["sector"].fillna("").str.title().replace("", pd.NA),
    })
    # A bare Canadian ticker is provably not unique across venues, so dedupe on
    # the suffixed form, which is what Yahoo is asked for.
    out = out.drop_duplicates("ticker")

    # Canadian sub-suffixes sit BEFORE the venue suffix (FTN.PR.A.TO), so they
    # need their own pass. Measured on the live list: 256 preferred, 122 NEX,
    # 59 warrants -- and every one of them failed to resolve at the data
    # provider, which is how they surfaced.
    #   drop  .PR/.PF  preferred series
    #         .WT/.RT  warrants and rights
    #         .DB      debentures
    #         .H       NEX board (issuers that fell below TSXV requirements)
    #   KEEP  .UN      income-trust/REIT units -- GRT.UN, BIP.UN are real and liquid
    #         .A/.B    share classes -- AGF.B is ordinary equity
    base = out["ticker"].str.replace(r"\.(TO|V|CN|NE)$", "", regex=True)
    drop = base.str.contains(r"-(?:PR|PF|WT|RT|DB)\b", regex=True) | base.str.endswith("-H")

    # .U is the USD-denominated twin of a CAD listing. Drop it only when the
    # CAD line also exists, otherwise a USD-only name (FIH.U) would vanish.
    bases = set(base[~drop])
    usd_dupe = base.str.endswith("-U") & base.str.replace(r"-U$", "", regex=True).isin(bases)

    removed = int((drop | usd_dupe).sum())
    out = out[~(drop | usd_dupe)]
    print(f"  CA: dropped {removed} preferred/warrant/NEX/USD-duplicate lines")
    print(f"  CA total: {len(out):,} ({out.exchange.value_counts().to_dict()})")
    return out.reset_index(drop=True)


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    us, ca = build_us(), build_ca()
    us.to_csv(DATA_DIR / "universe_us.csv", index=False)
    ca.to_csv(DATA_DIR / "universe_ca.csv", index=False)
    print(f"\n  wrote {DATA_DIR/'universe_us.csv'} and {DATA_DIR/'universe_ca.csv'}")
    print("  NOTE: this OVERWRITES universe_us.csv, dropping the cik/sic/sector columns.")
    print("        Run scripts/fetch_sectors.py straight after — they are a pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
