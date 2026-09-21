"""Attach a sector to every US ticker, from SEC SIC codes. Free, official.

    uv run python scripts/fetch_sectors.py

nasdaqtraded.txt carries no sector, which left rs_sector_20d null for every US
name and made the one-per-sector cap a no-op. SEC publishes a self-reported
SIC code for every registrant, and that is the free route to fixing both.

Why this shape, measured rather than assumed (2026-09-21):

  company_tickers_exchange.json   ticker -> CIK for 6,218 of 6,226 (99.9%).
                                  Carries NO sic field; it is the join key only.
  submissions.zip (bulk)          1.56 GB. Has SIC, but is disqualified on an
                                  8GB machine already deep in swap.
  Financial Statement Data Sets   58-122 MB per quarter, and sub.txt inside is
                                  only 2.3 MB and carries cik + sic. Streamed
                                  out of the zip, so the 600 MB num.txt in the
                                  same archive is never touched.
  data.sec.gov/submissions/CIK…   one request per company. Authoritative, and
                                  the only route for names the quarterlies miss.

Measured cumulative coverage of our 6,226-name universe:
    2026q2 alone      80.9%
    + 2026q1          88.1%
    + 2025q4          89.0%
    + 2025q3          89.6%   <- 4th quarter buys 0.6% for 122 MB

So: two quarters, then the per-CIK API for the ~738 stragglers. Those are
mostly foreign private issuers, funds and trusts, and recent IPOs, which file
outside the quarterly datasets.

SEC fair access: automated requests must carry a User-Agent identifying you
with a contact address, and the published ceiling is 10 requests/second. Both
are honoured below. Set SEC_CONTACT to change the address.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.sic import sic_to_sector  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / ".sec_cache"

import os  # noqa: E402

CONTACT = os.getenv("SEC_CONTACT", "aaron-thompson@outlook.com")
# No Accept-Encoding: urllib does not transparently decompress, so asking
# for gzip returns raw deflate bytes and json.loads dies on the 0x8b magic.
UA = {"User-Agent": f"StockStellar/0.1 ({CONTACT})", "Accept": "*/*"}

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
FSDS = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

QUARTERS = ("2026q2", "2026q1")
RATE = 9.0          # requests/sec, just under SEC's published 10
TIMEOUT = 30


def _get(url: str, timeout: int = TIMEOUT) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def _download(url: str, dest: Path, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    cached {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    fetching {dest.name} ...", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, \
         open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"    got {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
    return dest


def ticker_to_cik() -> pd.DataFrame:
    d = json.loads(_get(TICKERS_URL))
    df = pd.DataFrame(d["data"], columns=d["fields"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df[["ticker", "cik"]].drop_duplicates("ticker")


def sic_from_quarterlies(quarters=QUARTERS) -> pd.DataFrame:
    """cik -> sic from sub.txt, streamed out of each quarterly zip."""
    frames = []
    for q in quarters:
        z = _download(FSDS.format(q=q), CACHE / f"{q}.zip")
        with zipfile.ZipFile(z) as zf:
            # Only sub.txt is read. num.txt in the same archive is ~600 MB and
            # is never decompressed.
            s = pd.read_csv(zf.open("sub.txt"), sep="\t",
                            usecols=["cik", "sic", "name"], dtype={"cik": "Int64"})
        frames.append(s.dropna(subset=["sic"]))
    out = pd.concat(frames, ignore_index=True).drop_duplicates("cik", keep="first")
    out["sic"] = out["sic"].astype("Int64")
    return out[["cik", "sic"]]


def sic_from_api(ciks: list[int], limit: int | None = None) -> pd.DataFrame:
    """Per-CIK fallback, rate-limited to stay inside SEC's published ceiling."""
    todo = ciks[:limit] if limit else ciks
    rows, errors, t_last = [], 0, 0.0
    for i, cik in enumerate(todo, 1):
        dt = time.monotonic() - t_last
        if dt < 1.0 / RATE:
            time.sleep(1.0 / RATE - dt)
        t_last = time.monotonic()
        try:
            d = json.loads(_get(SUBMISSIONS.format(cik=int(cik)), timeout=15))
            sic = (d.get("sic") or "").strip()
            if sic:
                rows.append({"cik": int(cik), "sic": int(sic)})
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError):
            errors += 1
        if i % 200 == 0:
            print(f"    {i}/{len(todo)} ({len(rows)} found, {errors} errors)", flush=True)
    print(f"    api fill: {len(rows)} found, {errors} errors, {len(todo)} attempted")
    return pd.DataFrame(rows, columns=["cik", "sic"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="skip the per-CIK fallback")
    ap.add_argument("--api-limit", type=int, default=None, help="cap fallback requests")
    args = ap.parse_args()

    uni_path = DATA / "universe_us.csv"
    if not uni_path.exists():
        raise SystemExit("run scripts/build_universe.py first")
    uni = pd.read_csv(uni_path)
    print(f"  universe: {len(uni):,} US tickers")

    print("  [1/3] ticker -> CIK")
    uni = uni.drop(columns=[c for c in ("cik", "sic", "sector") if c in uni.columns])
    uni = uni.merge(ticker_to_cik(), on="ticker", how="left")
    print(f"    matched {uni.cik.notna().sum():,} ({uni.cik.notna().mean():.1%})")

    print("  [2/3] CIK -> SIC from quarterly datasets")
    uni = uni.merge(sic_from_quarterlies(), on="cik", how="left")
    print(f"    covered {uni.sic.notna().sum():,} ({uni.sic.notna().mean():.1%})")

    missing = uni[uni.sic.isna() & uni.cik.notna()]["cik"].astype(int).tolist()
    if missing and not args.no_api:
        print(f"  [3/3] per-CIK API for {len(missing):,} stragglers "
              f"(~{len(missing)/RATE/60:.1f} min at {RATE:.0f} req/s)")
        extra = sic_from_api(missing, limit=args.api_limit)
        if len(extra):
            uni = uni.merge(extra.rename(columns={"sic": "sic_api"}), on="cik", how="left")
            uni["sic"] = uni["sic"].fillna(uni.pop("sic_api"))
    else:
        print("  [3/3] skipped")

    uni["sector"] = uni["sic"].map(sic_to_sector)
    have = uni.sector.notna()
    print(f"\n  FINAL: {have.sum():,}/{len(uni):,} with a sector ({have.mean():.1%})")
    print(f"  sectors: {uni.sector.value_counts().to_dict()}")
    if (~have).any():
        print(f"  no sector: {(~have).sum():,} — these keep sector=NA and are not "
              f"capped against each other in selection")

    uni.to_csv(uni_path, index=False)
    print(f"\n  wrote {uni_path} with cik, sic, sector columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
