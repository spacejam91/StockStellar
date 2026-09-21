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
import io
import logging
import json
import sys
import time
import urllib.error
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

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

# Nine quarters costs ~4.9 MB via Range (below) and lifts coverage from 82% to
# ~90% before the API top-up. Annual-only filers need the wider window.
QUARTERS = ("2026q2", "2026q1", "2025q4", "2025q3", "2025q2",
            "2025q1", "2024q4", "2024q3", "2024q2")
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


def _range(url: str, start: int, length: int, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{start+length-1}"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def fetch_sub_txt(url: str) -> bytes:
    """Pull ONLY sub.txt out of a remote quarterly zip, via HTTP Range.

    The archive is 58-122 MB but sub.txt is ~580 KB compressed, and the other
    members include a ~600 MB num.txt nobody here wants. Reading the zip's
    central directory over Range and then fetching just this member's bytes
    turns a 145 MB job into a 5 MB one -- which matters on a machine that is
    already swapping, and matters again in CI where it runs every weekday.

    Reads the End Of Central Directory from the file's tail, walks the central
    directory to find sub.txt's local-header offset and compressed size, then
    range-fetches exactly that span and raw-inflates it.
    """
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA),
                                timeout=30) as r:
        total = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("server will not serve ranges")

    tail = _range(url, max(0, total - 65_536), min(65_536, total))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("no EOCD found")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    cd = tail[eocd - cd_size:eocd] if cd_size <= eocd else _range(url, cd_off, cd_size)

    pos, target = 0, None
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        comp_size, = struct.unpack("<I", cd[pos + 20:pos + 24])
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        lho, = struct.unpack("<I", cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        if name == "sub.txt":
            target = (lho, comp_size)
            break
        pos += 46 + nlen + elen + clen
    if target is None:
        raise RuntimeError("sub.txt not in central directory")

    lho, comp_size = target
    head = _range(url, lho, 30)
    nlen, elen = struct.unpack("<HH", head[26:30])
    raw = _range(url, lho + 30 + nlen + elen, comp_size)
    return zlib.decompress(raw, -15)


def sic_from_quarterlies(quarters=QUARTERS) -> pd.DataFrame:
    """cik -> sic from sub.txt across several quarters, newest wins."""
    frames, bytes_in = [], 0
    for q in quarters:
        url = FSDS.format(q=q)
        try:
            blob = fetch_sub_txt(url)
            bytes_in += len(blob)
            s = pd.read_csv(io.BytesIO(blob), sep="\t",
                            usecols=["cik", "sic"], dtype={"cik": "Int64"})
        except Exception as e:                                   # noqa: BLE001
            # Fall back to the whole archive rather than losing the quarter.
            log.warning("range fetch failed for %s (%s); downloading archive", q, e)
            try:
                z = _download(url, CACHE / f"{q}.zip")
                with zipfile.ZipFile(z) as zf:
                    s = pd.read_csv(zf.open("sub.txt"), sep="\t",
                                    usecols=["cik", "sic"], dtype={"cik": "Int64"})
            except Exception as e2:                              # noqa: BLE001
                log.warning("quarter %s unavailable: %s", q, e2)
                continue
        frames.append(s.dropna(subset=["sic"]))
    if not frames:
        raise SystemExit("no quarterly data could be fetched")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("cik", keep="first")
    out["sic"] = out["sic"].astype("Int64")
    print(f"    {len(quarters)} quarters, {bytes_in/1e6:.1f} MB decompressed, "
          f"{len(out):,} filers with a SIC")
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

    # Deduplicate: two tickers can share one CIK (a company with both common
    # shares and listed notes), so the same CIK would be requested twice and
    # the merge below would duplicate every row matching it.
    missing = sorted(set(uni[uni.sic.isna() & uni.cik.notna()]["cik"].astype(int)))
    if missing and not args.no_api:
        print(f"  [3/3] per-CIK API for {len(missing):,} stragglers "
              f"(~{len(missing)/RATE/60:.1f} min at {RATE:.0f} req/s)")
        extra = sic_from_api(missing, limit=args.api_limit)
        if len(extra):
            extra = extra.drop_duplicates("cik")
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
