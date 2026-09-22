"""News and filings for one ticker.

Uses PUBLISHED FEEDS, not page scraping. Google News exposes an RSS endpoint and
SEC exposes a JSON submissions API; both are meant for programmatic consumption,
both are free, and neither breaks when a site redesigns. Scraping the rendered
Google Finance or Yahoo pages would be faster to write, against their terms, and
broken within weeks.

Everything is cached on disk with a TTL, because the same three tickers get
opened repeatedly in one sitting and neither service should be asked twice in
five minutes for an answer that has not changed.

Nothing here interprets the news. It retrieves headlines and filings so a human
can read them; it does not classify them as good or bad, and it does not feed
them into the score.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / ".news_cache"
TTL_SECONDS = 20 * 60

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-CA&gl=CA&ceid=CA:en"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

_CA_VENUE = {".TO": "TSX", ".V": "TSX Venture", ".CN": "CSE", ".NE": "Cboe Canada"}

# Forms worth surfacing. 8-K is material events; 4 is insider transactions;
# SC 13D/G are activist and large stakes; 424/S-1 are offerings, i.e. dilution.
INTERESTING_FORMS = ("8-K", "6-K", "4", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
                     "424B5", "424B4", "S-1", "S-3", "10-Q", "10-K", "20-F", "40-F")


def _contact() -> str:
    import os
    return os.getenv("SEC_CONTACT", "aaron-thompson@outlook.com")


def _ua() -> dict:
    return {"User-Agent": f"StockStellar/0.1 ({_contact()})", "Accept": "*/*"}


def _cached(key: str, ttl: int = TTL_SECONDS):
    p = CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _store(key: str, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}.json"
    try:
        p.write_text(json.dumps(value))
    except OSError:
        pass


def _get(url: str, timeout: int = 12) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_ua()), timeout=timeout).read()


def split_ticker(ticker: str) -> tuple[str, str | None]:
    t = (ticker or "").strip().upper()
    for suffix, venue in _CA_VENUE.items():
        if t.endswith(suffix):
            return t[: -len(suffix)], venue
    return t, None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def fetch_news(ticker: str, market: str | None = None, limit: int = 8) -> list[dict]:
    """Recent headlines. Returns [] on any failure -- news is a nicety, and a
    dead feed must never take the page down with it."""
    key = f"news_{ticker}"
    if (hit := _cached(key)) is not None:
        return hit[:limit]

    base, venue = split_ticker(ticker)
    terms = [f'"{base}"', "stock"]
    terms.append(venue if venue else ("TSX" if (market or "").upper() == "CA" else "shares"))
    url = GOOGLE_NEWS.format(q=urllib.parse.quote_plus(" ".join(terms)))

    items: list[dict] = []
    try:
        root = ET.fromstring(_get(url))
        for it in list(root.iter("item"))[:limit]:
            title = _clean(it.findtext("title"))
            # Google appends " - Publisher" to the headline; split it back out.
            source = _clean(it.findtext("source")) or (
                title.rsplit(" - ", 1)[-1] if " - " in title else "")
            if source and title.endswith(f" - {source}"):
                title = title[: -(len(source) + 3)]
            items.append({
                "title": title,
                "source": source,
                "published": _clean(it.findtext("pubDate"))[:22],
                "url": _clean(it.findtext("link")),
            })
    except Exception as e:                                  # noqa: BLE001
        log.info("news fetch failed for %s: %s", ticker, e)
        return []

    _store(key, items)
    return items


def fetch_filings(cik: int | None, limit: int = 6) -> list[dict]:
    """Recent SEC filings for a CIK. US only -- Canada has no equivalent, since
    SEDAR+'s terms forbid automated access and building a database from it."""
    if not cik:
        return []
    key = f"filings_{int(cik)}"
    if (hit := _cached(key, ttl=60 * 60)) is not None:
        return hit[:limit]

    out: list[dict] = []
    try:
        d = json.loads(_get(SEC_SUBMISSIONS.format(cik=int(cik))))
        r = d.get("filings", {}).get("recent", {})
        forms, dates = r.get("form", []), r.get("filingDate", [])
        accs, docs = r.get("accessionNumber", []), r.get("primaryDocument", [])
        descs = r.get("primaryDocDescription", [])
        for i, form in enumerate(forms):
            if form not in INTERESTING_FORMS:
                continue
            acc = accs[i] if i < len(accs) else ""
            out.append({
                "form": form,
                "date": dates[i] if i < len(dates) else "",
                "desc": (descs[i] if i < len(descs) else "") or "",
                "url": SEC_FILING_URL.format(
                    cik=int(cik), acc_nodash=acc.replace("-", ""),
                    doc=docs[i] if i < len(docs) else ""),
            })
            if len(out) >= limit * 2:
                break
    except Exception as e:                                  # noqa: BLE001
        log.info("filings fetch failed for CIK %s: %s", cik, e)
        return []

    _store(key, out)
    return out[:limit]


# pandas' default na_values include "NA", "NULL", "NaN", "None" and "nan" --
# every one of which is a plausible ticker. NA.TO is National Bank of Canada.
# Reading a universe file without keep_default_na=False silently turns such a
# ticker into a float NaN, and .astype(str).str.upper() then resurrects it as
# the literal string "NAN" -- which is exactly how a phantom "NAN" ticker with
# 343 duplicate rows got into the bars.
def read_universe(path, **kw):
    import pandas as _pd
    return _pd.read_csv(path, keep_default_na=False, na_values=[""], **kw)


def cik_for(ticker: str) -> int | None:
    """Look up a CIK from the universe file the scanner already maintains."""
    import pandas as pd
    p = Path(__file__).parent.parent / "data" / "universe_us.csv"
    if not p.exists():
        return None
    try:
        u = pd.read_csv(p, usecols=["ticker", "cik"])
    except (ValueError, OSError):
        return None
    row = u[u["ticker"].str.upper() == ticker.strip().upper()]
    if row.empty or pd.isna(row.iloc[0]["cik"]):
        return None
    return int(row.iloc[0]["cik"])
