"""Outbound links for a ticker. One definition, used by the server and the
static build, so the two cannot drift.

Google Finance would be the nicer destination but its URLs require an exact
exchange code (NASDAQ / NYSE / NYSEAMERICAN / TSE / CVE / CNSX), and a logged
pick row carries only `market` (CA/US) -- not which venue inside it. A wrong
code 404s. Google search resolves every ticker, returns the knowledge panel
with price, chart and news, and cannot break on a missing exchange, so it is
the primary click target.

The suffix matters: searching "SHOP.TO stock" is noticeably worse than
"SHOP stock TSX", so Canadian tickers are unsuffixed and given their venue as
a word instead.
"""

from __future__ import annotations

from urllib.parse import quote_plus

# Yahoo suffix -> how a human would name that venue in a search
_CA_VENUE = {".TO": "TSX", ".V": "TSX Venture", ".CN": "CSE", ".NE": "Cboe Canada"}


def split_ticker(ticker: str) -> tuple[str, str | None]:
    """('SHOP.TO') -> ('SHOP', 'TSX'); ('AAPL') -> ('AAPL', None)."""
    t = (ticker or "").strip().upper()
    for suffix, venue in _CA_VENUE.items():
        if t.endswith(suffix):
            return t[: -len(suffix)], venue
    return t, None


def company_url(ticker: str, market: str | None = None) -> str:
    """Google search for the company — price, chart, news, key figures."""
    base, venue = split_ticker(ticker)
    terms = [base, "stock"]
    if venue:
        terms.append(venue)
    elif (market or "").upper() == "CA":
        terms.append("TSX")
    return "https://www.google.com/search?q=" + quote_plus(" ".join(terms))


def news_url(ticker: str, market: str | None = None) -> str:
    """Recent news only — the 'what happened' question a flag usually raises."""
    base, venue = split_ticker(ticker)
    terms = [base, "stock"]
    if venue:
        terms.append(venue)
    elif (market or "").upper() == "CA":
        terms.append("TSX")
    return "https://news.google.com/search?q=" + quote_plus(" ".join(terms))


def jinja_globals() -> dict:
    return {"company_url": company_url, "news_url": news_url}
