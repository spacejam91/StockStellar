"""Render the scan log to a static site for GitHub Pages.

    uv run python scripts/build_site.py [--out site]

Reuses app/templates/scan.html rather than keeping a second copy, so the page
you see on localhost and the page on your phone cannot drift apart.

Two things differ from the served version:

  * Paths are made relative. GitHub Pages serves a project site from
    /<repo>/, so an absolute "/static/icon-180.png" resolves to the user's
    root and 404s. Every "/static/..." becomes "static/...".
  * The per-session history links point at "/scan?as_of=..." which needs query
    routing a static host does not have. So one page per session is written to
    s/<date>.html and the links are rewritten to those. Without this the live
    site had no way to reach a past session at all -- the history table
    rendered as inert text.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

def _TH(c) -> dict:
    """Fallback meter limits. The real, per-session ones ride on each pick as
    thr_*; these only cover rows logged before those were carried."""
    return {"composite": c.composite_threshold, "rvol": c.qual_rvol,
            "ret_z": c.qual_ret_z, "range_ratio": c.qual_range_ratio,
            "gap_atr": c.qual_gap_atr, "min_qualifiers": c.min_qualifiers,
            "max_picks": c.max_picks, "scaled": c.qualifier_mode == "scaled"}


ROOT = Path(__file__).parent.parent
TEMPLATES = ROOT / "app" / "templates"
STATIC = ROOT / "app" / "static"

# Running `python scripts/build_site.py` puts scripts/ on sys.path, not the
# project root, so `import scanner` fails. Prepend the root explicitly rather
# than requiring callers to set PYTHONPATH -- the GitHub Actions job invokes
# this by path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build(out: Path) -> int:
    from scanner import store as scan_store

    day = scan_store.day_for(None)
    picks = scan_store.picks_for(day["as_of_date"]) if day else []
    hist = scan_store.day_history(limit=30)
    empty_share = (sum(1 for r in hist if r["empty_day"]) / len(hist)) if hist else None

    from app import links
    from scanner.config import ScanConfig

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      autoescape=select_autoescape(["html"]))
    env.globals.update(links.jinja_globals())
    c = ScanConfig()
    # The template calls url_for / request in the served path; give it a stub
    # so the same file renders headless.
    html = env.get_template("scan.html").render(
        request=None,
        day=day,
        picks=picks,
        history=hist,
        empty_share=empty_share,
        provider=(day or {}).get("provider") or "unknown",
        th=_TH(c),
        shealth=scan_store.session_health(day, hist),
        static_build=True,
    )

    def localise(page: str, depth: int = 0) -> str:
        """Absolute app paths -> paths that work on a project Pages site.

        GitHub Pages serves this from /<repo>/, so "/static/x" resolves to the
        user root and 404s. `depth` is how many directories deep the page sits.
        """
        up = "../" * depth
        page = (page.replace('href="/static/', f'href="{up}static/')
                    .replace('src="/static/', f'src="{up}static/'))
        page = re.sub(r'href="/scan\?as_of=([0-9-]+)"', rf'href="{up}s/\1.html"', page)
        # /name/XENE -> name/XENE.html. Without the extension this resolved to
        # a directory that was never written, so every ticker link and every
        # "details, news & filings" link on the published site was a 404. The
        # route exists in the FastAPI app; the static build simply never
        # rendered it.
        page = re.sub(r'href="/name/([^"]+)"', rf'href="{up}name/\1.html"', page)
        page = page.replace('href="/guide"', f'href="{up}guide.html"')
        page = page.replace('href="/api/scan/base-rates"', f'href="{up}base-rates.json"')
        page = page.replace('href="/"', f'href="{up}index.html"')
        return page

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(localise(html, 0), encoding="utf-8")

    # One page per ticker that appears in any published session, so the links
    # above resolve. News is fetched at build time under a wall-clock budget:
    # the newest session's names are done first and always get headlines, and
    # if the feeds are slow the older pages fall back to criteria plus outbound
    # links rather than stalling the build.
    write_name_pages(env, out, localise, scan_store, c, hist, picks)

    # The reading guide. Static, and deliberately reachable even when no scan
    # has ever run -- it is the page that explains every other one.
    (out / "guide.html").write_text(
        localise(env.get_template("guide.html").render(request=None), 0), encoding="utf-8")

    # One page per logged session, so the history table is navigable.
    sess_dir = out / "s"
    sess_dir.mkdir(exist_ok=True)
    written = 0
    for row in hist:
        d = row["as_of_date"]
        day_i = scan_store.day_for(d)
        if day_i is None:
            continue
        page = env.get_template("scan.html").render(
            request=None, day=day_i, picks=scan_store.picks_for(d), history=hist,
            empty_share=empty_share, provider=day_i.get("provider") or "unknown",
            th=_TH(c), shealth=scan_store.session_health(day_i, hist),
            static_build=True)
        (sess_dir / f"{d}.html").write_text(localise(page, 1), encoding="utf-8")
        written += 1
    print(f"  wrote {written} per-session pages to {sess_dir}")
    if STATIC.exists():
        shutil.copytree(STATIC, out / "static", dirs_exist_ok=True)
        # The manifest carries its own absolute paths, which the HTML rewrite
        # above does not touch. A project Pages site lives at /<repo>/, so
        # "/static/icon-192.png" resolves to the user root and 404s -- and a
        # manifest with unreachable icons silently downgrades the install.
        # Paths inside a manifest resolve relative to the manifest itself.
        mf = out / "static" / "manifest.webmanifest"
        if mf.exists():
            import json
            d = json.loads(mf.read_text())
            for icon in d.get("icons", []):
                icon["src"] = icon["src"].rsplit("/", 1)[-1]
            d["start_url"] = "../"
            d["scope"] = "../"
            mf.write_text(json.dumps(d, indent=2))
    # The base rates the disclaimer points at. A static host has no /api, and
    # the link went nowhere -- it is the one measured, probability-shaped output
    # this project has, so it has to resolve.
    try:
        import json as _json
        br = scan_store.score_band_base_rates(horizon=5)
        (out / "base-rates.json").write_text(_json.dumps({
            "horizon_days": 5,
            "note": "Measured from the point-in-time log: what actually followed each "
                    "score band, in the direction the scanner called. Read n and se_pp "
                    "before the headline number. Not a prediction.",
            "providers_logged": scan_store.provider_breakdown(),
            "bands": [] if br.empty else br.to_dict(orient="records"),
        }, indent=2, default=str))
    except Exception as e:                                       # noqa: BLE001
        print(f"  base rates unavailable ({e}) — writing an empty file")
        (out / "base-rates.json").write_text('{"bands": []}')

    # Pages runs Jekyll by default, which ignores files beginning with an
    # underscore and can mangle output. This opts out.
    (out / ".nojekyll").write_text("")

    n = len(list(out.rglob("*")))
    print(f"  wrote {out}/index.html  ({len(html):,} bytes, {n} files total)")
    print(f"  session {day['as_of_date'] if day else 'none'}  "
          f"picks {len(picks)}  provider {(day or {}).get('provider', 'n/a')}")
    if not day:
        print("  WARNING: no scan logged — the page will say so rather than showing stale data")
    return 0


# How long the whole build may spend waiting on news and filings feeds. Past
# this the remaining pages are still written, just without headlines -- a page
# that renders the criteria and links out is far better than a build that hangs
# on a slow RSS endpoint and publishes nothing.
NEWS_BUDGET_SECONDS = float(os.getenv("NEWS_BUDGET_SECONDS", "240"))


def write_name_pages(env, out: Path, localise, scan_store, cfg, hist, latest_picks) -> int:
    """A page per flagged ticker: criteria verdict, meters, base rates, news."""
    import time

    from app.main import _thresholds, _verdict
    from scanner import news as news_mod

    # Newest first, so the names on the front page are the ones that always get
    # live headlines if the budget runs out.
    seen: dict[str, dict] = {}
    for row in hist:
        for p in scan_store.picks_for(row["as_of_date"]):
            seen.setdefault(str(p["ticker"]).upper(), p)
    for p in latest_picks:
        seen[str(p["ticker"]).upper()] = p
    if not seen:
        return 0

    ordered = [t for t in (str(p["ticker"]).upper() for p in latest_picks) if t in seen]
    ordered += [t for t in seen if t not in ordered]

    try:
        bands = scan_store.score_band_base_rates(horizon=5)
    except Exception:                                            # noqa: BLE001
        bands = None

    ndir = out / "name"
    ndir.mkdir(exist_ok=True)
    th = _thresholds()
    started, with_news = time.time(), 0
    for ticker in ordered:
        pick = seen[ticker]
        band = None
        if bands is not None and not bands.empty and pick.get("score_pct") is not None:
            hit = bands[bands["score_band"] == min(int(pick["score_pct"] // 10 * 10), 90)]
            if not hit.empty:
                band = hit.iloc[0].to_dict() | {"horizon": 5}

        news = filings = []
        if time.time() - started < NEWS_BUDGET_SECONDS:
            try:
                news = news_mod.fetch_news(ticker, pick.get("market"), limit=8)
                filings = news_mod.fetch_filings(news_mod.cik_for(ticker), limit=6)
                with_news += 1 if news else 0
            except Exception as e:                               # noqa: BLE001
                print(f"    {ticker}: news unavailable ({e})")

        page = env.get_template("name.html").render(
            request=None, ticker=ticker, pick=pick, market=pick.get("market"),
            sector=pick.get("sector"), th=th, verdict=_verdict(pick, th),
            band=band, news=news, filings=filings, static_build=True)
        (ndir / f"{ticker}.html").write_text(localise(page, 1), encoding="utf-8")
    print(f"  wrote {len(ordered)} name pages to {ndir} "
          f"({with_news} with live headlines, {time.time() - started:.0f}s)")
    return len(ordered)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site", help="output directory (default: site)")
    args = ap.parse_args()
    return build(ROOT / args.out)


if __name__ == "__main__":
    sys.exit(main())
