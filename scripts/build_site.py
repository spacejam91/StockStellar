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
        page = page.replace('href="/name/', f'href="{up}name/')
        page = page.replace('href="/guide"', f'href="{up}guide.html"')
        page = page.replace('href="/"', f'href="{up}index.html"')
        return page

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(localise(html, 0), encoding="utf-8")

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site", help="output directory (default: site)")
    args = ap.parse_args()
    return build(ROOT / args.out)


if __name__ == "__main__":
    sys.exit(main())
