"""Render the scan log to a static site for GitHub Pages.

    uv run python scripts/build_site.py [--out site]

Reuses app/templates/scan.html rather than keeping a second copy, so the page
you see on localhost and the page on your phone cannot drift apart.

Two things differ from the served version:

  * Paths are made relative. GitHub Pages serves a project site from
    /<repo>/, so an absolute "/static/icon-180.png" resolves to the user's
    root and 404s. Every "/static/..." becomes "static/...".
  * The per-session history links point at "/scan?as_of=..." which needs a
    running server. Static output has no query routing, so they render as
    plain text instead of dead links.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

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
        th={"composite": c.composite_threshold, "rvol": c.qual_rvol,
            "ret_z": c.qual_ret_z, "range_ratio": c.qual_range_ratio,
            "gap_atr": c.qual_gap_atr, "min_qualifiers": c.min_qualifiers,
            "max_picks": c.max_picks},
        static_build=True,
    )

    html = html.replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    # Session links need a server; leave the date visible, drop the anchor.
    html = re.sub(r'<a href="/scan\?as_of=[^"]*">([^<]*)</a>', r"\1", html)

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
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
