"""Generate the home-screen icons, with no image library.

    uv run python scripts/make_icons.py

Pillow is not a dependency and is not worth adding on an 8GB machine for four
flat-colour squares, so this writes the PNGs directly. A PNG is a signature,
an IHDR chunk, zlib-compressed scanlines each prefixed with a filter byte, and
an IEND chunk -- about thirty lines all in.

iOS needs a real PNG at apple-touch-icon or it screenshots the page and uses
that as the icon, which looks broken. Sizes: 180 for iOS, 192 and 512 for the
web app manifest.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUT = Path(__file__).parent.parent / "app" / "static"

BG = (14, 17, 22)        # --bg
BAR = (88, 166, 255)     # --accent
HIGH = (63, 185, 80)     # --green


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    h, w = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in pixels)
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(raw, 9))
           + _chunk(b"IEND", b""))
    path.write_bytes(png)


def render(size: int) -> list[list[tuple[int, int, int]]]:
    """Four ascending bars: reads as 'activity' at 40px and still at 512."""
    px = [[BG for _ in range(size)] for _ in range(size)]
    pad = round(size * 0.20)
    inner = size - 2 * pad
    n = 4
    gap = max(1, round(inner * 0.085))
    bw = (inner - gap * (n - 1)) // n
    # Fractional heights, last one tallest and in green so the eye lands there.
    heights = [0.34, 0.52, 0.72, 1.0]
    for i, frac in enumerate(heights):
        colour = HIGH if i == n - 1 else BAR
        bh = max(2, round(inner * frac))
        x0 = pad + i * (bw + gap)
        y0 = pad + inner - bh
        for y in range(y0, pad + inner):
            row = px[y]
            for x in range(x0, min(x0 + bw, size)):
                row[x] = colour
    return px


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (180, 192, 512):
        p = OUT / f"icon-{s}.png"
        write_png(p, render(s))
        print(f"  wrote {p.relative_to(OUT.parent.parent)}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
