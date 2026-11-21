"""Animated README banner (pure SVG + CSS, renders on GitHub) drawn from the predicted city -> assets/banner.svg

    python tools/make_banner.py
"""
import base64
import math
import sys
from pathlib import Path

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import LINKS, PRED, read  # noqa: E402

W, H = 1280, 400
MAP = (560, 10, 1270, 390)          # x0, y0, x1, y1 of the city on the canvas
TEAL, LIGHT, LINE, DARK, AMBER = "#0E8C7F", "#F4F7F9", "#DCE4EA", "#0B1F33", "#F5A623"


def main():
    nav = read(PRED / "TeamA" / LINKS)
    geoms = shapely.from_wkt(nav["geom"].to_numpy())
    fc = nav["road_class"].astype(int).to_numpy()
    d = nav["direction"].to_numpy()
    b = shapely.total_bounds(geoms)
    k = math.cos(math.radians((b[1] + b[3]) / 2))
    sx = (MAP[2] - MAP[0]) / ((b[2] - b[0]) * k)
    sy = (MAP[3] - MAP[1]) / (b[3] - b[1])
    s = min(sx, sy)
    ox = MAP[2] - (b[2] - b[0]) * k * s
    oy = MAP[1] + ((MAP[3] - MAP[1]) - (b[3] - b[1]) * s) / 2

    def path(idx, reverse_t=False):
        out = []
        for i in idx:
            c = np.asarray(geoms[i].coords)
            if reverse_t and d[i] == "T":
                c = c[::-1]
            x = ox + (c[:, 0] - b[0]) * k * s
            y = oy + (b[3] - c[:, 1]) * s
            ln = shapely.simplify(shapely.linestrings(np.c_[x, y]), 0.7)
            pts = np.round(shapely.get_coordinates(ln), 1)
            if len(pts) < 2 or np.hypot(*(pts[-1] - pts[0])) < 0.8:
                continue
            out.append("M" + " ".join(f"{p[0]:g},{p[1]:g}" for p in pts))
        return "".join(out)

    simple = lambda m: [i for i in np.flatnonzero(m)]  # noqa: E731
    bg = path(simple(fc >= 4))
    main_roads = path(simple((fc <= 3) & (d == "B")))
    oneway = path(simple(np.isin(d, ["F", "T"]) & (fc <= 4)), reverse_t=True)
    oneway_small = path(simple(np.isin(d, ["F", "T"]) & (fc == 5)), reverse_t=True)

    font = base64.b64encode((ROOT / "deck/vendor/fonts/GoogleSansFlex-latin.woff2").read_bytes()).decode()
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
<style>
@font-face{{font-family:"GSF";src:url(data:font/woff2;base64,{font}) format("woff2");font-weight:100 1000}}
.t{{font-family:"GSF","Google Sans","Segoe UI",system-ui,sans-serif}}
.bg{{fill:none;stroke:{LINE};stroke-opacity:.10;stroke-width:.55}}
.mr{{fill:none;stroke:{TEAL};stroke-opacity:.55;stroke-width:1.3;stroke-linecap:round}}
.ow{{fill:none;stroke:{TEAL};stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}}
.ows{{fill:none;stroke:{TEAL};stroke-opacity:.8;stroke-width:.8;stroke-linecap:round}}
.fl{{fill:none;stroke:{LIGHT};stroke-width:1.6;stroke-linecap:round;stroke-dasharray:2 22;animation:flow 1.6s linear infinite}}
.fls{{fill:none;stroke:{LIGHT};stroke-opacity:.75;stroke-width:.9;stroke-linecap:round;stroke-dasharray:1.5 16;animation:flow 2.2s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-24}}}}
.sweep{{animation:sweep 7s cubic-bezier(.4,0,.2,1) infinite}}
@keyframes sweep{{0%{{transform:translateX(-760px)}}55%,100%{{transform:translateX(0)}}}}
.pulse{{animation:pulse 2.4s ease-in-out infinite}}
@keyframes pulse{{50%{{opacity:.35}}}}
.up{{animation:up 1.1s cubic-bezier(.2,.7,.2,1) both}}
.d1{{animation-delay:.15s}} .d2{{animation-delay:.35s}} .d3{{animation-delay:.55s}}
@keyframes up{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
</style>
<linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="{DARK}"/><stop offset=".42" stop-color="{DARK}" stop-opacity=".92"/><stop offset=".62" stop-color="{DARK}" stop-opacity="0"/></linearGradient>
<linearGradient id="lit" x1="0" x2="1"><stop offset="0" stop-color="#fff"/><stop offset=".85" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<mask id="reveal"><rect class="sweep" x="{MAP[0] - 60}" y="0" width="{W}" height="{H}" fill="url(#lit)"/></mask>
<path id="ows" d="{oneway_small}"/>
<path id="ow" d="{oneway}"/>
<radialGradient id="glow" cx=".72" cy=".5" r=".55"><stop offset="0" stop-color="{TEAL}" stop-opacity=".22"/><stop offset="1" stop-color="{TEAL}" stop-opacity="0"/></radialGradient>
</defs>
<rect width="{W}" height="{H}" rx="22" fill="{DARK}"/>
<rect width="{W}" height="{H}" rx="22" fill="url(#glow)"/>
<path class="bg" d="{bg}"/>
<g mask="url(#reveal)">
<path class="mr pulse" d="{main_roads}"/>
<use href="#ows" class="ows"/>
<use href="#ows" class="fls"/>
<use href="#ow" class="ow"/>
<use href="#ow" class="fl"/>
</g>
<rect width="{W * .62:g}" height="{H}" fill="url(#fade)"/>
<g class="t">
<text class="up" x="56" y="92" fill="{TEAL}" font-size="15" font-weight="650" letter-spacing="3">TEAM A · HACKATHON 2026 · 3RD PRIZE</text>
<text class="up d1" x="52" y="176" fill="{LIGHT}" font-size="74" font-weight="760" letter-spacing="-2">one-way-or-another</text>
<text class="up d2" x="56" y="226" fill="{LIGHT}" fill-opacity=".86" font-size="23" font-weight="420">Which way does this street go? We fill in a blanked-out</text>
<text class="up d2" x="56" y="258" fill="{LIGHT}" fill-opacity=".86" font-size="23" font-weight="420">city map from street photos and the shape of the network.</text>
<g class="up d3">
<rect x="56" y="296" width="198" height="40" rx="20" fill="{DARK}" stroke="{TEAL}"/>
<circle cx="78" cy="316" r="5.5" fill="{TEAL}"/><text x="92" y="322" fill="{LIGHT}" font-size="16" font-weight="600">direction of travel</text>
<rect x="266" y="296" width="186" height="40" rx="20" fill="{DARK}" stroke="{LINE}" stroke-opacity=".35"/>
<circle cx="288" cy="316" r="5.5" fill="{AMBER}"/><text x="302" y="322" fill="{LIGHT}" font-size="16" font-weight="600">turn restrictions</text>
<rect x="464" y="296" width="140" height="40" rx="20" fill="{DARK}" stroke="{LINE}" stroke-opacity=".35"/>
<circle cx="486" cy="316" r="5.5" fill="#D9472B"/><text x="500" y="322" fill="{LIGHT}" font-size="16" font-weight="600">truck limits</text>
</g>
</g>
</svg>'''
    out = ROOT / "assets" / "banner.svg"
    out.write_text(svg, encoding="utf-8")
    print("done", out, f"{out.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
