"""Truck access restrictions (truck limits: HEIGHT / WEIGHT / NO TRUCKS) from Mapillary signs.

    python src/trucks.py                    # -> predictions/TeamA/truck_limits.csv
    python src/trucks.py --eval-known       # how many given rows in the known zone the rules recover

Method
  1. Sign heading = bearing camera -> sign (shared with direction).
  2. The restriction applies to the link AHEAD of the sign that runs along the heading (either
     digitising direction; truck limits are not directional).
     HEIGHT: prefer, within --height-reach m ahead, a tunnel link or a link passing under a
     bridge link (where the clearance physically is); else the first link ahead.
  3. Values: Mapillary sign classes carry no number. Value = the most common value among the given
     rows of that type (--learn-defaults), else a typical value (HEIGHT 410 cm, WEIGHT 30 t).
  4. One row per (link, type); original rows kept, new rows appended.
     Units: HEIGHT in cm, WEIGHT in kg (1 t = 1,000 kg).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction import EXT, ROOT, Network, Params, _TO_RD, angdiff, mly_signs  # noqa: E402
from paths import CHALLENGE, LIMITS, LINKS, PRED  # noqa: E402

KIND = {
    "regulatory--height-limit": "HEIGHT", "information--height-limit": "HEIGHT",
    "regulatory--weight-limit": "WEIGHT", "regulatory--weight-limit-per-axle": "WEIGHT",
    "regulatory--weight-limit-with-trucks": "WEIGHT",
    "regulatory--no-heavy-goods-vehicles": "NO TRUCKS", "regulatory--no-heavy-goods-vehicles-or-buses": "NO TRUCKS",
}



def kind_of(v):
    return KIND.get(re.sub(r"--g\d+$", "", v)) if isinstance(v, str) else None


def wkt_inner(g):
    return g[g.index("(") + 1: g.rindex(")")].strip()


def yes(v):
    return isinstance(v, str) and v.strip().upper() in ("Y", "YES", "T", "TRUE", "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="TeamA", help="output folder under predictions/")
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--max-align", type=float, default=35.0)
    ap.add_argument("--height-reach", type=float, default=150.0)
    ap.add_argument("--min-conf", type=float, default=0.2)
    ap.add_argument("--default-height-cm", type=int, default=410)
    ap.add_argument("--default-weight-t", type=float, default=30.0)
    ap.add_argument("--learn-defaults", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--eval-known", action="store_true", help="no output: per type, rows in the known zone matching GIVEN rows (counts)")
    ap.add_argument("--types", default="WEIGHT", help="comma-separated types to write, e.g. WEIGHT,HEIGHT,NO TRUCKS")
    a = ap.parse_args()
    run(a, CHALLENGE / LINKS, CHALLENGE / LIMITS, PRED / a.team, print)


def run(a, nav_src, cond_src, out_dir, log):
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])   # noqa: E731
    nav = rd(nav_src)
    net = Network(nav)
    tunnel = np.array([yes(v) for v in nav["tunnel"]])
    bridge_idx = [i for i, v in enumerate(nav["bridge"]) if yes(v)]
    from shapely.strtree import STRtree
    btree = STRtree([net.geoms[i] for i in bridge_idx]) if bridge_idx else None

    def under_bridge(i):
        if btree is None or tunnel[i]:
            return tunnel[i]
        g = net.geoms[i]
        for k in btree.query(g):
            if g.crosses(net.geoms[bridge_idx[k]]):
                return True
        return False

    want = {t.strip().upper() for t in a.types.split(",")}
    # default values: most common value among the provider's own given rows of that type (learned from the
    # training labels, like the direction model); general-knowledge fallback when none are given
    _orig = rd(cond_src)
    _t = _orig["limit_type"].str.strip().str.upper()
    _v = pd.to_numeric(_orig["limit_value"], errors="coerce")
    for typ, attr, scale in (("HEIGHT", "default_height_cm", 1), ("WEIGHT", "default_weight_t", 1e-3)):
        vals = _v[_t == typ].dropna()
        if a.learn_defaults and len(vals):
            setattr(a, attr, type(getattr(a, attr))(vals.mode().iat[0] * scale))
            log(f"default {typ} learned from {len(vals)} given rows")
    found = {}                                     # (link_idx, type) -> (conf, value)
    signs = mly_signs(Params(), kind_fn=kind_of)
    log(f"truck-related signs with heading: {len(signs)}")
    for typ, sx, sy, h, conf in signs:
        if typ not in want or conf < a.min_conf:
            continue
        ux, uy = math.sin(math.radians(h)), math.cos(math.radians(h))
        pt = Point(sx, sy)
        reach = a.height_reach if typ == "HEIGHT" else a.radius
        cands = []
        for i in net.tree.query(pt.buffer(reach)):
            g = net.geoms[i]
            d = g.distance(pt)
            mid = g.interpolate(0.5, normalized=True)
            ahead = (mid.x - sx) * ux + (mid.y - sy) * uy
            if ahead < -5 or d > reach:
                continue
            bF = net.f_bearing_at(i, pt if d <= a.radius else mid)
            dd = min(angdiff(h, bF), angdiff(h, (bF + 180) % 360))
            if dd > a.max_align:
                continue
            cands.append((d, ahead, i))
        if not cands:
            continue
        near = [c for c in cands if c[0] <= a.radius]
        pick = None
        if typ == "HEIGHT":
            ub = [c for c in cands if under_bridge(c[2])]
            if ub:
                pick = min(ub, key=lambda c: c[1])[2]
        if pick is None and near:
            pick = min(near, key=lambda c: (c[0] > 3, c[1]))[2]
        if pick is None:
            continue
        val = None
        if typ == "HEIGHT":
            value = str(int(round(val * 100))) if val is not None else str(a.default_height_cm)
        elif typ == "WEIGHT":
            value = str(int(round((val if val is not None else a.default_weight_t) * 1_000)))
        else:
            value = None
        key = (pick, typ)
        if key not in found or conf > found[key][0]:
            found[key] = (conf, value)
    log(f"candidate restrictions: {len(found)}  by type: "
        f"{pd.Series([k[1] for k in found]).value_counts().to_dict() if found else {}}")

    # ---- write, mimicking the original file's conventions
    orig = rd(cond_src)
    tcol, vcol = "limit_type", "limit_value"
    spelled = {t.strip().upper(): t for t in orig[tcol].dropna()}          # keep original spelling/casing
    nt_vals = orig.loc[orig[tcol].str.strip().str.upper() == "NO TRUCKS", vcol]
    no_trucks_value = nt_vals.iat[0] if len(nt_vals) else None                 # same format as given rows
    have = {(l, t.strip().upper()) for l, t in zip(orig["link_id"], orig[tcol])}
    admin_col = [c for c in orig.columns if c.startswith("admin")]
    admin = orig[admin_col[0]].mode().iat[0] if admin_col and orig[admin_col[0]].notna().any() else None
    if a.eval_known:
        known = nav["direction"].isin(["B", "F", "T"]).to_numpy()
        gv = {(l, t.strip().upper()): v for l, t, v in zip(orig["link_id"], orig[tcol], orig[vcol])}
        gk = [(l, t) for (l, t) in gv]
        kl = set(nav.loc[known, "link_id"])
        print("type: our rows in known zone / same link+type as given / same value; given rows of that type in known zone")
        for t in sorted({k[1] for k in found} | {k[1] for k in gk}):
            ours = [(nav.at[i, "link_id"], v) for (i, ty), (c, v) in found.items() if ty == t and known[i]]
            m = [(l, v) for l, v in ours if (l, t) in gv]
            mv = sum(1 for l, v in m if str(gv[(l, t)]) == str(v))
            print(f"  {t}: {len(ours)} / {len(m)} / {mv}; given {sum(1 for l, ty in gk if ty == t and l in kl)}")
        return
    rows = []
    for (i, typ), (conf, value) in sorted(found.items(), key=lambda kv: -kv[1][0]):
        lid = nav.at[i, "link_id"]
        if (lid, typ) in have:
            continue
        r = {c: None for c in orig.columns}
        r.update({"link_id": lid, tcol: spelled.get(typ, typ),
                  vcol: value if typ != "NO TRUCKS" else no_trucks_value,
                  "geom": f"MULTILINESTRING (({wkt_inner(nav.at[i, 'geom'])}))"})
        if admin_col:
            r[admin_col[0]] = admin
        rows.append(r)
    out = pd.concat([orig, pd.DataFrame(rows, columns=orig.columns)], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / LIMITS
    out.to_csv(out_csv, index=False)
    log(f"new rows: {len(rows)}")
    print("done", out_csv)


if __name__ == "__main__":
    main()
