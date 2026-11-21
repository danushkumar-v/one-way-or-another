"""Restricted Driving Manoeuvres (no-left / no-right / no-U-turn) from Mapillary turn signs.

Sign-rule baseline (the learned model is src/turn_model.py; it reuses the helpers here):
    python src/turn_rules.py                    # -> predictions/sign_rules/turn_bans.csv

Method
  1. Sign heading = bearing camera -> sign over the images that detected it (shared with direction).
  2. Approach: the link the viewer drives on, aligned with the heading, whose node AHEAD of the sign
     is a junction you can drive into. That link = "from", that node = "via".
  3. Exits at the via node, classified by turn angle relative to the heading:
     left (-160..-30), straight (|a| < 30), right (30..160).
  4. Sign -> prohibited classes:
       no-left-turn: L   no-right-turn: R   no-turns: L,R   no-straight-through: S   no-u-turn: [from]
       turn-left (only): S,R   turn-right: S,L   go-straight: L,R   go-straight-or-turn-left: R
       go-straight-or-turn-right: L   turn-left-or-right: S   turn-left-ahead: S,R   turn-right-ahead: S,L
     Exits you cannot legally enter anyway (one-way against you) are skipped - no map would code them.
  5. Votes over repeated detections are summed; rows above --min-score are written
     after the original rows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction import ROOT, Network, Params, angdiff, bearing, mly_signs  # noqa: E402

PROHIBIT = {
    "no-left-turn": "L", "no-right-turn": "R", "no-turns": "LR", "no-straight-through": "S", "no-u-turn": "U",
    "turn-left-ahead": "SR", "turn-right-ahead": "SL", "turn-left": "SR", "turn-right": "SL",
    "go-straight-or-turn-left": "R", "go-straight-or-turn-right": "L", "go-straight": "LR",
    "turn-left-or-right": "S",
}
AHEAD_TYPES = {"turn-left-ahead", "turn-right-ahead"}     # often placed beyond the junction


def kind_of(v):
    if not isinstance(v, str) or not v.startswith("regulatory--"):
        return None
    k = re.sub(r"--g\d+$", "", v[len("regulatory--"):])
    return k if k in PROHIBIT else None


def wkt_inner(g):
    return g[g.index("(") + 1: g.rindex(")")].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="sign_rules", help="output folder under predictions/")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--max-ahead", type=float, default=80.0)
    ap.add_argument("--use-ahead-signs", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--eval-known", action="store_true",
                    help="no output file: per sign kind, how many rows inside the known zone match the GIVEN rows (counts only)")
    ap.add_argument("--kinds", default=None, help="comma-separated sign kinds to use (default: all in PROHIBIT)")
    ap.add_argument("--dir-path", default=None, help="links prediction CSV used for one-way awareness (default: predictions/TeamA/)")
    ap.add_argument("--use-mandatory", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    a = ap.parse_args()

    from paths import CHALLENGE, LINKS, PRED, TURNS
    dir_src = Path(a.dir_path) if a.dir_path else PRED / "TeamA" / LINKS
    run(a, CHALLENGE / LINKS, CHALLENGE / TURNS, dir_src, PRED / a.team, print)


def run(a, nav_src, turnban_src, dir_src, out_dir, log):
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])   # noqa: E731
    nav = rd(nav_src)
    net = Network(nav)
    # direction for legality checks: filled prediction if available, else given (blank = B)
    dirs = rd(dir_src)["direction"].to_numpy() if dir_src.exists() else nav["direction"].fillna("B").to_numpy()
    if not dir_src.exists():
        log(f"note: {dir_src} not found, treating blank direction as B")
    dirs = np.array([d if d in ("F", "T", "B") else "B" for d in dirs], dtype=object)

    def can_enter(i, n):
        return dirs[i] == "B" or (dirs[i] == "F" and n == net.nref[i]) or (dirs[i] == "T" and n == net.ref[i])

    def can_leave(i, n):
        return dirs[i] == "B" or (dirs[i] == "F" and n == net.ref[i]) or (dirs[i] == "T" and n == net.nref[i])

    def node_xy(i, n):
        g = net.geoms[i]
        at_start = net.ref_first[i] if n == net.ref[i] else not net.ref_first[i]
        return g.coords[0] if at_start else g.coords[-1]

    signs = mly_signs(Params(), kind_fn=kind_of)
    log(f"turn-related signs with heading: {len(signs)}")
    votes = defaultdict(float)
    vkind = defaultdict(set)
    matched = 0
    for kind, sx, sy, h, conf in signs:
        if a.kinds and kind not in a.kinds.split(","):
            continue
        if kind in AHEAD_TYPES and not a.use_ahead_signs and not a.kinds:
            continue
        if kind not in ("no-left-turn", "no-right-turn", "no-turns", "no-straight-through", "no-u-turn") and not a.use_mandatory:
            continue
        ux, uy = math.sin(math.radians(h)), math.cos(math.radians(h))
        lo = -35.0 if kind in AHEAD_TYPES else -10.0
        best = None
        for i in net.tree.query(Point(sx, sy).buffer(a.radius)):
            if net.geoms[i].distance(Point(sx, sy)) > a.radius:
                continue
            for n in (net.ref[i], net.nref[i]):
                deg = net.degree(n)
                if deg < (2 if kind == "no-u-turn" else 3) or not can_enter(i, n):
                    continue
                nx_, ny_ = node_xy(i, n)
                ahead = (nx_ - sx) * ux + (ny_ - sy) * uy
                lateral = abs((nx_ - sx) * uy - (ny_ - sy) * ux)
                if not (lo <= ahead <= a.max_ahead) or lateral > 30:
                    continue
                inbound = (net.outward(i, n) + 180) % 360
                if angdiff(inbound, h) > 40:
                    continue
                score = conf * math.exp(-abs(ahead - 15) / 40) * (1 - angdiff(inbound, h) / 80)
                if best is None or score > best[0]:
                    best = (score, i, n, inbound)
        if best is None:
            continue
        matched += 1
        score, i, n, inbound = best
        banned = PROHIBIT[kind]
        if "U" in banned:
            votes[(net.df.at[i, "link_id"],)] += score
            vkind[(net.df.at[i, "link_id"],)].add(kind)
            continue
        for k in net.node_links.get(n, ()):
            if k == i or not can_leave(k, n):
                continue
            rel = (net.outward(k, n) - inbound + 180) % 360 - 180
            cls = "S" if abs(rel) < 30 else ("L" if -160 <= rel <= -30 else ("R" if 30 <= rel <= 160 else None))
            if cls and cls in banned:
                votes[(net.df.at[i, "link_id"], net.df.at[k, "link_id"])] += score
                vkind[(net.df.at[i, "link_id"], net.df.at[k, "link_id"])].add(kind)
    log(f"signs matched to an approach: {matched}, candidate manoeuvres: {len(votes)}")

    # ---- write: original rows + new rows
    orig = rd(turnban_src)
    have = set()
    for s in orig["link_ids"].dropna():
        try:
            have.add(tuple(str(x) for x in json.loads(s)))
        except Exception:
            pass
    if a.eval_known:
        known = set(nav.loc[nav["direction"].isin(["B", "F", "T"]), "link_id"])
        stats = defaultdict(lambda: [0, 0])
        for seq, sc in votes.items():
            if sc < a.min_score or not all(l in known for l in seq):
                continue
            for k in vkind[seq]:
                stats[k][0] += 1
                stats[k][1] += seq in have
        tot = [sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())]
        print("kind: rows in known zone / matching given")
        for k, (n_, m_) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k}: {n_} / {m_}")
        print(f"  ALL: {tot[0]} / {tot[1]}  (given rows in file: {len(have)})")
        ours = [seq for seq, sc in votes.items() if sc >= a.min_score and all(l in known for l in seq)]
        idx = {l: i for i, l in enumerate(nav["link_id"])}
        def via(seq):
            i = idx.get(seq[0])
            if i is None:
                return None
            if len(seq) == 1:
                return "U"
            k = idx.get(seq[1])
            c = {net.ref[i], net.nref[i]} & {net.ref[k], net.nref[k]} if k is not None else set()
            return next(iter(c)) if c else None
        ofrom = {q[0] for q in ours}
        oto = {q[-1] for q in ours}
        ovia = {via(q) for q in ours}
        sx = [xy for xy in ((s_[1], s_[2]) for s_ in signs)]
        sx = np.array(sx)
        near = 0
        for q in have:
            i = idx.get(q[0])
            if i is None:
                continue
            g = net.geoms[i]
            d = np.hypot(sx[:, 0] - g.centroid.x, sx[:, 1] - g.centroid.y).min()
            near += d < 60
        print(f"given rows: same from-link as ours {sum(q[0] in ofrom for q in have)}, same to-link {sum(q[-1] in oto for q in have)}, "
              f"same junction {sum(via(q) in ovia for q in have)}, Mapillary turn sign within 60 m of from-link {near}")
        return
    admin = orig["city"].mode().iat[0] if "city" in orig and orig["city"].notna().any() else ""
    geom_of = dict(zip(nav["link_id"], nav["geom"]))
    new = []
    for seq, sc in sorted(votes.items(), key=lambda kv: -kv[1]):
        if sc < a.min_score or seq in have:
            continue
        geom = "MULTILINESTRING (" + ",".join(f"({wkt_inner(geom_of[l])})" for l in seq) + ")"
        new.append({"link_ids": json.dumps(list(seq), separators=(",", ":")), "city": admin, "geom": geom})
    out = pd.concat([orig, pd.DataFrame(new, columns=orig.columns)], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "turn_bans.csv"
    out.to_csv(out_csv, index=False)
    log(f"new turn-ban rows: {len(new)} (U-turn {sum(1 for r in new if r['link_ids'].count(',') == 0)})")
    print("done")
    print(out_csv)


if __name__ == "__main__":
    main()
