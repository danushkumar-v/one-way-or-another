"""Direction-of-travel evidence from Mapillary, plus a transparent rule-based baseline.

    python src/direction.py                      # writes predictions/sign_rules/ (rule baseline)

Evidence (--sources, comma separated):
  mly   Mapillary no-entry / one-way signs. Facing = bearing camera -> sign of the images that
        detected it; matched to the link ahead whose direction fits.
  twin  Dual carriageway: parallel same-name/same-class link; right-hand traffic means the
        opposite carriageway is on the LEFT of travel.
  prop  One-way-ness continues along same-name, near-straight links (from known zone + evidence).
Decision per blank link: one-way if max(F,T) >= threshold, dominates the other direction by
min_ratio and beats the two-way score; else B. One-way into a non-edge dead end -> B.

The learned model in src/mly_graph.py + src/flow_repair.py reuses the evidence functions here.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data"                     # EXT / "mapillary" = data/mapillary
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
F, T, B = 0, 1, 2


# ============================================================ params
@dataclass
class Params:
    sources: str = "mly,twin,prop"
    # Mapillary
    sign_radius: float = 30.0
    max_align: float = 35.0
    w_sign: float = 1.0
    old_sign_year: int = 2019
    # twin carriageways
    w_twin: float = 0.7
    twin_max_dist: float = 40.0
    twin_max_angle: float = 20.0
    # propagation
    decay_deg2: float = 0.85
    decay_junction: float = 0.5
    max_turn: float = 35.0
    min_strength: float = 0.15
    # decision
    threshold: float = 0.5
    min_ratio: float = 2.0
    deadend_fix: bool = True

    def on(self, s):
        return s in {x.strip() for x in self.sources.split(",")}


# ============================================================ geometry helpers
def to_rd(g):
    return shapely.transform(g, lambda c: np.column_stack(_TO_RD.transform(c[:, 0], c[:, 1])))


def bearing(x0, y0, x1, y1):
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def as_line(g):
    if g.geom_type == "MultiLineString":
        m = linemerge(g)
        g = m if m.geom_type == "LineString" else LineString([c for p in g.geoms for c in p.coords])
    return LineString([c[:2] for c in g.coords])


def norm_name(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return s or None


# ============================================================ network
class Network:
    def __init__(self, nav: pd.DataFrame):
        self.df = nav.reset_index(drop=True)
        wgs = [as_line(wkt.loads(g)) for g in self.df["geom"]]
        self.ref_first = np.array([(a[1], a[0]) <= (b[1], b[0]) for a, b in ((g.coords[0], g.coords[-1]) for g in wgs)])
        self.geoms = [to_rd(g) for g in wgs]
        self.tree = STRtree(self.geoms)
        self.ref = self.df["node_a"].to_numpy()
        self.nref = self.df["node_b"].to_numpy()
        self.name = np.array([norm_name(v) for v in self.df["street_name"]], dtype=object)
        self.fc = self.df["road_class"].to_numpy()
        d = self.df["direction"].str.strip().str.upper()
        self.given = d.where(d.isin(["F", "T", "B"])).to_numpy()
        self.node_links = {}
        for i, (r, n) in enumerate(zip(self.ref, self.nref)):
            for node in (r, n):
                self.node_links.setdefault(node, []).append(i)
        b = np.array([g.bounds for g in self.geoms])
        self.bbox = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())

    def __len__(self):
        return len(self.geoms)

    def degree(self, node):
        return len(self.node_links.get(node, ()))

    def f_bearing_at(self, i, pt, half=6.0):
        g = self.geoms[i]
        d = g.project(pt)
        a, b = g.interpolate(max(0.0, d - half)), g.interpolate(min(g.length, d + half))
        if a.distance(b) < 0.5:
            a, b = Point(g.coords[0]), Point(g.coords[-1])
        brg = bearing(a.x, a.y, b.x, b.y)
        return brg if self.ref_first[i] else (brg + 180.0) % 360.0

    def chord_f(self, i):
        g = self.geoms[i]
        (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
        b = bearing(x0, y0, x1, y1)
        return b if self.ref_first[i] else (b + 180.0) % 360.0

    def outward(self, i, node, dist=10.0):
        g = self.geoms[i]
        at_start = self.ref_first[i] if node == self.ref[i] else not self.ref_first[i]
        if at_start:
            p0, p1 = Point(g.coords[0]), g.interpolate(min(dist, g.length))
        else:
            p0, p1 = Point(g.coords[-1]), g.interpolate(max(0.0, g.length - dist))
        return bearing(p0.x, p0.y, p1.x, p1.y)

    def inner_dead_end(self, i, margin=150.0):
        x0, y0, x1, y1 = self.bbox
        g = self.geoms[i]
        for node in (self.ref[i], self.nref[i]):
            if self.degree(node) > 1:
                continue
            at_start = self.ref_first[i] if node == self.ref[i] else not self.ref_first[i]
            x, y = g.coords[0] if at_start else g.coords[-1]
            if min(x - x0, x1 - x, y - y0, y1 - y) > margin:
                return True
        return False


# ============================================================ evidence: Mapillary
_KINDS = [("no_entry", "no-entry"), ("oneway_left", "one-way-left"),
          ("oneway_right", "one-way-right"), ("oneway_straight", "one-way-straight")]


def _kind(v):
    for k, pat in _KINDS:
        if isinstance(v, str) and pat in v:
            return k
    return None


def mly_signs(p: Params, kind_fn=None, with_meta=False):
    """[(kind, x, y, traffic_heading, conf)] for Mapillary signs where kind_fn(value) is not None.
    with_meta=True appends a dict (id, value, n_images, last_seen, image_id of the best facing photo)."""
    kind_fn = kind_fn or _kind
    det = pd.read_parquet(EXT / "mapillary" / "mly_sign_detail.parquet")
    det["kind"] = det["value"].map(kind_fn)
    det = det[det["kind"].notna()].copy()
    imgs = pd.read_parquet(EXT / "mapillary" / "mly_sign_images.parquet")
    imgs = imgs[imgs["sign_id"].isin(det["id"])]
    ix, iy = _TO_RD.transform(imgs["image_lon"].to_numpy(), imgs["image_lat"].to_numpy())
    imgs = imgs.assign(x=ix, y=iy)
    sx, sy = _TO_RD.transform(det["lon"].to_numpy(), det["lat"].to_numpy())
    det = det.assign(x=sx, y=sy)
    by_sign = dict(tuple(imgs.groupby("sign_id")))
    out = []
    for r in det.itertuples():
        o = by_sign.get(r.id)
        if o is None:
            continue
        dx, dy = r.x - o["x"].to_numpy(), r.y - o["y"].to_numpy()
        d = np.hypot(dx, dy)
        b = np.degrees(np.arctan2(dx, dy)) % 360            # camera -> sign = traffic heading
        comp = o["compass_angle"].to_numpy(dtype=float)
        pano = o["is_pano"].fillna(False).to_numpy(dtype=bool)
        ok = (d >= 6) & (d <= 60)
        facing = pano | np.isnan(comp) | (np.abs((comp - b + 180) % 360 - 180) <= 75)
        ok &= facing
        if not ok.any():
            continue
        w = np.minimum(d[ok], 40) / 40
        vx, vy = (w * np.sin(np.radians(b[ok]))).sum(), (w * np.cos(np.radians(b[ok]))).sum()
        R = math.hypot(vx, vy) / w.sum()
        conf = R * min(1.0, ok.sum() / 2)
        last = pd.to_datetime(r.last_seen_at, errors="coerce", utc=True)
        if pd.notna(last) and last.year < p.old_sign_year:
            conf *= 0.6
        row = (r.kind, r.x, r.y, math.degrees(math.atan2(vx, vy)) % 360, conf)
        if with_meta:
            k = np.nonzero(ok)[0][np.argmin(np.abs(d[ok] - 15))]     # photo taken ~15 m in front
            row += ({"id": r.id, "value": r.value, "n_images": int(ok.sum()),
                     "last_seen": str(last.date()) if pd.notna(last) else "",
                     "image_id": str(o["image_id"].iat[k])},)
        out.append(row)
    return out


def match_dir_sign(net: Network, kind, sx, sy, h, conf, p: Params):
    """Link a no-entry / one-way sign applies to -> (weight, link idx, F|T) or None."""
    if kind == "no_entry":
        allowed, ext = (h + 180) % 360, h
    elif kind == "oneway_straight":
        allowed = ext = h
    elif kind == "oneway_left":
        allowed = ext = (h - 90) % 360
    else:
        allowed = ext = (h + 90) % 360
    pt = Point(sx, sy)
    ex, ey = math.sin(math.radians(ext)), math.cos(math.radians(ext))
    best = None
    for i in net.tree.query(pt.buffer(p.sign_radius)):
        g = net.geoms[i]
        d = g.distance(pt)
        if d > p.sign_radius:
            continue
        mid = g.interpolate(0.5, normalized=True)
        if (mid.x - sx) * ex + (mid.y - sy) * ey < 0:
            continue
        bF = net.f_bearing_at(i, pt)
        dF, dT = angdiff(allowed, bF), angdiff(allowed, (bF + 180) % 360)
        dd, k = (dF, F) if dF <= dT else (dT, T)
        if dd > p.max_align:
            continue
        w = conf * p.w_sign * math.exp(-d / 15.0) * (1 - 0.5 * dd / p.max_align)
        if best is None or w > best[0]:
            best = (w, i, k)
    return best


def ev_mly(net: Network, p: Params):
    E = np.zeros((len(net), 3))
    for kind, sx, sy, h, conf in mly_signs(p):
        best = match_dir_sign(net, kind, sx, sy, h, conf, p)
        if best:
            E[best[1], best[2]] += best[0]
    return E


# ============================================================ evidence: twin carriageways
def ev_twin(net: Network, p: Params):
    """Returns {i: (j, F|T)}; label-dependent filtering happens in infer()."""
    out = {}
    for i, g in enumerate(net.geoms):
        if net.name[i] is None or g.length < 15:
            continue
        mid = g.interpolate(0.5, normalized=True)
        bi = net.chord_f(i)
        best = None
        for j in net.tree.query(mid.buffer(p.twin_max_dist)):
            if j == i or net.name[j] != net.name[i] or net.fc[j] != net.fc[i]:
                continue
            if {net.ref[i], net.nref[i]} == {net.ref[j], net.nref[j]}:
                continue
            gj = net.geoms[j]
            dm = gj.distance(mid)
            if dm < 4 or dm > p.twin_max_dist:
                continue
            a = angdiff(bi, net.chord_f(j))
            if min(a, 180 - a) > p.twin_max_angle:
                continue
            # must lie BESIDE i (lateral offset), not ahead/behind it on the same street
            t = gj.project(mid)
            if t <= 1.0 or t >= gj.length - 1.0:
                continue
            q = gj.interpolate(t)
            side = angdiff(bi, bearing(mid.x, mid.y, q.x, q.y))
            if abs(side - 90) > 30:
                continue
            if best is None or dm < best[0]:
                best = (dm, j)
        if best:
            j = best[1]
            q = net.geoms[j].interpolate(net.geoms[j].project(mid))
            b = math.radians(net.f_bearing_at(i, mid))
            cross = math.sin(b) * (q.y - mid.y) - math.cos(b) * (q.x - mid.x)
            out[i] = (j, F if cross > 0 else T)       # opposite carriageway on the left
    return out


# ============================================================ inference
def infer(net: Network, p: Params, log=print):
    n = len(net)
    E = np.zeros((n, 3))
    if p.on("mly"):
        E += ev_mly(net, p)
        log(f"mly: links with sign evidence {(E[:, :2].sum(1) > 0).sum()}")
    seeds = {i: d for i, d in enumerate(net.given) if isinstance(d, str)}
    if p.on("twin"):
        tw = ev_twin(net, p)
        k = 0
        for i, (j, d) in tw.items():
            if seeds.get(i) == "B" or seeds.get(j) == "B":
                continue                                  # known two-way neighbour = service road
            if E[i, B] > 0 and E[i, B] >= E[i, :2].max():
                continue                                  # evidence says two-way
            E[i, d] += p.w_twin
            k += 1
        log(f"twin: links voted {k}")

    P = np.zeros((n, 2))
    if p.on("prop"):
        heap, best = [], {}
        for i, lab in seeds.items():
            if lab in ("F", "T"):
                heap.append((-1.0, i, F if lab == "F" else T))
        for i in range(n):
            if i not in seeds:
                hi, lo = E[i, :2].max(), E[i, :2].min()
                if hi - lo >= p.threshold and hi > E[i, B]:
                    heap.append((-min(1.0, hi - lo), i, int(E[i, :2].argmax())))
        heapq.heapify(heap)
        while heap:
            s, i, d = heapq.heappop(heap)
            s = -s
            if s < best.get((i, d), 0.0) - 1e-12:
                continue
            exit_node = net.nref[i] if d == F else net.ref[i]
            for node in (net.ref[i], net.nref[i]):
                links = net.node_links.get(node, ())
                ns = s * (p.decay_deg2 if len(links) == 2 else p.decay_junction)
                if ns < p.min_strength:
                    continue
                oi = net.outward(i, node)
                for j in links:
                    if j == i or j in seeds or net.name[j] is None or net.name[j] != net.name[i]:
                        continue
                    if 180 - angdiff(oi, net.outward(j, node)) > p.max_turn:
                        continue
                    if node == exit_node:
                        dj = F if node == net.ref[j] else T
                    else:
                        dj = F if node == net.nref[j] else T
                    if ns > best.get((j, dj), 0.0):
                        best[(j, dj)] = ns
                        P[j, dj] = ns
                        heapq.heappush(heap, (-ns, j, dj))
        log(f"prop: links reached {(P.sum(1) > 0).sum()}")

    S = E.copy()
    S[:, :2] += P
    pred = net.given.copy()
    for i in range(n):
        if i in seeds:
            continue
        sF, sT, sB = S[i]
        hi, lo = max(sF, sT), min(sF, sT)
        lab = "B"
        if hi >= p.threshold and hi >= p.min_ratio * lo and hi > sB:
            lab = "F" if sF >= sT else "T"
            if p.deadend_fix and net.inner_dead_end(i):
                lab = "B"
        pred[i] = lab
    return pred, S


# ============================================================ CLI
def main():
    from paths import CHALLENGE, LINKS, PRED, read
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="sign_rules", help="output folder under predictions/")
    for f in fields(Params):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(flag, type={"float": float, "int": int, "str": str}[f.type], default=f.default)
    a = ap.parse_args()
    p = Params(**{f.name: getattr(a, f.name) for f in fields(Params)})
    nav = read(CHALLENGE / LINKS)
    net = Network(nav)
    pred, S = infer(net, p, print)
    out = nav.copy()
    out["direction"] = pred
    out_dir = PRED / a.team
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / LINKS, index=False)
    (out_dir / "params.json").write_text(json.dumps(p.__dict__, indent=2))
    blank = pd.isna(net.given)
    print("predicted on blank links:", pd.Series(pred[blank]).value_counts().to_dict())
    print("done", out_dir / LINKS)


if __name__ == "__main__":
    main()
