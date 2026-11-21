"""Extra Mapillary evidence for direction: which SIDE of the road signs and road markings stand on.

Right-hand traffic: a sign faces the traffic on its right-hand side of the road; stop lines, give-way
triangles and lane arrows are painted in the lane that carries traffic toward the junction. So for each
detected object near a link we measure its signed lateral offset from the centre line:
    right of the ref->nref direction  -> supports F traffic
    left  of the ref->nref direction  -> supports T traffic
One-way streets get objects on one side only; two-way streets on both. Fully vectorised (shapely 2),
no per-object Python loop, so it scales linearly with objects and links.

Used by src/mly_graph.py configs *_SIDE (pairs 'side_sign', 'side_mark', 'stopline').
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from direction import EXT, _TO_RD

SIGN_KEEP = ("regulatory--", "warning--")              # signs that face road traffic
SIGN_SKIP = ("bicycles", "mopeds", "pedestrians")       # these stand on cycle/foot paths beside the road
MARK_ARROW = "marking--discrete--arrow"
MARK_STOP = ("marking--discrete--stop-line", "marking--discrete--give-way-row", "marking--discrete--give-way-single",
             "marking--discrete--text--stop")


def _side_counts(net, x, y, w, max_dist=15.0, min_off=1.5):
    """F/T support from objects at (x, y) with weights w."""
    n = len(net)
    geoms = np.array(net.geoms, dtype=object)
    tree = STRtree(geoms)
    pts = shapely.points(x, y)
    pi, li = tree.query_nearest(pts, max_distance=max_dist, all_matches=False)
    if len(pi) == 0:
        return np.zeros(n), np.zeros(n)
    g, p = geoms[li], pts[pi]
    s = shapely.line_locate_point(g, p)
    L = shapely.length(g)
    a = shapely.line_interpolate_point(g, np.clip(s - 1.0, 0, L))
    b = shapely.line_interpolate_point(g, np.clip(s + 1.0, 0, L))
    q = shapely.line_interpolate_point(g, s)
    tx, ty = shapely.get_x(b) - shapely.get_x(a), shapely.get_y(b) - shapely.get_y(a)
    ox, oy = shapely.get_x(p) - shapely.get_x(q), shapely.get_y(p) - shapely.get_y(q)
    cross = tx * oy - ty * ox                       # >0: left of digitised direction
    off = np.hypot(ox, oy)
    ok = (off >= min_off) & (np.hypot(tx, ty) > 0.1)
    right_of_digit = cross < 0
    rf = np.asarray(net.ref_first, bool)[li]
    supports_F = np.where(rf, right_of_digit, ~right_of_digit)
    ww = np.asarray(w, float)[pi]
    F = np.bincount(li[ok & supports_F], weights=ww[ok & supports_F], minlength=n)
    T = np.bincount(li[ok & ~supports_F], weights=ww[ok & ~supports_F], minlength=n)
    return F, T


def _load(name, net, margin=50):
    d = pd.read_parquet(EXT / "mapillary" / f"{name}.parquet", columns=["value", "lon", "lat", "last_seen_at"])
    x, y = _TO_RD.transform(d["lon"].to_numpy(), d["lat"].to_numpy())
    x0, y0, x1, y1 = net.bbox
    m = (x > x0 - margin) & (x < x1 + margin) & (y > y0 - margin) & (y < y1 + margin)
    d = d.loc[m].assign(x=np.asarray(x)[m], y=np.asarray(y)[m])
    yr = pd.to_datetime(d["last_seen_at"], errors="coerce").dt.year
    d["w"] = np.where(yr.fillna(2020) < 2019, 0.6, 1.0)
    return d


def side_pairs(net):
    ts = _load("mly_traffic_signs", net)
    ts = ts[ts["value"].str.startswith(SIGN_KEEP) & ~ts["value"].str.contains("|".join(SIGN_SKIP))]
    pt = _load("mly_points", net)
    arrows = pt[pt["value"].str.startswith(MARK_ARROW)]
    stops = pt[pt["value"].isin(MARK_STOP)]
    return {
        "side_sign": _side_counts(net, ts["x"].to_numpy(), ts["y"].to_numpy(), ts["w"].to_numpy()),
        "side_mark": _side_counts(net, arrows["x"].to_numpy(), arrows["y"].to_numpy(), arrows["w"].to_numpy()),
        "stopline": _side_counts(net, stops["x"].to_numpy(), stops["y"].to_numpy(), stops["w"].to_numpy()),
    }
