"""Build the animated presentation (MapLibre, offline, no map tiles, no API key) -> docs/

    python src/make_deck.py          # needs run_all.py and src/experiments.py to have run
    then open docs/index.html        # or the live page on GitHub Pages

The road network itself is the map. Every number on the slides comes from results/*.json.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from common import heading, run
from paths import CHALLENGE, MLY, LINKS, PRED, PROC, RESULTS, ROOT, read

DECK_SRC = ROOT / "deck"
OUT = ROOT / "docs"
TO_LL = Transformer.from_crs(28992, 4326, always_xy=True)
TO_RD = Transformer.from_crs(4326, 28992, always_xy=True)
DIR_SIGN = re.compile(r"one-way|no-entry", re.I)
TIMING = [["Load network + index", 0.7], ["Match signs to links", 1.6], ["Driving traces (2.3 M photos)", 19.1],
          ["Turn candidates", 4.1], ["Flow solver", 0.9]]


def ll(xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    lon, lat = TO_LL.transform(xy[:, 0], xy[:, 1])
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]


def circ_mean(deg):
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def adiff(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def traces(D, g):
    """Mapillary image headings along each link. Cached in data/simulated/processed."""
    f = PROC / "evidence_traces.parquet"
    if f.exists():
        return pd.read_parquet(f)
    I = pd.read_parquet(MLY / "mly_images.parquet", columns=["sequence_id", "compass_angle", "lon", "lat"])
    x, y = TO_RD.transform(I.lon.to_numpy(), I.lat.to_numpy())
    b = shapely.total_bounds(g)
    m = (x > b[0] - 20) & (x < b[2] + 20) & (y > b[1] - 20) & (y < b[3] + 20) & I.compass_angle.notna().to_numpy()
    pts = shapely.points(x[m], y[m])
    pi, li = shapely.STRtree(g).query_nearest(pts, max_distance=8, all_matches=False)
    e, p = g[li], pts[pi]
    s, Lx = shapely.line_locate_point(e, p), shapely.length(e)
    a = shapely.get_coordinates(shapely.line_interpolate_point(e, np.clip(s - 3, 0, Lx)))
    c = shapely.get_coordinates(shapely.line_interpolate_point(e, np.clip(s + 3, 0, Lx)))
    d = adiff(I.compass_angle.to_numpy()[m][pi], heading(a[:, 0], a[:, 1], c[:, 0], c[:, 1]))
    t = pd.DataFrame({"li": li, "seq": I.sequence_id.astype(str).to_numpy()[m][pi], "fwd": d <= 30, "bwd": d >= 150})
    t = t[t.fwd | t.bwd]
    out = pd.DataFrame({"link_id": D.link_id.to_numpy()})
    for k, col in (("seq_fwd", "fwd"), ("seq_bwd", "bwd")):
        out[k] = t[t[col]].groupby("li").seq.nunique().reindex(range(len(D)), fill_value=0).to_numpy()
    out.to_parquet(f)
    return out


def hero_photo(SI, sid, h_cam):
    """One Mapillary photo of the hero sign: cached in deck/assets, else fetched with MAPILLARY_TOKEN."""
    cache = DECK_SRC / "assets" / "hero.jpg"
    meta = DECK_SRC / "assets" / "hero.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get("sign_id") == str(sid):
        return cache, json.loads(meta.read_text()).get("year")
    tok = os.environ.get("MAPILLARY_TOKEN")
    if not tok:
        return None, None
    import requests
    ims = SI[SI.sign_id == sid].copy()
    ims["o"] = adiff(ims.compass_angle, h_cam)
    iid = ims.sort_values(["o", "captured_at"], ascending=[True, False]).image_id.iloc[0]
    js = requests.get(f"https://graph.mapillary.com/{iid}", headers={"Authorization": f"OAuth {tok}"},
                      params={"fields": "thumb_1024_url,captured_at"}, timeout=30).json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(requests.get(js["thumb_1024_url"], timeout=60).content)
    year = int(pd.to_datetime(js["captured_at"], unit="ms").year) if js.get("captured_at") else None
    meta.write_text(json.dumps({"sign_id": str(sid), "image_id": str(iid), "year": year, "credit": "© Mapillary contributors, CC BY-SA 4.0"}))
    return cache, year


def numbers(nav_known, n_missing):
    s = json.loads((RESULTS / "story.json").read_text())
    f = s["final"]
    d, c, r = f["direction"], f["truck_limits"], f["turn_bans"]
    lad = s["ladder"]
    rr, cc = r["exact_sequence"], c["all"]
    return dict(
        ladder=[["Everything two-way", s["all_two_way"]["exam"]]] + [[x["step"], x["exam"]] for x in lad]
               + [["+ flow-consistency solver · final", d["accuracy"]]],
        real_model=lad[-1]["exam"], real_final=d["accuracy"], real_p=d["oneway_detection"]["precision"],
        real_r=d["oneway_detection"]["recall"], real_ft=d["direction_given_oneway_found"]["accuracy"], real_n=n_missing,
        random_attr=s["how_we_started"]["random_70_30"], spatial_attr=s["how_we_started"]["hidden_areas"],
        allb_real=s["all_two_way"]["blocks"],
        turnban=dict(new=rr["predicted"], tp=rr["tp"], pred=rr["predicted"], p=rr["precision"], f1=rr["f1"], given=nav_known["turnban"]),
        cond=dict(new=cc["predicted"], found=cc["tp"], of=cc["predicted"], values=cc["tp"], f1=round(cc["f1"], 2), given=nav_known["cond"]),
        timing=TIMING)


def main():
    L = gpd.read_parquet(PROC / "links.parquet")
    D = L.drop(columns="geometry").merge(pd.read_parquet(PROC / "split.parquet")[["link_id", "role", "block"]], on="link_id")
    known = (D.role == "known").to_numpy()
    g = L.geometry.to_crs(28992).to_numpy()
    g = np.array([x if s else shapely.reverse(x) for x, s in zip(g, D.geom_starts_at_ref)], dtype=object)     # ref -> nref
    ln = shapely.length(g)
    ref, nref = D.node_a.to_numpy(), D.node_b.to_numpy()
    idx = {v: i for i, v in enumerate(D.link_id.to_numpy())}

    sub = read(PRED / "TeamA" / LINKS).set_index("link_id").direction.reindex(D.link_id).to_numpy()
    d = np.where(known, D.dir_label.to_numpy(), sub)
    d = np.where(np.isin(d, ["B", "F", "T"]), d, "B")
    truth = read(ROOT / "data" / "simulated" / "answers" / LINKS).set_index("link_id").direction.reindex(D.link_id).to_numpy()

    S = pd.read_parquet(MLY / "mly_traffic_signs.parquet")
    S = S[S["value"].str.contains(DIR_SIGN, na=False)].reset_index(drop=True)
    sx, sy = TO_RD.transform(S.lon.to_numpy(), S.lat.to_numpy())
    tree = shapely.STRtree(g)
    spi, sli = tree.query_nearest(shapely.points(sx, sy), max_distance=15, all_matches=False)
    has_sign = np.zeros(len(D), bool)
    has_sign[sli] = True
    T = traces(D, g)
    has_trace = ((T.seq_fwd + T.seq_bwd) >= 2).to_numpy()
    stage = np.where(has_sign, 1, np.where(has_trace, 2, 3))      # signs -> driving traces -> network logic
    blk = D.block.to_numpy()
    sign_pts = ll(np.c_[sx[spi], sy[spi]])

    mid = D[["mid_x", "mid_y"]].to_numpy()
    dn = np.zeros(len(D))
    if known.any() and (~known).any():
        dk = cKDTree(mid[known]).query(mid[~known])[0]
        dn[~known] = np.clip(dk / max(np.quantile(dk, 0.98), 1), 0, 1)
    ig = np.where(known, -1.0, (stage - 1) + 0.92 * dn)

    feats = []
    for i in range(len(D)):
        c = list(g[i].coords)
        if d[i] == "T":
            c = c[::-1]                              # geometry runs in the travel direction for one-way links
        feats.append({"type": "Feature", "properties": {"d": d[i], "k": int(known[i]), "ig": round(float(ig[i]), 3), "b": int(blk[i]),
                                                         "fc": int(pd.to_numeric(D.road_class.iloc[i], errors="coerce") or 5)},
                      "geometry": {"type": "LineString", "coordinates": ll(c)}})

    def on_link(i, node, dist):
        dd = min(dist, 0.45 * ln[i])
        p = g[i].interpolate(dd if node == ref[i] else ln[i] - dd)
        return np.array([p.x, p.y])

    def node_xy(i, node):
        p = g[i].interpolate(0.0 if node == ref[i] else ln[i])
        return np.array([p.x, p.y])

    # ---- hero: a no-entry sign in the blank zone whose rule, the prediction and the answer key all agree
    hero = None
    det = pd.read_parquet(MLY / "mly_sign_detail.parquet")
    det = det[det["value"].fillna("").str.contains("no-entry")].dropna(subset=["lon", "lat"]).reset_index(drop=True)
    SI = pd.read_parquet(MLY / "mly_sign_images.parquet").dropna(subset=["compass_angle"])
    hx, hy = TO_RD.transform(det.lon.to_numpy(), det.lat.to_numpy())
    hp = shapely.points(hx, hy)
    pi, li = tree.query_nearest(hp, max_distance=12, all_matches=False)
    cams = SI.groupby("sign_id").compass_angle.agg(list)
    best = None
    for p_, i in zip(pi, li):
        sid = det.id.iloc[p_]
        if known[i] or d[i] == "B" or truth[i] != d[i] or not (70 <= ln[i] <= 260) or sid not in cams.index or len(cams[sid]) < 3:
            continue
        s = g[i].project(hp[p_])
        end = "ref" if s <= 25 else "nref" if s >= ln[i] - 25 else None
        if end is None:
            continue
        a_, b_ = (g[i].interpolate(0), g[i].interpolate(min(10, ln[i]))) if end == "ref" else (g[i].interpolate(ln[i]), g[i].interpolate(ln[i] - min(10, ln[i])))
        h_cam = circ_mean(cams[sid])
        if adiff(h_cam, float(heading(a_.x, a_.y, b_.x, b_.y))) > 40:
            continue                                             # sign faces another street
        if ("F" if end == "nref" else "T") != d[i]:
            continue
        score = len(cams[sid])
        if best is None or score > best[0]:
            best = (score, p_, i, sid, end, h_cam)
    OUT.mkdir(parents=True, exist_ok=True)
    if best:
        _, p_, i, sid, end, h_cam = best
        sp = np.array([hp[p_].x, hp[p_].y])
        snap = g[i].interpolate(g[i].project(hp[p_]))
        face = (h_cam + 180) % 360
        cone = [sp] + [sp + 22 * np.array([np.sin(np.radians(face + a)), np.cos(np.radians(face + a))]) for a in np.linspace(-24, 24, 9)] + [sp]
        col = [sp + np.array(o) for o in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5), (-.5, -.5))]
        c = list(g[i].coords)
        if d[i] == "T":
            c = c[::-1]
        mc = g[i].interpolate(0.5, normalized=True)
        mlon, mlat = TO_LL.transform(mc.x, mc.y)
        hero = dict(final=True, center=[mlon, mlat], link=ll(c), ref=ll(node_xy(i, ref[i]))[0], sign=ll(sp)[0], snap=ll([snap.x, snap.y])[0],
                    cone=ll(cone), column=ll(col), bearing=float(D.bearing_ref_to_nref.iloc[i]), sign_value=str(det["value"].iloc[p_]),
                    n_images=int(len(cams[sid])), end=end, result=str(d[i]), conf=100, agree=["answer key"],
                    length=round(float(ln[i])), photo=None, buildings=None)
        try:
            photo, year = hero_photo(SI, sid, h_cam)
            if photo:
                shutil.copy(photo, OUT / "hero.jpg")
                hero["photo"], hero["photo_year"] = "hero.jpg", year
        except Exception as e:  # noqa
            print("hero photo skipped:", type(e).__name__)

    # ---- restrictions as objects (given examples from the known zone)
    cond_obj, turnban_obj, cond_pts, turnban_pts = None, None, [], []
    C = pd.read_parquet(PROC / "truck_limits.parquet")
    C = C[C.in_nav]
    for r in C.itertuples():
        mp = g[idx[r.link_id]].interpolate(0.5, normalized=True)
        cond_pts.append(ll([mp.x, mp.y])[0])
    pick = C[C.limit_type == "HEIGHT"]
    pick = (pick if len(pick) else C).iloc[0] if len(C) else None
    if pick is not None:
        i = idx[pick.link_id]
        mp, a_, b_ = g[i].interpolate(0.5, normalized=True), g[i].interpolate(ln[i] / 2 - 2), g[i].interpolate(ln[i] / 2 + 2)
        u = np.array([b_.x - a_.x, b_.y - a_.y])
        u = u / max(np.hypot(*u), 1e-9)
        nrm, c0 = np.array([-u[1], u[0]]), np.array([mp.x, mp.y])

        def rect(center, along, across):
            return ll([center + sa * along * u + sc * across * nrm for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
        t, v = pick.limit_type, float(pick.value)
        label = f"{v / 100:.1f} m" if t == "HEIGHT" else f"{v / 1e3:.1f} t" if t == "WEIGHT" else "no trucks"
        cond_obj = dict(center=ll(c0)[0], bearing=float(heading(a_.x, a_.y, b_.x, b_.y)), link=ll(list(g[i].coords)),
                        bar=rect(c0, 0.55, 7.5), posts=[rect(c0 + s * 7.5 * nrm, 0.55, 0.55) for s in (-1, 1)],
                        type=t, label=label, clear=round(v / 100, 1) if t == "HEIGHT" else 4.0)
    R = pd.read_parquet(PROC / "turn_bans.parquet")
    for r in R.itertuples():
        if isinstance(r.turn_node, str) and r.from_link in idx:
            turnban_pts.append(ll(node_xy(idx[r.from_link], r.turn_node))[0])
    ok = R[(R.n_links == 2) & R.all_in_nav & (R.turn_node_degree >= 3) & R.turn_type.isin(["left", "right"]) & R.turn_node.notna()]
    if len(ok):
        r = ok.sort_values("turn_node_degree", ascending=False).iloc[0]
        i, j, node = idx[r.from_link], idx[r.next_link], r.turn_node

        def arc(a, b):
            p0, pc, p2 = on_link(a, node, 24), node_xy(a, node), on_link(b, node, 24)
            t = np.linspace(0, 1, 18)[:, None]
            return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p2

        def head(pts, size=3.2):
            tip, prev = pts[-1], pts[-3]
            u = (tip - prev) / max(np.hypot(*(tip - prev)), 1e-9)
            nrm = np.array([-u[1], u[0]])
            return ll([tip + u * size, tip - u * size * .4 + nrm * size * .8, tip - u * size * .4 - nrm * size * .8, tip + u * size])

        def can_leave(k):
            return d[k] == "B" or (d[k] == "F" and node == ref[k]) or (d[k] == "T" and node == nref[k])
        others = [k for k in np.flatnonzero((ref == node) | (nref == node)) if k not in (i, j) and can_leave(k)]
        fa = arc(i, j)
        u = fa[-1] - fa[-3]
        u = u / max(np.hypot(*u), 1e-9)
        nrm = np.array([-u[1], u[0]])
        turnban_obj = dict(center=ll(node_xy(i, node))[0], bearing=float((D.bearing_ref_to_nref.iloc[i] + (0 if node == nref[i] else 180)) % 360),
                       from_link=ll(list(g[i].coords)), to_link=ll(list(g[j].coords)), forbidden=ll(fa),
                       block=ll([fa[-1] + nrm * 5.5, fa[-1] - nrm * 5.5]),
                       allowed=[dict(line=ll(arc(i, k)), head=head(arc(i, k))) for k in others[:4]],
                       turn=str(r.turn_type), others=[ll(list(g[k].coords)) for k in np.flatnonzero((ref == node) | (nref == node)) if k not in (i, j)])

    n_known_rows = dict(turnban=len(read(CHALLENGE / "turn_bans.csv")), cond=len(read(CHALLENGE / "truck_limits.csv")))
    nm = int((~known).sum())
    stats = dict(links=int(len(D)), known=int(known.sum()), missing=nm,
                 oneway_known=round(float(np.isin(d[known], ["F", "T"]).mean()), 3),
                 pred_mix={k: int((d[~known] == k).sum()) for k in "BFT"},
                 stage_counts=[int(((stage == s) & ~known).sum()) for s in (1, 2, 3)], dark=0,
                 truck_limits=len(cond_pts), turn_bans=len(turnban_pts), twin=False, final=True,
                 n_sign_links=int(has_sign.sum()), n_trace_links=int(has_trace.sum()), n_dir_signs=int(len(sign_pts)),
                 mly_signs=int(len(pd.read_parquet(MLY / "mly_traffic_signs.parquet", columns=["id"]))),
                 mly_images=int(pd.read_parquet(MLY / "mly_images.parquet", columns=["lon"]).shape[0]),
                 R=numbers(n_known_rows, nm))
    b = L.total_bounds
    data = dict(bounds=[[float(b[0]), float(b[1])], [float(b[2]), float(b[3])]], links={"type": "FeatureCollection", "features": feats},
                ig_sorted=sorted(round(float(v), 3) for v in ig[~known]), signs=sign_pts, hero=hero, cond=cond_obj, turnban=turnban_obj,
                cond_pts=cond_pts, turnban_pts=turnban_pts, stats=stats)
    shutil.copytree(DECK_SRC / "vendor", OUT / "vendor", dirs_exist_ok=True)
    shutil.copy(DECK_SRC / "index.html", OUT / "index.html")
    (OUT / "data.js").write_text("window.DECK=" + json.dumps(data, separators=(",", ":")).replace("</", "<\\/") + ";", encoding="utf-8")
    (OUT / ".nojekyll").write_text("")
    print("hero:", bool(hero), "| photo:", bool(hero and hero["photo"]), "| truck gate:", bool(cond_obj), "| blocked turn:", bool(turnban_obj))
    return [OUT / "index.html"]


if __name__ == "__main__":
    run(main)
