"""Learned turn restrictions: score every (from-link, to-link) turn at every junction.

Why: maps code many restrictions that have no turn sign (e.g. left turns across a dual-carriageway
median). A sign-only method cannot find those. Here each possible turn is a row with
  * geometry/attributes: turn class and angle, one-way flags, road classes, same street, junction degree,
    dual-carriageway (twin) flags, link lengths, internal short links
  * Mapillary: turn signs facing this approach whose ban covers this turn class, other regulatory signs
    near the junction, lane-arrow markings on the approach (left arrow painted -> left allowed),
    traffic lights near the junction
and LightGBM learns which turns the map restricts, from the restrictions already given in the known zone.

    python src/turn_model.py --write                # add predicted rows -> predictions/TeamA/ (needs its links file)
    python src/turn_model.py --cv                   # 5-block CV inside the known zone
    python src/turn_model.py --eval                 # train on the known zone, score the missing zone vs the answer key
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction import EXT, ROOT, _TO_RD, Network, Params, angdiff, ev_twin, mly_signs  # noqa: E402
from turn_rules import PROHIBIT, kind_of, wkt_inner  # noqa: E402

LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, scale_pos_weight=20, verbose=-1, random_state=0)


def candidates(net, dirs):
    def can_enter(i, n):
        return dirs[i] == "B" or (dirs[i] == "F" and n == net.nref[i]) or (dirs[i] == "T" and n == net.ref[i])

    def can_leave(i, n):
        return dirs[i] == "B" or (dirs[i] == "F" and n == net.ref[i]) or (dirs[i] == "T" and n == net.nref[i])

    rows = []
    for n, links in net.node_links.items():
        if len(links) < 3:
            continue
        for i in links:
            if net.ref[i] == net.nref[i] or not can_enter(i, n):
                continue
            inbound = (net.outward(i, n) + 180) % 360
            exits = []
            for k in links:
                if k == i or not can_leave(k, n):
                    continue
                rel = (net.outward(k, n) - inbound + 180) % 360 - 180
                exits.append((k, rel))
            for k, rel in exits:
                rows.append((i, k, n, rel, len(exits), sum(1 for _, r in exits if r < rel), inbound))
    return pd.DataFrame(rows, columns=["i", "k", "n", "rel", "n_exits", "rank_from_left", "inbound"])


def node_xy(net, n):
    i = net.node_links[n][0]
    g = net.geoms[i]
    at_start = net.ref_first[i] if n == net.ref[i] else not net.ref_first[i]
    return g.coords[0] if at_start else g.coords[-1]


def features(net, nav, dirs, C):
    num = lambda c: pd.to_numeric(nav[c], errors="coerce").to_numpy()   # noqa: E731
    fc = num("road_class")
    length = np.array([g.length for g in net.geoms])
    ow = np.isin(dirs, ["F", "T"]).astype(int)
    twin = np.zeros(len(net), int)
    partner = {}
    for i, (j, _) in ev_twin(net, Params()).items():
        twin[i] = 1
        partner[i] = j
    i, k = C["i"].to_numpy(), C["k"].to_numpy()
    rel = C["rel"].to_numpy()
    cls = np.where(np.abs(rel) < 30, 0, np.where(rel < 0, -1, 1))          # -1 L, 0 S, 1 R
    same = np.array([net.name[a] is not None and net.name[a] == net.name[b] for a, b in zip(i, k)], int)
    F = pd.DataFrame({
        "rel": rel, "abs_rel": np.abs(rel), "cls": cls, "deg": [net.degree(n) for n in C["n"]],
        "n_exits": C["n_exits"], "rank_from_left": C["rank_from_left"],
        "fc_from": fc[i], "fc_to": fc[k], "fc_diff": fc[i] - fc[k], "ow_from": ow[i], "ow_to": ow[k],
        "twin_from": twin[i], "twin_to": twin[k], "len_from": length[i], "len_to": length[k], "same_name": same,
        "short_to": (length[k] < 30).astype(int), "short_from": (length[i] < 30).astype(int),
    })
    # node-level: twin links at the node, min link length at the node
    node_twin = {n: sum(twin[j] for j in ls) for n, ls in net.node_links.items()}
    node_min = {n: min(length[j] for j in ls) for n, ls in net.node_links.items()}
    F["node_twin"] = [node_twin[n] for n in C["n"]]
    # median crossings: exit is the approach's own opposite carriageway, or both sides are dual roads
    F["to_is_partner"] = np.array([partner.get(a) == b or partner.get(b) == a for a, b in zip(i, k)], int)
    F["twin_cross"] = ((twin[i] == 1) & (twin[k] == 1) & (same == 0)).astype(int)
    node_ow = {n: sum(ow[j] for j in ls) for n, ls in net.node_links.items()}
    F["node_oneways"] = [node_ow[n] for n in C["n"]]
    F["node_oneway_share"] = F["node_oneways"] / F["deg"]
    F["node_minlen"] = [node_min[n] for n in C["n"]]

    # ---- Mapillary: turn signs facing this approach, whose ban covers this class
    key = {(a, n): r for r, (a, n) in enumerate(zip(i, C["n"]))}
    appr_sign = {}                                                   # (i, n) -> {kind: score}
    for kind, sx, sy, h, conf in mly_signs(Params(), kind_fn=kind_of):
        best = None
        for a in net.tree.query(Point(sx, sy).buffer(25)):
            if net.geoms[a].distance(Point(sx, sy)) > 25:
                continue
            for n in (net.ref[a], net.nref[a]):
                if (a, n) not in key:
                    continue
                nx_, ny_ = node_xy(net, n)
                ux, uy = math.sin(math.radians(h)), math.cos(math.radians(h))
                ahead = (nx_ - sx) * ux + (ny_ - sy) * uy
                if not (-35 <= ahead <= 80):
                    continue
                inbound = (net.outward(a, n) + 180) % 360
                if angdiff(inbound, h) > 40:
                    continue
                sc = conf * math.exp(-abs(ahead - 15) / 40) * (1 - angdiff(inbound, h) / 80)
                if best is None or sc > best[0]:
                    best = (sc, a, n)
        if best:
            d = appr_sign.setdefault((best[1], best[2]), {})
            d[kind] = d.get(kind, 0) + best[0]
    cname = np.array(["L", "S", "R"])[cls + 1]
    ban_this, ban_other, any_sign = np.zeros(len(C)), np.zeros(len(C)), np.zeros(len(C))
    groups = {"bans": ("no-left-turn", "no-right-turn", "no-turns", "no-straight-through"),
              "only": ("turn-left", "turn-right", "go-straight", "go-straight-or-turn-left", "go-straight-or-turn-right", "turn-left-or-right"),
              "ahead": ("turn-left-ahead", "turn-right-ahead"), "uturn": ("no-u-turn",)}
    G = {g: np.zeros(len(C)) for g in groups}
    for r, (a, n) in enumerate(zip(i, C["n"])):
        d = appr_sign.get((a, n))
        if not d:
            continue
        for kd, sc in d.items():
            any_sign[r] += sc
            (ban_this if cname[r] in PROHIBIT[kd] else ban_other)[r] += sc
            for g, ks in groups.items():
                if kd in ks:
                    G[g][r] += sc
    F["sign_bans_this"], F["sign_bans_other"], F["sign_any"] = ban_this, ban_other, any_sign
    for g, v in G.items():
        F[f"sign_{g}"] = v

    # ---- Mapillary points: lane arrows on the approach, traffic lights at the junction
    pt = pd.read_parquet(EXT / "mapillary" / "mly_points.parquet", columns=["value", "lon", "lat"])
    x, y = _TO_RD.transform(pt["lon"].to_numpy(), pt["lat"].to_numpy())
    x0, y0, x1, y1 = net.bbox
    m = (x > x0 - 50) & (x < x1 + 50) & (y > y0 - 50) & (y < y1 + 50)
    pt = pt.loc[m].assign(x=np.asarray(x)[m], y=np.asarray(y)[m])
    from scipy.spatial import cKDTree
    nodes = list(net.node_links)
    nxy = np.array([node_xy(net, n) for n in nodes])
    nidx = {n: j for j, n in enumerate(nodes)}
    ntree = cKDTree(nxy)
    arrows = {"left": ("arrow--left", "split-left"), "right": ("arrow--right", "split-right"), "straight": ("arrow--straight", "or-straight")}
    # arrow -> nearest link (<8 m) and that link's nearer end node (<60 m) = an approach marking
    from shapely.strtree import STRtree
    import shapely
    geoms = np.array(net.geoms, dtype=object)
    lt = STRtree(geoms)
    for name, pats in arrows.items():
        sel = pt[pt["value"].str.contains("|".join(pats), regex=True) & pt["value"].str.startswith("marking--discrete--arrow")]
        cnt = {}
        if len(sel):
            P = shapely.points(sel["x"].to_numpy(), sel["y"].to_numpy())
            pi, li = lt.query_nearest(P, max_distance=8, all_matches=False)
            for p_, l_ in zip(pi, li):
                px, py = sel["x"].iat[p_], sel["y"].iat[p_]
                best = min((net.ref[l_], net.nref[l_]), key=lambda n: math.dist(node_xy(net, n), (px, py)))
                if math.dist(node_xy(net, best), (px, py)) <= 60:
                    cnt[(l_, best)] = cnt.get((l_, best), 0) + 1
        F[f"arrow_{name}"] = [cnt.get((a, n), 0) for a, n in zip(i, C["n"])]
    F["arrow_this"] = np.select([cls == -1, cls == 0, cls == 1], [F["arrow_left"], F["arrow_straight"], F["arrow_right"]])
    F["arrow_any"] = F["arrow_left"] + F["arrow_straight"] + F["arrow_right"]
    F["arrow_missing_this"] = ((F["arrow_any"] > 0) & (F["arrow_this"] == 0)).astype(int)
    tl = pt[pt["value"].str.startswith("object--traffic-light")]
    tl_cnt = np.zeros(len(nodes))
    if len(tl):
        for lst in ntree.query_ball_point(tl[["x", "y"]].to_numpy(), r=30):
            for j in lst:
                tl_cnt[j] += 1
    F["traffic_lights"] = [tl_cnt[nidx[n]] for n in C["n"]]
    reg = pd.read_parquet(EXT / "mapillary" / "mly_traffic_signs.parquet", columns=["value", "lon", "lat"])
    rx, ry = _TO_RD.transform(reg["lon"].to_numpy(), reg["lat"].to_numpy())
    keep_cnt = np.zeros(len(nodes))
    kr = reg["value"].str.contains("keep-right|keep-left|pass-on-either", regex=True).to_numpy()
    if kr.any():
        for lst in ntree.query_ball_point(np.c_[np.asarray(rx)[kr], np.asarray(ry)[kr]], r=40):
            for j in lst:
                keep_cnt[j] += 1
    F["keep_signs"] = [keep_cnt[nidx[n]] for n in C["n"]]
    return F


def given_keys(df):
    out = set()
    for s in df["link_ids"].dropna():
        try:
            out.add(tuple(str(x) for x in json.loads(s)))
        except Exception:
            pass
    return out


def consistent(sel, prob, C):
    """Flow rule for turns: a ban must never remove EVERY legal exit of an approach (traffic arriving on
    link i at junction n must be able to leave). Candidates are already only the legal exits under the
    predicted direction, so if all of an approach's candidates are selected, drop the least likely one.
    Returns (new selection, number of approaches fixed)."""
    sel = sel.copy()
    fixed = 0
    df = pd.DataFrame({"i": C["i"].to_numpy(), "n": C["n"].to_numpy(), "sel": sel, "p": prob})
    for _, g in df.groupby(["i", "n"]):
        if len(g) and g["sel"].all():
            sel[g["p"].idxmin()] = False
            fixed += 1
    return sel, fixed


def prf(sel, y):
    tp = int((sel & y).sum())
    p, r = tp / max(sel.sum(), 1), tp / max(y.sum(), 1)
    return tp, int(sel.sum()), p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", dest="evaluate", action="store_true", help="score the missing zone against the answer key")
    ap.add_argument("--team", default="TeamA")
    ap.add_argument("--dir-path", default=None)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--top", type=int, default=None, help="number of model rows to add (default: everything above --thr)")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--exit-rule", type=lambda v: v.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--bags", type=int, default=1, help="average this many seeds")
    ap.add_argument("--drop", default="to_is_partner,twin_cross,node_oneways,node_oneway_share", help="comma-separated feature names to drop (ablation)")
    ap.add_argument("--lgb", default=None, help='JSON overrides for LightGBM, e.g. {"num_leaves": 31}')
    ap.add_argument("--no-mly", action="store_true", help="ablation: drop every Mapillary feature")
    a = ap.parse_args()
    run(a)


def run(a):
    if a.lgb:
        LGB.update(json.loads(a.lgb))
    from paths import ANSWERS, CHALLENGE, LINKS, PRED, TURNS, read as rd
    nav_src, turnban_src = CHALLENGE / LINKS, CHALLENGE / TURNS
    dir_src = Path(a.dir_path) if a.dir_path else PRED / a.team / LINKS
    out_dir = PRED / a.team
    nav = rd(nav_src)
    net = Network(nav)
    dirs = rd(dir_src)["direction"].to_numpy(object)
    dirs = np.array([d if d in ("F", "T", "B") else "B" for d in dirs], dtype=object)
    known_link = nav["direction"].isin(["B", "F", "T"]).to_numpy()
    C = candidates(net, dirs)
    X = features(net, nav, dirs, C)
    MLY = [c for c in X.columns if c.startswith(("sign_", "arrow_")) or c in ("traffic_lights", "keep_signs")]
    if a.no_mly:
        X = X.drop(columns=MLY)
    if a.drop:
        X = X.drop(columns=[c for c in a.drop.split(",") if c in X])
    lid = nav["link_id"].to_numpy()
    pair = list(zip(lid[C["i"]], lid[C["k"]]))
    given = given_keys(rd(turnban_src))
    y_given = np.array([p in given for p in pair])
    in_known = known_link[C["i"]] & known_link[C["k"]]
    print(f"candidate turns {len(C)}, in known zone {in_known.sum()}, positives there {y_given[in_known].sum()} "
          f"(given 2-link rows {sum(len(g) == 2 for g in given)})")

    class Bag:
        def __init__(self, ms):
            self.ms = ms
            self.booster_ = ms[0].booster_
            self.feature_importances_ = ms[0].feature_importances_

        def predict_proba(self, Z):
            return np.mean([m.predict_proba(Z) for m in self.ms], axis=0)

    def fit(tr):
        return Bag([lgb.LGBMClassifier(**dict(LGB, random_state=s_)).fit(X[tr], y_given[tr]) for s_ in range(a.bags)])

    if a.cv or a.evaluate:
        # spatial 5-block CV inside the known zone (blocks by x of the via node)
        nx_ = np.array([node_xy(net, n)[0] for n in C["n"]])
        q = np.quantile(nx_[in_known], [0.2, 0.4, 0.6, 0.8])
        blk = np.digitize(nx_, q)
        oof = np.full(len(C), np.nan)
        for b in range(5):
            tr, te = in_known & (blk != b), in_known & (blk == b)
            if y_given[tr].sum() < 3 or te.sum() == 0:
                continue
            oof[te] = fit(tr).predict_proba(X[te])[:, 1]
        ok = ~np.isnan(oof)
        print("known-zone 5-block CV (tp / predicted / P / R / F1):")
        for t in (0.3, 0.5, 0.7, 0.8, 0.9):
            tp, n_, p, r, f = prf(ok & (oof >= t), y_given & ok)
            sc, nfix = consistent(ok & (oof >= t), np.nan_to_num(oof), C)
            tp2, n2, p2, r2, f2 = prf(sc, y_given & ok)
            print(f"  thr {t:.1f}: {tp} / {n_} / {p:.3f} / {r:.3f} / {f:.3f}   + exit rule: {tp2} / {n2} / {p2:.3f} / {r2:.3f} / {f2:.3f} (approaches fixed {nfix})")
        sb = ok & (X["sign_bans_this"].to_numpy() > 0) if "sign_bans_this" in X else ok & False
        tp, n_, p, r, f = prf(sb, y_given & ok)
        print(f"  sign rule (any facing sign bans this turn): {tp} / {n_} / {p:.3f} / {r:.3f} / {f:.3f}")
        imp = pd.Series(fit(in_known).booster_.feature_importance("gain"), X.columns)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        print("gain share, Mapillary features:", round(imp[[c for c in MLY if c in imp]].sum(), 3), " top:", imp.head(8).round(3).to_dict())
    m = fit(in_known)
    if a.evaluate:
        truth = given_keys(rd(ANSWERS / TURNS)) - given
        y_t = np.array([p in truth for p in pair])
        te = ~in_known
        pr = m.predict_proba(X[te])[:, 1]
        print(f"missing zone: answer-key 2-link rows reachable as candidates {y_t[te].sum()} of {sum(len(t) == 2 for t in truth)}")
        Cte = C[te].reset_index(drop=True)
        for t in (0.3, 0.5, 0.7, 0.8, 0.9):
            tp, n_, p, r, f = prf(pr >= t, y_t[te])
            sc, nfix = consistent(pr >= t, pr, Cte)
            tp2, n2, p2, r2, f2 = prf(sc, y_t[te])
            print(f"  thr {t:.1f}: {tp} / {n_} / P {p:.3f} R {r:.3f} F1 {f:.3f}   + exit rule: {tp2} / {n2} / F1 {f2:.3f} (approaches fixed {nfix})")
        imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False)
        print("top features:", imp.head(12).to_dict())
    if a.write:
        te = ~in_known
        pr = m.predict_proba(X[te])[:, 1]
        order = np.argsort(-pr)
        pick = order[: a.top] if a.top else order[pr[order] >= a.thr]
        if a.exit_rule:
            selm = np.zeros(len(pr), bool)
            selm[pick] = True
            selm, nfix = consistent(selm, pr, C[te].reset_index(drop=True))
            pick = np.nonzero(selm)[0]
            print(f"exit rule: approaches fixed {nfix}")
        base = rd(turnban_src)
        have = given_keys(base)
        geom_of = dict(zip(nav["link_id"], nav["geom"]))
        admin = base["city"].mode().iat[0] if "city" in base and base["city"].notna().any() else ""
        idx = np.nonzero(te)[0][pick]
        new = []
        for r in idx:
            seq = pair[r]
            if seq in have:
                continue
            have.add(seq)
            geom = "MULTILINESTRING (" + ",".join(f"({wkt_inner(geom_of[l])})" for l in seq) + ")"
            new.append({"link_ids": json.dumps(list(seq), separators=(",", ":")), "city": admin, "geom": geom})
        out = pd.concat([base, pd.DataFrame(new, columns=base.columns)], ignore_index=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / TURNS, index=False)
        print(f"added {len(new)} rows")
        print("done", out_dir / TURNS)


if __name__ == "__main__":
    main()
