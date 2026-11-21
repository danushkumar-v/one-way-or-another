"""direction with Mapillary-first evidence and graph models, scored on the same spatial checks
on hidden areas: A border strip, B five blocks, C known zone -> whole blank zone.

    python src/mly_graph.py --eval                    # leaderboard of every configuration vs the answer key
    python src/mly_graph.py --predict FULL_MLY        # write predictions/<team>/ with one configuration

Needs: python src/prepare.py && python src/split.py

Evidence per link (oriented ones are pairs F/T relative to ref -> nref)
  sign   Mapillary no-entry / one-way signs matched to the link (src/direction.py)
  seq    distinct Mapillary sequences that DROVE along the link in each direction
         (motion heading from consecutive images; speed >= 6 m/s so cyclists/pedestrians drop out)
  twin   dual-carriageway geometry vote (right-hand traffic)
Graph
  continuation neighbours = links at the same node that continue the street (same name, or a degree-2
  node) within 35 deg of straight. Their oriented evidence is re-expressed in this link's frame and summed
  (hop 1, hop 2) = a simple graph convolution. ICA = collective classification: neighbours' predicted
  one-way probability and direction become features, repeated for a few rounds (OOF on training links).
Models
  3class   LightGBM B/F/T on all features in one model
  2stage   LightGBM one-way? on orientation-free features, then LightGBM F vs T on oriented
           differences, trained with mirrored copies (F<->T swap) so direction is symmetric.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction import EXT, ROOT, _TO_RD, Network, Params, angdiff, ev_mly, ev_twin  # noqa: E402
from mly_side import side_pairs  # noqa: E402

CLS = np.array(["B", "F", "T"])
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


# ======================================================================= evidence
def seq_traversal(net: Network, min_speed=6.0, max_dist=10.0):
    im = pd.read_parquet(EXT / "mapillary" / "mly_images.parquet", columns=["sequence_id", "captured_at", "lon", "lat"])
    x, y = _TO_RD.transform(im["lon"].to_numpy(), im["lat"].to_numpy())
    x0, y0, x1, y1 = net.bbox
    m = (x > x0 - 50) & (x < x1 + 50) & (y > y0 - 50) & (y < y1 + 50)
    im = im.loc[m, ["sequence_id", "captured_at"]].assign(x=np.asarray(x)[m], y=np.asarray(y)[m])
    t = pd.to_datetime(im["captured_at"], errors="coerce", utc=True)
    im["t"] = (t - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds()
    im = im.dropna(subset=["t"]).sort_values(["sequence_id", "t"]).reset_index(drop=True)
    seq = im["sequence_id"].to_numpy()
    X, Y, Tt = im["x"].to_numpy(), im["y"].to_numpy(), im["t"].to_numpy()
    n = len(im)
    ok = np.r_[False, seq[1:] == seq[:-1]] & np.r_[seq[:-1] == seq[1:], False]
    k = np.arange(n)
    pi, ni = np.clip(k - 1, 0, None), np.clip(k + 1, None, n - 1)
    dx, dy = X[ni] - X[pi], Y[ni] - Y[pi]
    d, dt = np.hypot(dx, dy), Tt[ni] - Tt[pi]
    speed = np.where(dt > 0, d / np.maximum(dt, 1e-6), 0)
    ok &= (d >= 2) & (d <= 80) & (speed >= min_speed) & (speed <= 45)
    head = np.degrees(np.arctan2(dx, dy)) % 360
    sel = np.nonzero(ok)[0]

    segs, seg_link, seg_brg = [], [], []
    for i, g in enumerate(net.geoms):
        c = np.asarray(g.coords)[:, :2]
        if len(c) < 2:
            continue
        L = np.hypot(*(c[1:] - c[:-1]).T)
        keep = L > 0.1
        if not keep.any():
            continue
        a, b = c[:-1][keep], c[1:][keep]
        brg = np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) % 360
        if not net.ref_first[i]:
            brg = (brg + 180) % 360
        segs.append(shapely.linestrings(np.stack([a, b], axis=1)))
        seg_link.append(np.full(len(a), i))
        seg_brg.append(brg)
    segs, seg_link, seg_brg = np.concatenate(segs), np.concatenate(seg_link), np.concatenate(seg_brg)
    tree = STRtree(segs)
    inp, sg = tree.query_nearest(shapely.points(X[sel], Y[sel]), max_distance=max_dist, all_matches=False)
    diff = np.abs((head[sel][inp] - seg_brg[sg] + 180) % 360 - 180)
    dirn = np.where(diff <= 30, 0, np.where(diff >= 150, 1, -1))
    df = pd.DataFrame({"link": seg_link[sg], "seq": seq[sel][inp], "d": dirn})
    df = df[df["d"] >= 0].drop_duplicates()
    cnt = df.groupby(["link", "d"]).size().unstack(fill_value=0).reindex(range(len(net)), fill_value=0)
    cnt = cnt.reindex(columns=[0, 1], fill_value=0)
    return cnt[0].to_numpy(float), cnt[1].to_numpy(float)


def continuation_mats(net: Network, max_turn=35.0):
    """S: same-orientation continuation, Xm: crossed orientation, A: any neighbour (row-normalised)."""
    n = len(net)
    rs, cs, rx, cx, ra, ca = [], [], [], [], [], []
    for node, links in net.node_links.items():
        if len(links) < 2:
            continue
        outs = {i: net.outward(i, node) for i in links}
        for i in links:
            if net.ref[i] == net.nref[i]:
                continue
            for j in links:
                if j == i or net.ref[j] == net.nref[j]:
                    continue
                ra.append(i), ca.append(j)
                named = net.name[i] is not None and net.name[i] == net.name[j]
                if not (named or len(links) == 2):
                    continue
                if 180 - angdiff(outs[i], outs[j]) > max_turn:
                    continue
                same = (node == net.nref[i]) == (node == net.ref[j])
                (rs if same else rx).append(i)
                (cs if same else cx).append(j)
    S = sparse.csr_matrix((np.ones(len(rs)), (rs, cs)), shape=(n, n))
    Xm = sparse.csr_matrix((np.ones(len(rx)), (rx, cx)), shape=(n, n))
    A = sparse.csr_matrix((np.ones(len(ra)), (ra, ca)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    A = sparse.diags(1 / np.maximum(deg, 1)) @ A
    return S, Xm, A


def agg(S, Xm, f, t):
    return S @ f + Xm @ t, S @ t + Xm @ f


def build_features(net: Network, nav: pd.DataFrame, proc: Path, log):
    """Link attributes + Mapillary evidence + graph aggregates. Nothing else is read."""
    n = len(net)
    yes = lambda s: s.astype(str).str.strip().str.upper().isin(["Y", "YES", "1", "TRUE"]).to_numpy(int)   # noqa: E731
    deg = np.array([net.degree(r) for r in net.ref]), np.array([net.degree(r) for r in net.nref])
    length = np.array([g.length for g in net.geoms])
    chord = np.array([np.hypot(g.coords[-1][0] - g.coords[0][0], g.coords[-1][1] - g.coords[0][1]) for g in net.geoms])
    A = pd.DataFrame({
        "road_class": pd.to_numeric(nav["road_class"], errors="coerce").to_numpy(),
        "divider": nav["divider"].fillna("?").astype("category").cat.codes.to_numpy(),
        "bridge": yes(nav["bridge"]), "tunnel": yes(nav["tunnel"]), "priority": yes(nav["priority_road"]),
        "length": length, "sinuosity": np.clip(length / np.where(chord > 0.5, chord, np.nan), None, 5),
        "has_name": (net.name != None).astype(int),   # noqa: E711
        "deg_ref": deg[0], "deg_nref": deg[1], "loop": (net.ref == net.nref).astype(int),
    })
    pairs = {}
    t0 = time.perf_counter()
    E = ev_mly(net, Params())
    pairs["sign"] = (E[:, 0], E[:, 1])
    t1 = time.perf_counter()
    sF, sT = seq_traversal(net)
    pairs["seq"] = (sF, sT)
    t2 = time.perf_counter()
    pairs.update(side_pairs(net))
    t3 = time.perf_counter()
    tw = ev_twin(net, Params())
    tF, tT = np.zeros(n), np.zeros(n)
    for i, (_, d) in tw.items():
        (tF if d == 0 else tT)[i] = 1.0
    pairs["twin"] = (tF, tT)
    t4 = time.perf_counter()
    log(f"timing (s): signs {t1 - t0:.1f}  drive-through {t2 - t1:.1f}  side {t3 - t2:.1f}  twin {t4 - t3:.1f}")
    for k in ("side_sign", "side_mark", "stopline"):
        log(f"evidence coverage {k}: {((pairs[k][0] + pairs[k][1]) > 0).mean():.3f}")
    log(f"evidence coverage: sign {(E[:, :2].sum(1) > 0).mean():.3f}  seq {((sF + sT) > 0).mean():.3f}  twin {(tF + tT).mean():.3f}")
    extra_feats = None
    t5 = time.perf_counter()
    S, Xm, Aadj = continuation_mats(net)
    log(f"timing (s): continuation graph {time.perf_counter() - t5:.1f}")
    graph = {}
    for k, (f, t) in pairs.items():
        f1, t1 = agg(S, Xm, f, t)
        f2, t2 = agg(S, Xm, f1, t1)
        graph[f"g1_{k}"], graph[f"g2_{k}"] = (f1, t1), (f2, t2)
    log(f"graph: continuation edges same {S.nnz}, crossed {Xm.nnz}")
    return A, pairs, graph, extra_feats, (S, Xm, Aadj)


# ======================================================================= models
def stage_frames(A, pairs, extra=None, use_attrs=True):
    """orientation-free frame (stage 1) and oriented-difference frame (stage 2)."""
    f1 = A.copy() if use_attrs else pd.DataFrame(index=A.index)
    f2 = pd.DataFrame(index=A.index)
    for k, (f, t) in pairs.items():
        f1[f"{k}_sum"], f1[f"{k}_absdiff"] = f + t, np.abs(f - t)
        f2[f"{k}_diff"], f2[f"{k}_sum"] = f - t, f + t
    if extra is not None:
        for c in extra.columns:
            f1[c] = extra[c].to_numpy()
    return f1, f2


def fit_2stage(F1, F2, y, tr, pred_rows):
    """returns p_oneway, p_F for pred_rows."""
    ow = np.isin(y, ["F", "T"]).astype(int)
    m1 = lgb.LGBMClassifier(**LGB).fit(F1[tr], ow[tr])
    p_ow = m1.predict_proba(F1[pred_rows])[:, 1]
    trd = tr & (ow == 1)
    if F2.shape[1] == 0:                                   # no oriented evidence: direction is a coin flip
        return p_ow, np.full(pred_rows.sum(), 0.5)
    X2 = np.vstack([F2[trd], _mirror(F2[trd])])          # mirrored copy: F<->T, labels swapped below
    y2 = np.r_[(y[trd] == "F").astype(int), (y[trd] == "T").astype(int)]
    m2 = lgb.LGBMClassifier(**LGB).fit(X2, y2)
    pF = m2.predict_proba(F2[pred_rows])[:, 1]
    return p_ow, pF


def _mirror(F2):
    M = F2.copy()
    M[:, 0::2] = -M[:, 0::2]           # columns alternate diff, sum -> negate diffs
    return M


def fit_3class(Fall, y, tr, pred_rows):
    yt = pd.Categorical(y[tr], categories=CLS).codes
    m = lgb.LGBMClassifier(**LGB).fit(Fall[tr], yt)
    P = np.zeros((pred_rows.sum(), 3))
    P[:, m.classes_] = m.predict_proba(Fall[pred_rows])
    return P


def predict_cfg(cfg, A, pairs, graph, extra_feats, mats, y, tr, te, return_proba=False):
    """-> labels for te rows."""
    use = {k: v for k, v in pairs.items() if k in cfg["src"]}
    if cfg["graph"]:
        use.update({k: v for k, v in graph.items() if k.split("_", 1)[1] in cfg["src"]})
    Ause = A.drop(columns=["divider"]) if cfg.get("no_divider") else A
    extra = extra_feats
    if cfg["model"] == "3class":
        F1, F2 = stage_frames(Ause, use, extra)
        Fall = pd.concat([F1, F2.add_prefix("o_")], axis=1).to_numpy(float)
        P = fit_3class(Fall, y, tr, te)
        return CLS[P.argmax(1)]
    S, Xm, Aadj = mats
    rounds = cfg.get("ica", 0)
    p_ow_all = pF_all = None
    for r in range(rounds + 1):
        cur = dict(use)
        if r > 0:
            w = p_ow_all
            cur["ica"] = agg(S, Xm, w * pF_all, w * (1 - pF_all))
            cur["ica_any"] = (Aadj @ w, Aadj @ w)
        F1, F2 = stage_frames(Ause, cur, extra)
        F1, F2 = F1.to_numpy(float), F2.to_numpy(float)
        if r < rounds:          # predictions for every link: OOF on training rows, full model elsewhere
            p_ow_all, pF_all = np.zeros(len(y)), np.full(len(y), 0.5)
            idx = np.nonzero(tr)[0]
            rng = np.random.default_rng(r)
            fold = rng.integers(0, 5, len(idx))
            for f in range(5):
                trf = np.zeros(len(y), bool)
                trf[idx[fold != f]] = True
                prf = np.zeros(len(y), bool)
                prf[idx[fold == f]] = True
                a, b = fit_2stage(F1, F2, y, trf, prf)
                p_ow_all[prf], pF_all[prf] = a, b
            a, b = fit_2stage(F1, F2, y, tr, ~tr)
            p_ow_all[~tr], pF_all[~tr] = a, b
        else:
            p_ow, pF = fit_2stage(F1, F2, y, tr, te)
    if return_proba:                                     # columns B, F, T
        return np.c_[1 - p_ow, p_ow * pF, p_ow * (1 - pF)]
    return np.where(p_ow >= cfg.get("thr", 0.5), np.where(pF >= 0.5, "F", "T"), "B")


def metrics(p, t):
    ow_p, ow_t = np.isin(p, ["F", "T"]), np.isin(t, ["F", "T"])
    tp = (ow_p & ow_t).sum()
    pr, rc = tp / max(ow_p.sum(), 1), tp / max(ow_t.sum(), 1)
    both = ow_p & ow_t
    return dict(accuracy=(p == t).mean(), oneway_P=pr, oneway_R=rc, oneway_F1=2 * pr * rc / max(pr + rc, 1e-9),
                F_vs_T=(p[both] == t[both]).mean() if both.any() else np.nan, n=len(t))


CONFIGS = {
    "ATTR_3class": dict(src=[], graph=False, model="3class"),
    "ATTR_2stage": dict(src=[], graph=False, model="2stage"),
    "MLY_local_2stage (sign+seq+twin)": dict(src=["sign", "seq", "twin"], graph=False, model="2stage"),
    "MLY_graph_2stage (+hop1/2)": dict(src=["sign", "seq", "twin"], graph=True, model="2stage"),
    "FULL_MLY (graph + ICA x2)": dict(src=["sign", "seq", "twin"], graph=True, model="2stage", ica=2),
    "FULL_MLY no divider": dict(src=["sign", "seq", "twin"], graph=True, model="2stage", ica=2, no_divider=True),
    "FULL_MLY thr 0.40": dict(src=["sign", "seq", "twin"], graph=True, model="2stage", ica=2, thr=0.40),
    "FULL_MLY thr 0.45": dict(src=["sign", "seq", "twin"], graph=True, model="2stage", ica=2, thr=0.45),
    "FULL_MLY_SIDE (+sign/marking side)": dict(src=["sign", "seq", "twin", "side_sign", "side_mark", "stopline"], graph=True, model="2stage", ica=2),
    "FULL_MLY_SIDE thr 0.45": dict(src=["sign", "seq", "twin", "side_sign", "side_mark", "stopline"], graph=True, model="2stage", ica=2, thr=0.45),
    "FULL_MLY_SIDE thr 0.40": dict(src=["sign", "seq", "twin", "side_sign", "side_mark", "stopline"], graph=True, model="2stage", ica=2, thr=0.40),
}
ALIASES = {"FULL_MLY": "FULL_MLY (graph + ICA x2)",
           "MLY_graph": "MLY_graph_2stage (+hop1/2)",
           "FULL_MLY_SIDE": "FULL_MLY_SIDE (+sign/marking side)"}


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", dest="evaluate", action="store_true", help="score every configuration against data/simulated/answers")
    ap.add_argument("--team", default="TeamA")
    ap.add_argument("--predict", default=None, help="config name or alias to write a prediction with")
    ap.add_argument("--only", default=None, help="comma-separated config names/aliases for the leaderboard")
    a = ap.parse_args()
    if not (a.evaluate or a.predict):
        sys.exit("use --eval and/or --predict CONFIG")
    run(a, print)


def run(a, log):
    from paths import ANSWERS, CHALLENGE, LINKS, PRED, PROC, RESULTS, read
    nav = read(CHALLENGE / LINKS)
    net = Network(nav)
    A, pairs, graph, extra_feats, mats = build_features(net, nav, PROC, log)
    given = nav["direction"].where(nav["direction"].isin(["B", "F", "T"])).to_numpy(object)
    known = pd.notna(given)

    if a.evaluate:
        truth = read(ANSWERS / LINKS).set_index("link_id")["direction"]
        y = nav["link_id"].map(truth).to_numpy(object)
        sp = pd.read_parquet(PROC / "split.parquet").set_index("link_id").reindex(nav["link_id"])
        hold, block = sp["holdout_A"].fillna(False).to_numpy(bool), sp["block"].fillna(-1).to_numpy(int)
        checks = {"A border strip": [(known & ~hold, hold)],
                  "B 5 blocks": [(known & (block != b), block == b) for b in range(5)],
                  "C known->missing": [(known, ~known)]}
        names = list(CONFIGS)
        if a.only:
            names = [ALIASES.get(s.strip(), s.strip()) for s in a.only.split(",")]
        rows = []
        for name in names:
            for chk, folds in checks.items():
                P, Tt = [], []
                for tr, te in folds:
                    P.append(predict_cfg(CONFIGS[name], A, pairs, graph, extra_feats, mats, y, tr, te))
                    Tt.append(y[te])
                m = metrics(np.concatenate(P), np.concatenate(Tt))
                rows.append(dict(config=name, check=chk, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
                print(f"{name:38s} {chk:18s} acc {m['accuracy']:.3f}  oneway P/R/F1 {m['oneway_P']:.3f}/{m['oneway_R']:.3f}/{m['oneway_F1']:.3f}  F-vs-T {m['F_vs_T']:.3f}", flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(RESULTS / "leaderboard_models.csv", index=False)
        print(RESULTS / "leaderboard_models.csv")

    if a.predict:
        name = ALIASES.get(a.predict, a.predict)
        lab = predict_cfg(CONFIGS[name], A, pairs, graph, extra_feats, mats, given, known, ~known)
        outnav = nav.copy()
        outnav.loc[~known, "direction"] = lab
        dest = PRED / a.team
        dest.mkdir(parents=True, exist_ok=True)
        outnav.to_csv(dest / LINKS, index=False)
        print("done", dest / LINKS)


if __name__ == "__main__":
    main()
