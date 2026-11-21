"""Choose direction settings inside the known zone only (the answer key is never read).

5 spatial blocks of the known zone (by x of the link midpoint). For each block: train on the other four,
predict the block AND the whole blank zone (both unknown, as in the real task), solve the flow rules over
all of them, score on the block against the given labels.

Settings searched
  model   : FULL_MLY (current) / bigger LightGBM / 3 ICA rounds
  w       : one-way prior; P(B) is multiplied by w (w < 1 = call one-way more readily)
  lam, mu_twin, mu_cont : flow-rule penalties (src/flow_repair.py)

    python src/tune.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mly_graph as mg  # noqa: E402
from direction import ROOT, Network  # noqa: E402
from flow_repair import CLS, interior_nodes, pair_rules, solve  # noqa: E402


def shift(P, w):
    Q = P.copy()
    Q[:, 0] *= w
    return Q / Q.sum(1, keepdims=True)


def score(lab, y):
    m = mg.metrics(lab, y)
    return m["accuracy"], m["oneway_P"], m["oneway_R"], m["F_vs_T"]


def main():
    from paths import CHALLENGE, LINKS, PROC, RESULTS, read
    nav = read(CHALLENGE / LINKS)
    net = Network(nav)
    A, pairs, graph, _, mats = mg.build_features(net, nav, PROC, lambda *_: None)
    y = nav["direction"].where(nav["direction"].isin(["B", "F", "T"])).to_numpy(object)
    known = pd.notna(y)
    mx = np.array([g.interpolate(0.5, normalized=True).x for g in net.geoms])
    q = np.quantile(mx[known], [0.2, 0.4, 0.6, 0.8])
    blk = np.where(known, np.digitize(mx, q), -1)
    nodes = interior_nodes(net)
    twin, cont = pair_rules(net)
    base_lgb = dict(mg.LGB)
    models = {
        "FULL_MLY": (dict(base_lgb), mg.CONFIGS[mg.ALIASES["FULL_MLY"]]),
        "FULL_MLY big": (dict(base_lgb, n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20),
                         mg.CONFIGS[mg.ALIASES["FULL_MLY"]]),
        "FULL_MLY ica3": (dict(base_lgb), dict(mg.CONFIGS[mg.ALIASES["FULL_MLY"]], ica=3)),
    }
    print(f"known links {known.sum()}, blocks {np.bincount(blk[known]).tolist()}")
    rows = []
    for mname, (lgbp, cfg) in models.items():
        mg.LGB.clear()
        mg.LGB.update(lgbp)
        t0 = time.perf_counter()
        Ps = []
        for b in range(5):
            tr = known & (blk != b)
            te = (blk == b) | ~known
            Ps.append((tr, te, mg.predict_cfg(cfg, A, pairs, graph, None, mats, y, tr, te, return_proba=True)))
        print(f"[{mname}] 5 block fits in {time.perf_counter() - t0:.0f}s", flush=True)
        grid = [dict(w=w) for w in (0.6, 0.7, 0.8, 0.9, 1.0, 1.2)]
        grid += [dict(w=w, lam=lam, mu_twin=mt, mu_cont=mc) for w in (0.7, 0.8, 0.9, 1.0) for lam in (2.0, 4.0, 8.0)
                 for mt, mc in ((0, 0), (2, 0.5), (1, 1), (4, 2))]
        for g in grid:
            lab_all, y_all = [], []
            for tr, te, P in Ps:
                Q = shift(P, g["w"])
                free = np.nonzero(te)[0]
                if "lam" in g:
                    fixed = np.where(tr, y, "B").astype(object)
                    lab, _, _ = solve(net, Q, free, fixed, nodes, lam=g["lam"], twin=twin, mu_twin=g["mu_twin"],
                                      cont=cont, mu_cont=g["mu_cont"])
                else:
                    lab = CLS[np.where(Q[:, 0] >= 0.5, 0, np.where(Q[:, 1] >= Q[:, 2], 1, 2))]
                inb = known[free]                       # score only the held-out known block
                lab_all.append(lab[inb])
                y_all.append(y[free][inb])
            acc, p, r, ft = score(np.concatenate(lab_all), np.concatenate(y_all))
            rows.append(dict(model=mname, **{k: g.get(k, "-") for k in ("w", "lam", "mu_twin", "mu_cont")},
                             acc=round(acc, 4), oneway_P=round(p, 3), oneway_R=round(r, 3), F_vs_T=round(ft, 3)))
        best = max((r for r in rows if r["model"] == mname), key=lambda r: r["acc"])
        print(f"[{mname}] best: {best}", flush=True)
    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "tuning_known_zone.csv"
    df.to_csv(out, index=False)
    print(df.head(15).to_string(index=False))
    ref = df[(df.model == "FULL_MLY") & (df.w == 1.0) & (df.lam == 4.0) & (df.mu_twin == 2) & (df.mu_cont == 0.5)]
    print("current settings:", ref.to_dict("records"))
    print(out)


if __name__ == "__main__":
    main()
