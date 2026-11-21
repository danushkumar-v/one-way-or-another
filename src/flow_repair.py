"""Flow-consistent direction: pick the most likely B/F/T labelling under traffic-flow rules.

Rule: at every junction inside the map (not on the dataset edge) traffic that can enter must be able to
leave, so each junction needs at least one way IN and one way OUT. A one-way street therefore cannot end
in a dead end, and a junction cannot have every street pointing into it (a sink) or out of it (a source).
The per-link model (src/mly_graph.py) predicts links almost independently and breaks this rule; the
true network almost never does.

Method: integer linear program (scipy milp / HiGHS)
    x[i,c] in {0,1}, sum_c x[i,c] = 1                       for every link to predict, c in {B, F, T}
    minimise  sum -log p(i,c) * x[i,c]  +  lam * sum slack  (p from the model)
    in(n)  + s_in(n)  >= 1,  out(n) + s_out(n) >= 1         for every interior junction n
where in(n) / out(n) count the incident links whose label lets traffic enter / leave n (given labels
are constants). lam = inf makes the rule hard; a finite lam lets very confident evidence win.

    python src/flow_repair.py                 # final model + solver -> predictions/TeamA/
    python src/flow_repair.py --eval          # score checks A and C against the answer key, several lam
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction import ROOT, Network, angdiff  # noqa: E402
from mly_graph import ALIASES, CONFIGS, build_features, metrics, predict_cfg  # noqa: E402

CLS = np.array(["B", "F", "T"])
LETS_IN = {"nref": ("B", "F"), "ref": ("B", "T")}       # label lets traffic ENTER the node at that end
LETS_OUT = {"nref": ("B", "T"), "ref": ("B", "F")}


def node_xy(net, n):
    i = net.node_links[n][0]
    g = net.geoms[i]
    at_start = net.ref_first[i] if n == net.ref[i] else not net.ref_first[i]
    return g.coords[0] if at_start else g.coords[-1]


def interior_nodes(net, margin=150.0):
    x0, y0, x1, y1 = net.bbox
    out = []
    for n in net.node_links:
        x, y = node_xy(net, n)
        if x0 + margin < x < x1 - margin and y0 + margin < y < y1 - margin:
            out.append(n)
    return out


def violations(net, lab, nodes):
    bad = 0
    for n in nodes:
        i_ok = o_ok = False
        for i in net.node_links[n]:
            if net.ref[i] == net.nref[i]:
                i_ok = o_ok = True
                continue
            end = "nref" if n == net.nref[i] else "ref"
            i_ok |= lab[i] in LETS_IN[end]
            o_ok |= lab[i] in LETS_OUT[end]
        bad += (not i_ok) + (not o_ok)
    return bad


def pair_rules(net):
    """(i, j, same) pairs for the two extra rules.
    twin:  the two carriageways of a dual road (src/direction.ev_twin) must not run the same way.
    cont:  a street continuing through a node with exactly two links keeps its direction.
    same = True when i and j have the same ref->nref orientation in the world (so F_i ~ F_j)."""
    from direction import ev_twin, Params
    twin = set()
    for i, (j, _) in ev_twin(net, Params()).items():
        a, b = min(i, j), max(i, j)
        twin.add((a, b, angdiff(net.chord_f(a), net.chord_f(b)) < 90))
    cont = []
    for n, links in net.node_links.items():
        if len(links) != 2:
            continue
        i, j = links
        if i == j or net.ref[i] == net.nref[i] or net.ref[j] == net.nref[j]:
            continue
        if 180 - angdiff(net.outward(i, n), net.outward(j, n)) > 35:
            continue
        same = (n == net.nref[i]) == (n == net.ref[j])           # i flows into n, j flows out of n
        cont.append((i, j, same))
    return sorted(twin), cont


def solve(net, P, free, fixed_lab, nodes, lam=np.inf, eps=1e-4, twin=(), mu_twin=0.0, cont=(), mu_cont=0.0):
    """P: (n_free, 3) probs for links idx `free`; fixed_lab: labels for every link (used where not free).
    Rows are  lo <= sum coef*x + slack-terms <= hi ; every rule is soft with its own penalty."""
    pos = {i: k for k, i in enumerate(free)}
    nf = len(free)
    cost = list((-np.log(np.clip(P, eps, 1))).ravel())
    rows, lo, hi = [], [], []
    slack_cost = []

    def term(i, c):                     # -> (var index or None, constant 0/1)
        if i in pos:
            return 3 * pos[i] + "BFT".index(c), 0
        return None, int(fixed_lab[i] == c)

    def add(coef, const, l, h, pen, sign):
        """sum coef + const in [l, h]; sign=+1 slack relaxes a >= row, -1 relaxes a <= row."""
        if not coef:
            return
        if np.isfinite(pen):
            coef = dict(coef)
            coef[3 * nf + len(slack_cost)] = float(sign)
            slack_cost.append(pen)
        rows.append(coef)
        lo.append(l - const)
        hi.append(h - const)

    for n in nodes:                                                     # rule 1: in & out at every junction
        links = net.node_links[n]
        if not any(i in pos for i in links):
            continue
        for kind in ("in", "out"):
            const, coef = 0, {}
            for i in links:
                if net.ref[i] == net.nref[i]:
                    const = 1
                    break
                end = "nref" if n == net.nref[i] else "ref"
                for c in (LETS_IN if kind == "in" else LETS_OUT)[end]:
                    v, k = term(i, c)
                    if v is None:
                        const = max(const, k)
                    else:
                        coef[v] = 1.0
            if const >= 1:
                continue
            add(coef, 0, 1, np.inf, lam, +1)

    if mu_twin > 0:                                                     # rule 2: dual carriageways opposite
        for i, j, same in twin:
            if i not in pos and j not in pos:
                continue
            for ci, cj in ((("F", "F"), ("T", "T")) if same else (("F", "T"), ("T", "F"))):
                coef, const = {}, 0
                for v, k in (term(i, ci), term(j, cj)):
                    if v is None:
                        const += k
                    else:
                        coef[v] = 1.0
                add(coef, const, -np.inf, 1, mu_twin, -1)

    if mu_cont > 0:                                                     # rule 3: continuation keeps direction
        for i, j, same in cont:
            if i not in pos and j not in pos:
                continue
            for ci in ("F", "T"):
                cj = ci if same else ("T" if ci == "F" else "F")
                vi, ki = term(i, ci)
                vj, kj = term(j, cj)
                coef, const = {}, 0
                if vi is None:
                    const += ki
                else:
                    coef[vi] = 1.0
                if vj is None:
                    const -= kj
                else:
                    coef[vj] = coef.get(vj, 0) - 1.0
                add(coef, const, 0, np.inf, mu_cont, +1)      # x_i - x_j >= 0 (soft)
                add(coef, const, -np.inf, 0, mu_cont, -1)     # x_i - x_j <= 0 (soft)

    ns = len(slack_cost)
    nv = 3 * nf + ns
    cost = np.r_[cost, np.array(slack_cost, float)]
    r, c, v = [], [], []
    for k, row in enumerate(rows):
        for j, w in row.items():
            r.append(k), c.append(j), v.append(w)
    E = sparse.csr_matrix((np.ones(3 * nf), (np.repeat(np.arange(nf), 3), np.arange(3 * nf))), shape=(nf, nv))
    cons = [LinearConstraint(E, 1, 1)]
    if rows:
        cons.append(LinearConstraint(sparse.csr_matrix((v, (r, c)), shape=(len(rows), nv)), np.array(lo), np.array(hi)))
    ub = np.r_[np.ones(3 * nf), np.full(ns, 2.0)]
    res = milp(cost, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, ub), options={"time_limit": 300})
    if res.x is None:
        return CLS[P.argmax(1)], len(rows), "no solution"
    X = np.round(res.x[: 3 * nf]).reshape(nf, 3)
    return CLS[X.argmax(1)], len(rows), res.message


def rule_breaks(net, lab, twin, cont):
    tb = sum(1 for i, j, same in twin if lab[i] in "FT" and lab[j] in "FT" and ((lab[i] == lab[j]) == same))
    cb = sum(1 for i, j, same in cont if (lab[i] in "FT" or lab[j] in "FT") and
             (lab[j] != (lab[i] if same else {"F": "T", "T": "F", "B": "B"}[lab[i]])))
    return tb, cb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", dest="evaluate", action="store_true", help="score against the answer key instead of writing predictions")
    ap.add_argument("--team", default="TeamA")
    ap.add_argument("--config", default="FULL_MLY")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--checks", default="C")
    ap.add_argument("--mu-twin", type=float, default=1.0, help="penalty: dual carriageways running the same way")
    ap.add_argument("--mu-cont", type=float, default=1.0, help="penalty: street changes direction at a 2-link node")
    ap.add_argument("--grid", default="base", help="--eval: which experiment grid to run (base | rules)")
    a = ap.parse_args()
    run(a)


def run(a):
    from paths import ANSWERS, CHALLENGE, LINKS, PRED, PROC, read as rd
    proc = PROC
    nav = rd(CHALLENGE / LINKS)
    net = Network(nav)
    log = print
    A, pairs, graph, extra_feats, mats = build_features(net, nav, proc, log)
    cfg = CONFIGS[ALIASES.get(a.config, a.config)]
    given = nav["direction"].where(nav["direction"].isin(["B", "F", "T"])).to_numpy(object)
    known = pd.notna(given)
    nodes = interior_nodes(net)

    if not a.evaluate:
        P = predict_cfg(cfg, A, pairs, graph, extra_feats, mats, given, known, ~known, return_proba=True)
        free = np.nonzero(~known)[0]
        twin, cont = pair_rules(net)
        lab, _, _ = solve(net, P, free, given, nodes, lam=a.lam, twin=twin, mu_twin=a.mu_twin, cont=cont, mu_cont=a.mu_cont)
        out = nav.copy()
        out.loc[~known, "direction"] = lab
        dest = PRED / a.team
        dest.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest / LINKS, index=False)
        print("done", dest / LINKS)
        return

    truth = rd(ANSWERS / LINKS).set_index("link_id")["direction"]
    y = nav["link_id"].map(truth).to_numpy(object)
    print(f"interior junctions {len(nodes)}; rule violations in the TRUE labels: {violations(net, y, nodes)}")
    sp = pd.read_parquet(proc / "split.parquet").set_index("link_id").reindex(nav["link_id"])
    hold = sp["holdout_A"].fillna(False).to_numpy(bool)
    checks = {"A": (known & ~hold, hold), "C": (known, ~known)}
    for chk in a.checks.split(","):
        tr, te = checks[chk]
        P = predict_cfg(cfg, A, pairs, graph, extra_feats, mats, y, tr, te, return_proba=True)
        free = np.nonzero(te)[0]
        fixed = np.where(tr | ~te, y, "B").astype(object)        # training labels known; nothing else is free
        base = CLS[np.where(P[:, 0] >= 0.5, 0, np.where(P[:, 1] >= P[:, 2], 1, 2))]
        lab0 = fixed.copy()
        lab0[free] = base
        m = metrics(base, y[te])
        print(f"check {chk}  model only          acc {m['accuracy']:.4f}  oneway P/R {m['oneway_P']:.3f}/{m['oneway_R']:.3f}  F-vs-T {m['F_vs_T']:.3f}  violations {violations(net, lab0, nodes)}")
        twin, cont = pair_rules(net)
        if chk == "C":
            print(f"rule pairs: dual carriageways {len(twin)}, 2-link continuations {len(cont)}; "
                  f"breaks in TRUE labels: twin {rule_breaks(net, y, twin, cont)[0]}, cont {rule_breaks(net, y, twin, cont)[1]}")
        tb, cb = rule_breaks(net, lab0, twin, cont)
        print(f"check {chk}  model only rule breaks: twin {tb}, cont {cb}")
        if a.grid == "base":
            grid = [dict(lam=l) for l in (1.0, 2.0, 4.0, 8.0, np.inf)]
        else:
            grid = ([dict(lam=4.0)] + [dict(lam=4.0, mu_twin=m) for m in (1.0, 2.0, 4.0, np.inf)]
                    + [dict(lam=4.0, mu_cont=m) for m in (0.5, 1.0, 2.0, 4.0)]
                    + [dict(lam=4.0, mu_twin=mt, mu_cont=mc) for mt in (2.0, 4.0) for mc in (0.5, 1.0)])
        for g in grid:
            t = time.perf_counter()
            lab, ncons, msg = solve(net, P, free, fixed, nodes, twin=twin, cont=cont, **g)
            l2 = fixed.copy()
            l2[free] = lab
            m = metrics(lab, y[te])
            tb, cb = rule_breaks(net, l2, twin, cont)
            tag = " ".join(f"{k}={v:g}" for k, v in g.items())
            print(f"check {chk}  flow {tag:28s} acc {m['accuracy']:.4f}  oneway P/R {m['oneway_P']:.3f}/{m['oneway_R']:.3f}  F-vs-T {m['F_vs_T']:.3f}  "
                  f"junction breaks {violations(net, l2, nodes)}  twin {tb}  cont {cb}  changed {(lab != base).sum()}  {time.perf_counter() - t:.1f}s", flush=True)

if __name__ == "__main__":
    main()
