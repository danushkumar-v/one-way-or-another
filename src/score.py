"""Score a prediction folder against the answer key in data/simulated/answers.

usage:
  python src/score.py <prediction_folder> [--truth data/simulated/answers] [--challenge data/simulated/challenge]

The prediction folder holds files in the challenge schema (any name containing links / truck limits / turn bans, .csv):
  links: link_id, direction               -> scored on links whose direction is blank in the challenge
  truck limits: link_id, limit_type, limit_value
  turn bans:       link_ids  (JSON list, from-link first)
For truck limits/turn bans, rows already given in the challenge are ignored on both sides: only NEW rows count.
A missing file = that task is skipped. Writes <prediction_folder>/score.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def find(folder, key):
    m = sorted(Path(folder).glob(f"*{key}*.csv"))
    return pd.read_csv(m[0], dtype=str, keep_default_na=False, na_values=[""]) if m else None


def prf(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r), 4) if p + r else 0.0,
                tp=int(tp), predicted=int(n_pred), truth=int(n_true))


def score_dir(pred, truth, chal):
    target = chal.loc[chal.direction.isna(), "link_id"]
    t = truth.set_index("link_id").direction.reindex(target)
    p = pred.drop_duplicates("link_id").set_index("link_id").direction.str.strip().str.upper().reindex(target)
    p = p.where(p.isin(["B", "F", "T"]))
    out = dict(links_to_predict=int(len(t)), answered=int(p.notna().sum()), accuracy=round(float((p == t).mean()), 4),
               baseline_all_B=round(float((t == "B").mean()), 4))
    f1s = []
    for c in ("B", "F", "T"):
        out[f"class_{c}"] = prf(((p == c) & (t == c)).sum(), (p == c).sum(), (t == c).sum())
        f1s.append(out[f"class_{c}"]["f1"])
    out["macro_f1"] = round(sum(f1s) / 3, 4)
    po, to = p.isin(["F", "T"]), t.isin(["F", "T"])
    out["oneway_detection"] = prf((po & to).sum(), po.sum(), to.sum())          # is it one-way at all?
    both = po & to
    out["direction_given_oneway_found"] = dict(n=int(both.sum()), accuracy=round(float((p[both] == t[both]).mean()), 4) if both.any() else None)
    return out


def cond_keys(df):
    return set(zip(df.link_id, df.limit_type.str.strip().str.upper()))


def score_cond(pred, truth, chal):
    given = cond_keys(chal)
    T, P = cond_keys(truth) - given, cond_keys(pred) - given
    out = dict(all=prf(len(T & P), len(P), len(T)))
    for typ in sorted({k[1] for k in T | P}):
        Tt, Pt = {k for k in T if k[1] == typ}, {k for k in P if k[1] == typ}
        out[typ] = prf(len(Tt & Pt), len(Pt), len(Tt))
    tv = {(a, b.strip().upper()): float(v) for a, b, v in zip(truth.link_id, truth.limit_type, truth.limit_value)}
    pv = {(a, b.strip().upper()): pd.to_numeric(v, errors="coerce") for a, b, v in zip(pred.link_id, pred.limit_type, pred.limit_value)}
    hit = [k for k in T & P if k[1] != "NO TRUCKS"]
    if hit:
        out["value_exact_on_matches"] = round(sum(pv[k] == tv[k] for k in hit) / len(hit), 4)
        out["value_within_10pct_on_matches"] = round(sum(abs(pv[k] - tv[k]) <= 0.1 * tv[k] for k in hit) / len(hit), 4)
    out["links_any_type"] = prf(len({k[0] for k in T} & {k[0] for k in P}), len({k[0] for k in P}), len({k[0] for k in T}))
    return out


def turnban_keys(df):
    def parse(s):
        try:
            return tuple(str(x) for x in json.loads(s))
        except Exception:
            return tuple(x.strip(' "\'') for x in str(s).strip("[]").split(",") if x.strip())
    return {parse(s) for s in df.link_ids.dropna()}


def score_turns(pred, truth, chal):
    given = turnban_keys(chal)
    T, P = turnban_keys(truth) - given, turnban_keys(pred) - given
    ft = lambda S: {(k[0], k[-1], len(k) == 1) for k in S}   # noqa: E731
    fr = lambda S: {k[0] for k in S}                          # noqa: E731
    return dict(exact_sequence=prf(len(T & P), len(P), len(T)),
                from_and_last_link=prf(len(ft(T) & ft(P)), len(ft(P)), len(ft(T))),
                from_link_only=prf(len(fr(T) & fr(P)), len(fr(P)), len(fr(T))))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    if not args:
        print(__doc__)
        sys.exit(1)
    pred_dir = Path(args[0])
    truth_dir = Path(opt.get("--truth", ROOT / "data" / "simulated" / "answers"))
    chal_dir = Path(opt.get("--challenge", ROOT / "data" / "simulated" / "challenge"))
    res = {}
    for name, key, fn in (("direction", "links", score_dir), ("truck_limits", "truck_limits", score_cond), ("turn_bans", "turn_bans", score_turns)):
        p = find(pred_dir, key)
        if p is None:
            res[name] = "skipped (no file)"
            continue
        res[name] = fn(p, find(truth_dir, key), find(chal_dir, key))
    print(json.dumps(res, indent=2))
    (pred_dir / "score.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
