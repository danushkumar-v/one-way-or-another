"""Step-by-step leaderboard for the README and the deck  ->  results/story.json

Every number comes from the simulated city in data/simulated and can be reproduced with this script.
    python src/experiments.py        (about 10 minutes; needs prepare.py, split.py and run_all.py first)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mly_graph as mg  # noqa: E402
from direction import Network  # noqa: E402
from paths import ANSWERS, CHALLENGE, LINKS, PROC, RESULTS, read  # noqa: E402

STEPS = [("Link attributes only (road class, length, shape)", "ATTR_2stage"),
         ("+ Mapillary signs, drive-through sequences, twin carriageways", "MLY_local_2stage (sign+seq+twin)"),
         ("+ graph features, 1 and 2 hops along the street", "MLY_graph_2stage (+hop1/2)"),
         ("+ collective classification, 2 rounds", "FULL_MLY (graph + ICA x2)")]


def main():
    nav = read(CHALLENGE / LINKS)
    net = Network(nav)
    A, pairs, graph, extra, mats = mg.build_features(net, nav, PROC, lambda *_: None)
    given = nav["direction"].where(nav["direction"].isin(["B", "F", "T"])).to_numpy(object)
    known = pd.notna(given)
    y = nav["link_id"].map(read(ANSWERS / LINKS).set_index("link_id")["direction"]).to_numpy(object)
    sp = pd.read_parquet(PROC / "split.parquet").set_index("link_id").reindex(nav["link_id"])
    block = sp["block"].fillna(-1).to_numpy(int)

    def acc(cfg, folds):
        P, T = [], []
        for tr, te in folds:
            P.append(mg.predict_cfg(cfg, A, pairs, graph, extra, mats, y, tr, te))
            T.append(y[te])
        return float((np.concatenate(P) == np.concatenate(T)).mean())

    blocks = [(known & (block != b), block == b) for b in range(5)]
    exam = [(known, ~known)]
    rng = np.random.default_rng(0)
    r = rng.random(len(nav))
    random_split = [(known & (r < 0.7), known & (r >= 0.7))]

    out = {"links": int(len(nav)), "known": int(known.sum()), "missing": int((~known).sum()),
           "all_two_way": {"blocks": float(np.mean(np.concatenate([y[te] for _, te in blocks]) == "B")),
                           "exam": float(np.mean(y[~known] == "B"))},
           "how_we_started": {"random_70_30": acc(mg.CONFIGS["ATTR_2stage"], random_split),
                              "hidden_areas": acc(mg.CONFIGS["ATTR_2stage"], blocks)},
           "ladder": []}
    for label, cfg in STEPS:
        row = {"step": label, "blocks": acc(mg.CONFIGS[cfg], blocks), "exam": acc(mg.CONFIGS[cfg], exam)}
        out["ladder"].append(row)
        print(f"{label:62s} hidden areas {row['blocks']:.4f}   exam {row['exam']:.4f}", flush=True)
    team = json.loads((RESULTS / "score_TeamA.json").read_text())
    out["final"] = team
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "story.json").write_text(json.dumps(out, indent=2))
    print("random 70/30 vs hidden areas (attributes only):", out["how_we_started"])
    print("done", RESULTS / "story.json")


if __name__ == "__main__":
    main()
