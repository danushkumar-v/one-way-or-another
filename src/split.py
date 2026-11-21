"""Spatial validation split -> <processed>/split.parquet   (no random split)

  role       known | missing            (direction given or blank)
  holdout_A  True for the ~30% of known links closest to the missing zone (the border strip)
  block      0..4 contiguous blocks of the known zone for block cross-validation (-1 for missing links)
  edge       True if the link touches a node shared with a link of another fold/role (easier cases)

usage: python src/split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from common import PROC, run

HOLDOUT_SHARE, N_BLOCKS = 0.30, 5


def main():
    L = pd.read_parquet(PROC / "links.parquet", columns=["link_id", "node_a", "node_b", "dir_label", "mid_x", "mid_y"])
    known = (L.dir_label != "missing").to_numpy()
    xy = L[["mid_x", "mid_y"]].to_numpy()
    d = np.full(len(L), np.nan)
    if (~known).any() and known.any():
        d[known] = cKDTree(xy[~known]).query(xy[known])[0]
    hold = np.zeros(len(L), bool)
    if known.any():
        hold[known] = d[known] <= np.nanquantile(d[known], HOLDOUT_SHARE)
    block = np.full(len(L), -1)
    if known.sum() >= N_BLOCKS:
        block[known] = KMeans(N_BLOCKS, n_init=10, random_state=0).fit_predict(xy[known])

    def edge_flag(group):
        g = pd.Series(group, index=L.index)
        t = pd.concat([pd.DataFrame({"n": L.node_a, "g": g}), pd.DataFrame({"n": L.node_b, "g": g})])
        mixed = t.groupby("n").g.nunique() > 1
        return (L.node_a.map(mixed) | L.node_b.map(mixed)).to_numpy()

    out = pd.DataFrame({"link_id": L.link_id, "role": np.where(known, "known", "missing"), "dist_to_missing_m": d,
                        "holdout_A": hold, "block": block,
                        "edge_A": edge_flag(np.where(~known, 2, hold.astype(int))), "edge_block": edge_flag(block)})
    out.to_parquet(PROC / "split.parquet")
    # aggregates only
    print(f"links {len(L):,} | known {known.sum():,} | missing {(~known).sum():,} | holdout A {hold.sum():,} "
          f"(strip width {np.nanmax(d[hold]) if hold.any() else 0:,.0f} m) | blocks {np.bincount(block[known]).tolist() if known.any() else []}")
    if True:
        y = L.dir_label.to_numpy()
        for name, m in (("train (known, not holdout)", known & ~hold), ("holdout A", hold)):
            if m.sum() >= 20:
                print(f"  {name}: one-way share {np.isin(y[m], ['F', 'T']).mean():.3f}")
    return [PROC / "split.parquet"]


if __name__ == "__main__":
    run(main)
