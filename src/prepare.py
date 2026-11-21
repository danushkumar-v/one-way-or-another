"""challenge CSVs -> data/simulated/processed/  (links, directed_edges, nodes, conditions, turn_bans)

Canonical direction representation = directed_edges: one row per allowed travel direction.
  B -> 2 rows, F -> ref->nref, T -> nref->ref, missing -> 2 candidate rows with status='unknown'.

usage: python src/prepare.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from common import PROC, heading, is_yes, load_truck_limits, load_nav, load_turn_bans, run

K_ZONE = 25  # neighbours used for the "blank zone" score


def classify_turn(a):
    if a is None or not np.isfinite(a):
        return "unknown"
    if abs(a) <= 30:
        return "straight"
    if abs(a) >= 150:
        return "U-turn"
    return "right" if a > 0 else "left"


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    nav = load_nav()
    n_raw = len(nav)
    nav = nav[nav["geom"].notna()].drop_duplicates("link_id").reset_index(drop=True)

    g4326 = gpd.GeoSeries.from_wkt(nav["geom"], crs=4326)
    g = g4326.to_crs(28992)
    a_ll, a = g4326.to_numpy(), g.to_numpy()
    multi = shapely.get_type_id(a) != 1
    if multi.any():  # defensive: merge multi-part lines, keep first part if still multi
        a = np.array([shapely.line_merge(x) if m else x for x, m in zip(a, multi)], dtype=object)
        a = np.array([x.geoms[0] if x.geom_type != "LineString" else x for x in a], dtype=object)
        a_ll = np.array([shapely.line_merge(x) if m else x for x, m in zip(a_ll, multi)], dtype=object)
        a_ll = np.array([x.geoms[0] if x.geom_type != "LineString" else x for x in a_ll], dtype=object)

    f_ll = shapely.get_coordinates(shapely.get_point(a_ll, 0))
    l_ll = shapely.get_coordinates(shapely.get_point(a_ll, -1))
    f_xy = shapely.get_coordinates(shapely.get_point(a, 0))
    l_xy = shapely.get_coordinates(shapely.get_point(a, -1))
    L = shapely.length(a)
    d = np.minimum(10.0, L / 2)
    pa = shapely.get_coordinates(shapely.line_interpolate_point(a, d))
    pb = shapely.get_coordinates(shapely.line_interpolate_point(a, L - d))
    mid = shapely.get_coordinates(shapely.line_interpolate_point(a, 0.5, normalized=True))
    h_first = heading(f_xy[:, 0], f_xy[:, 1], pa[:, 0], pa[:, 1])   # leaving the first vertex
    h_last = heading(pb[:, 0], pb[:, 1], l_xy[:, 0], l_xy[:, 1])    # arriving at the last vertex

    # --- which end of the geometry is node_a (the southern end)? (do not assume)
    # rule: ref node = lower latitude, tie -> lower longitude
    rule = (f_ll[:, 1] < l_ll[:, 1]) | ((f_ll[:, 1] == l_ll[:, 1]) & (f_ll[:, 0] < l_ll[:, 0]))
    closed = (f_ll == l_ll).all(axis=1)
    ref, nref = nav["node_a"].to_numpy(), nav["node_b"].to_numpy()

    def consensus(start_is_ref):
        rxy = np.where(start_is_ref[:, None], f_xy, l_xy)
        nxy = np.where(start_is_ref[:, None], l_xy, f_xy)
        t = pd.DataFrame({"n": np.concatenate([ref, nref]),
                          "x": np.concatenate([rxy[:, 0], nxy[:, 0]]),
                          "y": np.concatenate([rxy[:, 1], nxy[:, 1]])})
        c = t.groupby("n")[["x", "y"]].median()
        return rxy, nxy, c

    # topology check: every node id should sit at ONE coordinate. If swapping a link's ends
    # fits the node positions better than the latitude rule, trust topology and flag it.
    rxy, nxy, c = consensus(rule)
    cr, cn = c.reindex(ref).to_numpy(), c.reindex(nref).to_numpy()
    d_keep = np.hypot(*(rxy - cr).T) + np.hypot(*(nxy - cn).T)
    d_swap = np.hypot(*(nxy - cr).T) + np.hypot(*(rxy - cn).T)
    swap = d_swap + 1.0 < d_keep
    starts_at_ref = rule ^ swap
    rxy, nxy, c = consensus(starts_at_ref)
    cr, cn = c.reindex(ref).to_numpy(), c.reindex(nref).to_numpy()
    off_ref, off_nref = np.hypot(*(rxy - cr).T), np.hypot(*(nxy - cn).T)

    r_ll = np.where(starts_at_ref[:, None], f_ll, l_ll)
    n_ll = np.where(starts_at_ref[:, None], l_ll, f_ll)
    h_out_ref = np.where(starts_at_ref, h_first, (h_last + 180) % 360)   # leaving ref towards nref
    h_in_nref = np.where(starts_at_ref, h_last, (h_first + 180) % 360)   # arriving at nref from ref
    chord = heading(rxy[:, 0], rxy[:, 1], nxy[:, 0], nxy[:, 1])

    # --- blank-zone score: share of the K nearest links (by midpoint) with missing direction
    miss = nav["dir"].isna().to_numpy()
    k = min(K_ZONE + 1, len(nav))
    _, nn = cKDTree(mid).query(mid, k=k)
    zone_share = miss[nn[:, 1:]].mean(axis=1) if k > 1 else miss.astype(float)

    links = nav.drop(columns=["geom"]).copy()
    links["length_m"] = L
    links["geom_starts_at_ref"] = starts_at_ref
    links["ref_rule_agrees"] = ~swap            # latitude rule consistent with node topology
    links["closed_loop"] = closed
    links["ref_node_offset_m"] = off_ref
    links["nref_node_offset_m"] = off_nref
    for nm, arr in (("ref_x", rxy[:, 0]), ("ref_y", rxy[:, 1]), ("nref_x", nxy[:, 0]), ("nref_y", nxy[:, 1]),
                    ("ref_lon", r_ll[:, 0]), ("ref_lat", r_ll[:, 1]), ("nref_lon", n_ll[:, 0]), ("nref_lat", n_ll[:, 1]),
                    ("mid_x", mid[:, 0]), ("mid_y", mid[:, 1]),
                    ("bearing_ref_to_nref", chord), ("heading_out_ref", h_out_ref), ("heading_in_nref", h_in_nref),
                    ("zone_missing_share", zone_share)):
        links[nm] = arr
    links["blank_zone"] = links["zone_missing_share"] >= 0.5
    links["is_bridge"] = is_yes(links["bridge"])
    links["is_tunnel"] = is_yes(links["tunnel"])
    links["is_divided"] = links["divider"].notna() & (links["divider"] != "N")
    gpd.GeoDataFrame(links, geometry=gpd.GeoSeries(a_ll, crs=4326)).to_parquet(PROC / "links.parquet")

    # --- directed edges
    base = links[["link_id", "dir_label", "length_m"]]
    fwd = base.assign(orientation="ref_to_nref", from_node=ref, to_node=nref, bearing=chord,
                      bearing_start=h_out_ref, bearing_end=h_in_nref,
                      from_x=rxy[:, 0], from_y=rxy[:, 1], to_x=nxy[:, 0], to_y=nxy[:, 1])
    rev = base.assign(orientation="nref_to_ref", from_node=nref, to_node=ref, bearing=(chord + 180) % 360,
                      bearing_start=(h_in_nref + 180) % 360, bearing_end=(h_out_ref + 180) % 360,
                      from_x=nxy[:, 0], from_y=nxy[:, 1], to_x=rxy[:, 0], to_y=rxy[:, 1])
    edges = pd.concat([fwd[fwd.dir_label.isin(["B", "F", "missing"])],
                       rev[rev.dir_label.isin(["B", "T", "missing"])]], ignore_index=True)
    edges["status"] = np.where(edges.dir_label == "missing", "unknown", "allowed")
    edges = edges.rename(columns={"dir_label": "direction"}).sort_values(["link_id", "orientation"]).reset_index(drop=True)
    edges.to_parquet(PROC / "directed_edges.parquet")

    # --- nodes
    nodes = c.rename(columns={"x": "x", "y": "y"}).copy()
    lon, lat = Transformer.from_crs(28992, 4326, always_xy=True).transform(nodes.x.to_numpy(), nodes.y.to_numpy())
    nodes["lon"], nodes["lat"] = lon, lat
    nodes["degree"] = pd.concat([links.node_a, links.node_b]).value_counts().reindex(nodes.index).fillna(0).astype(int)
    ok = edges[edges.status == "allowed"]
    nodes["n_out"] = ok.from_node.value_counts().reindex(nodes.index).fillna(0).astype(int)
    nodes["n_in"] = ok.to_node.value_counts().reindex(nodes.index).fillna(0).astype(int)
    ml = links[miss]
    nodes["n_unknown_links"] = pd.concat([ml.node_a, ml.node_b]).value_counts().reindex(nodes.index).fillna(0).astype(int)
    full = (nodes.n_unknown_links == 0) & (nodes.degree >= 2)
    nodes["dead_end"] = nodes.degree == 1
    nodes["only_in"] = full & (nodes.n_out == 0)     # sink: you can enter but never leave
    nodes["only_out"] = full & (nodes.n_in == 0)     # source: you can leave but never enter
    nodes.index.name = "node_id"
    nodes.reset_index().to_parquet(PROC / "nodes.parquet")

    # --- conditions (joined to link attributes)
    keep = ["link_id", "street_name", "dir_label", "road_class", "divider", "bridge", "tunnel", "priority_road",
            "is_bridge", "is_tunnel", "is_divided", "length_m", "zone_missing_share", "blank_zone"]
    cond = load_truck_limits()
    cond = cond.merge(links[keep], on="link_id", how="left", indicator=True)
    cond["in_nav"] = cond.pop("_merge") == "both"
    cond.to_parquet(PROC / "truck_limits.parquet")

    # --- turn bans: turn geometry from link headings
    idx = {v: i for i, v in enumerate(links["link_id"].to_numpy())}
    deg = nodes["degree"].to_dict()
    fc, dv, dl = links["road_class"].to_numpy(), links["is_divided"].to_numpy(), links["dir_label"].to_numpy()
    nm, bz = links["street_name"].to_numpy(), links["blank_zone"].to_numpy()

    def shared(i, j):
        s = {ref[i], nref[i]} & {ref[j], nref[j]}
        return (sorted(s)[0] if s else None), len(s)

    def arrive(i, node):
        return h_in_nref[i] if node == nref[i] else (h_out_ref[i] + 180) % 360

    def depart(i, node):
        return h_out_ref[i] if node == ref[i] else (h_in_nref[i] + 180) % 360

    def norm(x):
        return (x + 540) % 360 - 180

    rd = load_turn_bans()
    rec = []
    for ls in rd["links"]:
        ii = [idx.get(x) for x in ls]
        r = dict(from_link=ls[0] if ls else None, next_link=ls[1] if len(ls) > 1 else None,
                 last_link=ls[-1] if len(ls) > 1 else None,
                 n_in_nav=sum(i is not None for i in ii), all_in_nav=bool(ii) and all(i is not None for i in ii),
                 turn_node=None, turn_angle=np.nan, overall_angle=np.nan, chain_connected=None,
                 turn_node_degree=np.nan, from_can_enter_node=None, to_can_leave_node=None, same_name=None)
        known = [i for i in ii if i is not None]
        r["any_divided"] = bool(known) and bool(any(dv[i] for i in known))
        r["from_divided"] = bool(dv[ii[0]]) if ii and ii[0] is not None else None
        r["from_road_class"] = fc[ii[0]] if ii and ii[0] is not None else None
        r["to_road_class"] = fc[ii[-1]] if len(ii) > 1 and ii[-1] is not None else None
        r["from_dir"] = dl[ii[0]] if ii and ii[0] is not None else None
        r["to_dir"] = dl[ii[-1]] if len(ii) > 1 and ii[-1] is not None else None
        r["n_links_missing_dir"] = int(sum(dl[i] == "missing" for i in known))
        r["n_links_blank_zone"] = int(sum(bool(bz[i]) for i in known))
        if len(ii) >= 2 and ii[0] is not None and ii[1] is not None:
            node, _ = shared(ii[0], ii[1])
            if node is not None:
                r["turn_node"] = node
                r["turn_node_degree"] = deg.get(node, np.nan)
                h_in = arrive(ii[0], node)
                r["turn_angle"] = norm(depart(ii[1], node) - h_in)
                f_into = "F" if node == nref[ii[0]] else "T"
                t_out = "F" if node == ref[ii[1]] else "T"
                r["from_can_enter_node"] = None if dl[ii[0]] == "missing" else bool(dl[ii[0]] in ("B", f_into))
                r["to_can_leave_node"] = None if dl[ii[1]] == "missing" else bool(dl[ii[1]] in ("B", t_out))
                r["same_name"] = bool(isinstance(nm[ii[0]], str) and nm[ii[0]] == nm[ii[-1]]) if ii[-1] is not None else None
                # walk the chain to the last link for the overall heading change
                ok_chain, cur = True, node
                for p, q in zip(ii[1:-1], ii[2:]):
                    if p is None or q is None:
                        ok_chain = False
                        break
                    nxt, _ = shared(p, q)
                    if nxt is None:
                        ok_chain = False
                        break
                    cur = nxt
                r["chain_connected"] = ok_chain
                if ok_chain and ii[-1] is not None:
                    r["overall_angle"] = norm(depart(ii[-1], cur) - h_in)
            else:
                r["chain_connected"] = False
        r["turn_type"] = "U-turn (single link)" if len(ls) == 1 else classify_turn(r["turn_angle"])
        r["overall_type"] = "U-turn (single link)" if len(ls) == 1 else classify_turn(r["overall_angle"])
        rec.append(r)
    rd = pd.concat([rd, pd.DataFrame(rec, index=rd.index)], axis=1)
    for col in ("turn_node", "from_road_class", "to_road_class", "from_dir", "to_dir", "from_link", "next_link", "last_link"):
        rd[col] = rd[col].astype("string")
    for col in ("chain_connected", "from_can_enter_node", "to_can_leave_node", "same_name", "from_divided"):
        rd[col] = rd[col].astype("boolean")
    rd.to_parquet(PROC / "turn_bans.parquet")

    meta = dict(rows_raw=int(n_raw), rows_links=int(len(links)), multi_part_geoms=int(multi.sum()),
                k_zone=K_ZONE)
    (PROC / "prepare_meta.json").write_text(json.dumps(meta))
    return [PROC / n for n in ("links.parquet", "directed_edges.parquet", "nodes.parquet",
                               "truck_limits.parquet", "turn_bans.parquet")]


if __name__ == "__main__":
    run(main)
