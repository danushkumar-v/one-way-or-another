# Data

Everything in this folder is **simulated**. We built the city from open map data, chose
the file layout ourselves and blanked the west half ourselves.

## `simulated/challenge/`: the puzzle

| File | Rows | What |
|---|---:|---|
| `links.csv` | 24,164 | road links; `direction` is blank for the 18,040 links in the west |
| `truck_limits.csv` | 40 | truck restrictions, given only in the known (east) zone |
| `turn_bans.csv` | 75 | turn restrictions, given only in the known (east) zone |

### links columns

| Column | Meaning |
|---|---|
| `link_id` | unique id of the link |
| `node_a`, `node_b` | ids of the two end nodes (`node_a` is the southern end) |
| `street_name` | street name (may be empty) |
| `direction` | `F` one-way from `node_a` to `node_b`, `T` one-way from `node_b` to `node_a`, `B` both ways, empty = to predict |
| `road_class` | road hierarchy, 1 (motorway) … 5 (local street) |
| `divider` | `A` divided road, `N` not divided |
| `bridge`, `tunnel`, `priority_road` | `Y` / `N` |
| `geom` | WKT `LINESTRING`, longitude latitude (EPSG:4326) |

**`node_a`** = the end node with the lower latitude (tie → lower longitude). The geometry is always written from
`node_a` to `node_b`.

### truck limits columns

`link_id`, `city`, `limit_type` (`HEIGHT`, `WEIGHT`, `NO TRUCKS`),
`limit_value` (HEIGHT in **cm**, WEIGHT in **kg**, `0` for NO TRUCKS), `geom` (WKT `MULTILINESTRING`).

### turn bans columns

`link_ids` (JSON list of link ids as strings: the **from** link first, then the to-link(s); a single link = no U-turn),
`city`, `geom` (WKT `MULTILINESTRING`, one part per link).

## `simulated/answers/`: the answer key

The same three files, fully filled in. `src/score.py` compares your predictions against them. Training on them is
cheating, and the leaderboard will know (well, it won't, but you will).

## `mapillary/`: street-level evidence

See [mapillary/README.md](mapillary/README.md).

## `simulated/processed/`

Written by `src/prepare.py` and `src/split.py`; not in git.

## Licence

The simulated city is derived from map data © OpenStreetMap contributors and is available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/). If you publish something built on
it, keep that attribution.
