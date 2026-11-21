"""Shared paths, loaders and the script runner."""
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CHALLENGE as RAW, PROC, ROOT  # noqa: E402,F401

OUT = ROOT / "outputs"

DIR_ORDER = ["B", "F", "T", "missing"]
DIR_COLORS = {"B": "#9ca3af", "F": "#06b6d4", "T": "#f59e0b", "missing": "#e11d9b"}
NAV_CATS = ["direction", "road_class", "divider", "bridge", "tunnel", "priority_road"]


def find(key: str) -> Path:
    m = sorted(RAW.rglob(f"*{key}*.csv"))
    if not m:
        raise FileNotFoundError(f"no *{key}*.csv under {RAW}")
    return m[0]


def _read(p: Path) -> pd.DataFrame:
    # everything as string; only the empty string counts as null ("N"/"NA" must survive)
    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    return df


def is_yes(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().isin(["Y", "YES", "T", "TRUE", "1"]).astype(bool)


def load_nav() -> pd.DataFrame:
    df = _read(find("links"))
    for c in NAV_CATS:
        df[c] = df[c].str.strip().str.upper()
    df["dir"] = df["direction"].where(df["direction"].isin(["F", "T", "B"]))
    df["dir_label"] = df["dir"].fillna("missing")
    return df


def load_truck_limits() -> pd.DataFrame:
    df = _read(find("truck_limits"))
    df["limit_type"] = df["limit_type"].str.strip().str.upper()
    df["value"] = pd.to_numeric(df["limit_value"], errors="coerce")
    return df


def load_turn_bans() -> pd.DataFrame:
    df = _read(find("turn_bans"))

    def parse(s):
        if not isinstance(s, str):
            return []
        try:
            v = json.loads(s)
        except Exception:
            v = [x for x in s.strip("[]{}() ").replace('"', "").replace("'", "").split(",")]
        if not isinstance(v, list):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()]

    df["links"] = df["link_ids"].map(parse)
    df["n_links"] = df["links"].map(len)
    return df


def heading(x0, y0, x1, y1):
    """Compass bearing in degrees (0 = north, clockwise) from point 0 to point 1 (metric coords)."""
    return (np.degrees(np.arctan2(np.asarray(x1) - x0, np.asarray(y1) - y0)) + 360.0) % 360.0


def run(main):
    warnings.simplefilter("ignore")
    try:
        outs = main() or []
        print("done")
        for o in outs:
            print(o)
    except Exception:  # noqa
        traceback.print_exc()
        sys.exit(1)
