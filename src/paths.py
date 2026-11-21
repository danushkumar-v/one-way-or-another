"""Where everything lives. Every dataset in this repo is simulated - see data/README.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHALLENGE = DATA / "simulated" / "challenge"      # the puzzle: direction blanked in the "missing" zone
ANSWERS = DATA / "simulated" / "answers"          # the answer key, only read when scoring
PROC = DATA / "simulated" / "processed"           # tables written by src/prepare.py and src/split.py
MLY = DATA / "mapillary"                          # street-level evidence
PRED = ROOT / "predictions"                       # one folder per team / method
RESULTS = ROOT / "results"

LINKS = "links.csv"
LIMITS = "truck_limits.csv"
TURNS = "turn_bans.csv"


def read(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
