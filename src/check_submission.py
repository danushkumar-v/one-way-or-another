"""Check a prediction folder against the challenge files: same files, columns, rows and given values.

    python src/check_submission.py predictions/<team>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "simulated" / "challenge"
FILES = ["links.csv", "truck_limits.csv", "turn_bans.csv"]
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail and not cond else ''}")


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)   # raw strings, "" stays ""


def main(sub):
    sub = Path(sub)
    print(f"folder name: {sub.name}  (must be your team name)")
    for fn in FILES:
        p = sub / fn
        print(f"\n{fn}")
        if not p.exists():
            print("  [----] not in submission yet")
            continue
        orig_p = next(RAW.rglob(fn), None)
        if orig_p is None:
            print("  [----] challenge file not found, cannot compare")
            continue
        s, o = read(p), read(orig_p)
        check("same column names and order", list(s.columns) == list(o.columns))
        if fn.startswith("links"):
            check("same number of rows", len(s) == len(o))
            if len(s) != len(o) or list(s.columns) != list(o.columns):
                continue
            check("same link_id in same order", (s["link_id"] == o["link_id"]).all())
            d = s["direction"].str.strip()
            bad = (~d.isin(["F", "T", "B"])).sum()
            check("every direction is F, T or B", bad == 0, f"({bad} rows)")
            od = o["direction"].str.strip()
            given = od.isin(["F", "T", "B"])
            changed = (given & (od != d)).sum()
            check("originally known direction values unchanged", changed == 0, f"({changed} rows)")
            others = [c for c in o.columns if c != "direction"]
            diff = [c for c in others if not (s[c] == o[c]).all()]
            check("all other columns byte-identical", not diff, f"(columns differing: {len(diff)})")
            check("no duplicate link_id", not s["link_id"].duplicated().any())
        else:
            for c in ("link_id", "link_ids", "limit_type"):
                if c in s.columns:
                    check(f"'{c}' has no blanks", (s[c].str.strip() != "").all())
            if "limit_type" in s.columns:
                t = s["limit_type"].str.strip().str.upper()
                check("restriction types in {HEIGHT, WEIGHT, NO TRUCKS}", t.isin(["HEIGHT", "WEIGHT", "NO TRUCKS"]).all())
            orig_rows = set(map(tuple, o.astype(str).values))
            sub_rows = set(map(tuple, s.astype(str).values))
            check("all original rows still present", orig_rows <= sub_rows)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
