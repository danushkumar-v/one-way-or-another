"""One command, whole pipeline:  prepare -> split -> 3 predictions -> format check -> score.

    python run_all.py                 # writes predictions/TeamA/ and results/score_TeamA.json
    python run_all.py --team Me       # same, under your own name
    python run_all.py --baselines     # also score the "everything two-way" and sign-rule baselines

Takes about two minutes on a laptop.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(title, *args):
    t = time.perf_counter()
    print(f"\n\033[1;36m▶ {title}\033[0m", flush=True)
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT)
    if r.returncode:
        sys.exit(f"step failed: {title}")
    print(f"\033[2m  ({time.perf_counter() - t:.0f} s)\033[0m", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="TeamA")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    pred = ROOT / "predictions" / a.team
    if pred.exists():
        shutil.rmtree(pred)

    step("1/7  read the city: links, nodes, directed edges", "src/prepare.py")
    step("2/7  cut the known zone into hidden test areas", "src/split.py")
    step("3/7  direction of travel: two-stage LightGBM + flow-consistency solver", "src/flow_repair.py", "--team", a.team)
    step("4/7  turn restrictions: turn classifier + exit rule", "src/turn_model.py", "--write", "--team", a.team)
    step("5/7  truck limits: Mapillary weight signs", "src/trucks.py", "--team", a.team)
    step("6/7  format check", "src/check_submission.py", pred)
    step("7/7  score against the answer key", "src/score.py", pred)

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    shutil.copy(pred / "score.json", res / f"score_{a.team}.json")

    if a.baselines:
        import pandas as pd
        sys.path.insert(0, str(ROOT / "src"))
        from paths import CHALLENGE, LINKS, read
        b = ROOT / "predictions" / "all_two_way"
        b.mkdir(parents=True, exist_ok=True)
        nav = read(CHALLENGE / LINKS)
        nav["direction"] = nav["direction"].fillna("B")
        nav.to_csv(b / LINKS, index=False)
        step("baseline: everything two-way", "src/score.py", b)
        step("baseline: sign rules only", "src/direction.py")
        step("baseline: sign rules only (score)", "src/score.py", ROOT / "predictions" / "sign_rules")
        for name in ("all_two_way", "sign_rules"):
            shutil.copy(ROOT / "predictions" / name / "score.json", res / f"score_{name}.json")

    s = json.loads((res / f"score_{a.team}.json").read_text())
    d, c, r = s["direction"], s["truck_limits"], s["turn_bans"]
    print(f"""
\033[1;32m✔ done.\033[0m  predictions/{a.team}/
   direction of travel   accuracy {d['accuracy']:.1%}   (everything two-way would get {d['baseline_all_B']:.1%})
   turn restrictions     {r['exact_sequence']['tp']} of {r['exact_sequence']['predicted']} new rows exactly right   F1 {r['exact_sequence']['f1']:.3f}
   truck limits          {c['all']['tp']} of {c['all']['predicted']} new rows right   F1 {c['all']['f1']:.3f}
""")


if __name__ == "__main__":
    main()
