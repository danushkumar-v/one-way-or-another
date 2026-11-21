# Contributing

Thanks for wanting to play. There are two ways in.

## 🏆 Submit a method to the leaderboard

1. Fork the repo and create a branch named after your method.
2. Put your code in `src/` (one script is fine) and make it write three files to `predictions/<your-name>/`:
   `links.csv`, `truck_limits.csv`, `turn_bans.csv`.
   Copy the challenge files and fill them in: same columns, same rows, given values untouched, new
   restriction rows appended.
3. Check and score it:
   ```bash
   python src/check_submission.py predictions/<your-name>     # must say ALL CHECKS PASSED
   python src/score.py predictions/<your-name>                # writes score.json
   ```
4. Open a pull request with your code, `predictions/<your-name>/score.json` and a new row in the README table.

House rules:

- Train only on `data/simulated/challenge/` and `data/mapillary/`. `data/simulated/answers/` is for scoring only.
- Validate on hidden areas (`src/split.py`), not random splits. The blank zone is one big area, so your
  validation should be too.
- Say what didn't work. The honest-corner paragraph is everyone's favourite part.

## 🛠️ Improve the code or the deck

- `python run_all.py` must still finish and pass the format check.
- If you change a number that appears in the README, regenerate it: `python src/experiments.py`,
  `python src/make_deck.py`, `python tools/record_gifs.py`.
- Keep pull requests small and say why, not only what.

## Setup

```bash
pip install -r requirements.txt
pip install playwright pillow && playwright install chromium     # only for tools/record_gifs.py
```
