# Borrow Khan

Football possession estimation for More than a Hack 2027, Phase 1.
Tracking, video features and a constrained Viterbi decoder produce one label per
frame: home (`H`), away (`A`), dead ball (`D`), or outside the graded half windows (`X`).

## Results

| Public metric | Score |
|---|---:|
| Macro-F1 | 0.963966 |
| Spell-F1 | 0.903226 |
| Frame accuracy | 0.964267 |

Evaluated on 7,500 public frames with five purged time folds. The output matches
14 of 17 GT possessions, with no spurious spells. Model selection uses this same
public interval; private and unseen-match performance are unknown.

## Run

Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-src.txt
python src/run.py --name Borrow_Khan
```

The command needs the licensed match data, fitted models and extracted features.
These are not included in Git. Expected paths and preparation commands are in
[src/README.md](src/README.md).

The output is `evaluation/results/Borrow_Khan.csv` (199,319 rows).

```bash
python evaluation/validate_submission.py evaluation/results/Borrow_Khan.csv
python evaluation/run_evaluation.py --gt public --bootstrap 0 --no-detail
```

The scorer reads all CSVs in `evaluation/results/`. Macro-F1 scores H/A/D;
predicting X on a graded frame counts as a miss.

## Method

Tracking and contact cues establish continuous ownership through passes. Video
classifiers refine H/A/D probabilities, while a separate broadcast model estimates
alive/dead timing. Viterbi decoding enforces transition and duration constraints.
Short-spell agreement limits fragmentation before team and phase outputs are merged.

## Repository

| Path | Contents |
|---|---|
| `src/possession/` | Tracking, features, models and sequence decoding |
| `src/cv/` | Ball recovery, video sampling and audio extraction |
| `src/bench/` | Training, tests and evaluation exports |
| `src/submission/` | IEEE report and editable LaTeX source |
| `evaluation/` | Official scorer, validator and public GT |
| `starter/` | Original challenge examples |

[Evaluation tables and learning curve](src/bench/EVALUATION_BUNDLE.md) ·
[Technical report](src/submission/README.md) ·
[Challenge](CHALLENGE.md) · [Data](DATA.md) · [Rules](RULES.md)

## Team

Taufiq Ramadhan Ahmad, Timotius Vivaldi Gunawan, Muhammad Kevinza Faiz,
Viktor Arsindiantoro Siringoringo, Muhammad Neo Cicero Koda.

Code is MIT licensed; match data has separate competition terms. See [LICENSE](LICENSE).
External tracking and event data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).
