# barcahackathon / src

Own pipeline for More than a Hack 2027, Phase 1. Everything outside `src/` is the upstream
repository (remote `upstream`), kept as is so `git pull upstream main` keeps working. The method,
the possession rules and what was tried and dropped are in [TECHNICAL.md](TECHNICAL.md).

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-src.txt
```

The match data lives outside git. `data/tactical` and `data/broadcast` are symlinks to the
unpacked bundle (or set `MTAH_DATA_DIR`). `data/*`, `cache/` and `evaluation/results/*.csv` are
gitignored: never commit the data or a submission.

## Run

```bash
.venv/bin/python src/run.py                                  # -> evaluation/results/barca.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py                # official scorer, public window
```

The first run parses the raw ATD (~200 MB) into `cache/` and fits the shape model (about two
minutes on 12 CPU cores); later runs reuse both. `src/run.py` regenerates the submitted file
exactly from `src/params.json`.

Analysis and tuning:

```bash
.venv/bin/python src/analyze.py --segments                   # error analysis, aggregates only
.venv/bin/python src/robustness.py --seeds 5                 # score with the ball hidden on 41%
.venv/bin/python src/tune.py --grid switch_cost=12,20 --hide 0.41
.venv/bin/python src/run.py --name try --set switch_cost=30  # one-off override
.venv/bin/python src/experiments/cv_probe.py                 # the CV check, needs requirements-experiments.txt
```

## How it works

```
ATD ──► clean ──► evidence ──► events ───────────────────────────► transition bonuses ─┐
                     │                                                                  │
                     └──► emissions (ball) ──► HSMM decode #1 ──► shape model ──► emissions ──► HSMM decode #2, #3
Smart Tagging ──► set-piece starts ───────────────────┘          (players only,
                                                                   self-trained)
```

| module | step | does |
|---|---|---|
| `possession/atd.py` | 1 | raw ATD to arrays, slot teams from the metadata; npz cache |
| `possession/clean.py` | 1 | metres; off-pitch detections and slot jumps dropped; ball spikes and flicker islands removed; short plausible gaps interpolated; velocities; attacking direction per half |
| `possession/evidence.py` | 2 | per frame: nearest player per team, ball-vs-player relative speed, control probability, ball location against the lines, resting ball, player speed |
| `possession/events.py` | 3 | touches, passes, turnovers, deflections, duels, shots, stoppages (ball over a line, ball vanishing outwards, collective slowdown, tagged set piece), restarts (placed ball struck, throw-in, ball found again), stoppage-to-restart dead intervals |
| `possession/shape.py` | 2/4 | players-only H/A/D classifier from team shape and motion, trained on the ball-based decode, cross-fitted over time blocks |
| `possession/fuse.py` | 4 | emissions over the HSMM states and per-frame transition bonuses from events |
| `possession/hsmm.py` | 4 | states H_ctrl, H_loose, A_ctrl, A_loose, D_1..D_m; Viterbi per half |
| `possession/tagging.py` | 3 | Smart Tagging to per-frame phase team and set-piece starts |
| `possession/config.py` | | every tunable number (`Params`); `src/params.json` is the submitted config |
| `possession/pipeline.py` | | glue, caching, writing and validating the CSV |
| `experiments/` | 5 | the CV ball-recovery check, not part of the pipeline |

## Rules to keep in mind

- Labels must come out of the algorithm, never hand-annotated.
- `src/run.py` must regenerate the submitted file exactly.
- No match data (or extracts) in the public repo, and none sent to hosted services. Analysis
  scripts print aggregates only; nothing here writes frames or tracking out.
