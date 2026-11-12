# barcahackathon / src

Own pipeline for More than a Hack 2027, Phase 1. Everything outside `src/` is the upstream
repository (remote `upstream`), kept as is so `git pull upstream main` keeps working. The method,
the possession rules and what was tried and dropped are in [TECHNICAL.md](TECHNICAL.md).

Latest: `valdi` is merged into `taufiq`. The default combines Taufiq's tracking/audio/shape,
Neo's contact ownership, and Valdi's FILM video validator: public spellF1 **0.7053**, macroF1
**0.9231**. Public CSV rows are independently decoded out-of-fold predictions. Exact video
frame plan, setup, model fitting and limits: [VALDI_INTEGRATION.md](VALDI_INTEGRATION.md).
The previous contact-only configuration is preserved in `configs/contact_default.json`;
its experiments are documented in [CONTACT_EXPERIMENTS.md](CONTACT_EXPERIMENTS.md).

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
# Earlier contact-only public-score candidate:
.venv/bin/python src/run.py --params src/configs/contact_public.json --name contact_public
# Previous contact-only default:
.venv/bin/python src/run.py --params src/configs/contact_default.json --name contact_default
# Reproduce the previous submission:
.venv/bin/python src/run.py --params src/configs/ownership_v2.json --name ownership_before_contact
```

The first run parses the raw ATD (~200 MB) into `cache/` and fits the shape model (about two
minutes on 12 CPU cores); later runs reuse both. `src/run.py` regenerates the submitted file
exactly from `src/params.json`.

### Video, audio and scoreboard inputs (November update, TECHNICAL.md section 9)

The submitted config reads three things derived locally from the tactical video. They go to
`cache/cv/` (gitignored, never committed or uploaded) and need a separate environment with the
detector runtime:

```bash
uv venv --python 3.11 .venv-cv
uv pip install --python .venv-cv -r requirements-cv.txt
.venv-cv/bin/python src/cv/export_models.py          # YOLO11s COCO -> OpenVINO, INT8 (NNCF)
# pass 1: crops around the ATD ball (about 1 h on 12 cores)
.venv-cv/bin/python src/cv/detect_crops.py --start 11185 --end 54000
# pass 2: ATD-independent grid, public window and selected frames (about 3 h)
# The frame lists are committed (src/cv/frames/, frame indices only) so the pass is reproducible;
# select_grid.py is how they were made (--public, then --stride 2 against an earlier decode).
.venv-cv/bin/python src/cv/detect_grid.py --frames src/cv/frames/grid_frames_pub.txt --tag grid_pub
.venv-cv/bin/python src/cv/detect_grid.py --frames src/cv/frames/grid_frames_pub_rest.txt --tag grid_pub_rest
.venv-cv/bin/python src/cv/detect_grid.py --frames src/cv/frames/grid_frames.txt --tag grid
.venv/bin/python src/cv/build_fused.py --tag atd+grid_pub+grid_pub_rest+grid --name fused_full
.venv/bin/python src/cv/whistle.py                    # audio -> cache/cv/whistle38.npy
```

The scoreboard reading (goal -> kick-off dead intervals) runs on first use and is cached in
`cache/cv/goal_dead_BAR-ATH.json`. The ownership checkpoints come from `src/bench/` (see
bench/README.md).

Analysis and tuning:

```bash
.venv/bin/python src/analyze.py --segments                   # error analysis, aggregates only
.venv/bin/python src/spells.py                               # why each GT possession fails
.venv/bin/python src/robustness.py --sim transplant context empirical --seeds 2
                                                             # leak-free hidden-ball test
.venv/bin/python src/tune.py --grid switch_cost=12,20 --hide 0.41   # fast screen, optimistic
.venv/bin/python src/run.py --name try --set switch_cost=30  # one-off override
.venv/bin/python src/experiments/cv_probe.py                 # the CV check, needs requirements-experiments.txt
.venv/bin/python src/experiments/overlay.py --n 20           # tracking drawn on frames, to cache/overlays
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
| `possession/contact.py` | 5 | original-ATD contact vote, second H/A decode with the D mask locked, short-island cleanup; never feeds shape training |
| `possession/hybrid.py` | 6 | Valdi camera/context + FILM features, public-trained classifier, independently decoded held-out public folds; preserves D and uncovered frames |
| `experiments/` | 5 | the CV ball-recovery check and a local overlay viewer, not part of the pipeline |

## Rules to keep in mind

- Labels must come out of the algorithm, never hand-annotated.
- `src/run.py` must regenerate the submitted file exactly.
- No match data (or extracts) in the public repo, and none sent to hosted services. Analysis
  scripts print aggregates only. `overlay.py` writes frames to `cache/overlays/` (gitignored) for
  local viewing only.
- Hidden-ball numbers are robustness checks on simulated gaps, not estimates of the private score;
  `tune.py --hide` reuses shape probabilities from the unmasked run and is optimistic.
