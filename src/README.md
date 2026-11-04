# barcahackathon / src

Own pipeline for More than a Hack 2027, Phase 1. Everything outside `src/` is the upstream
repository (remote `upstream`), kept as is so `git pull upstream main` keeps working.

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
.venv/bin/python src/run.py --name try --set switch_cost=30  # one-off override
.venv/bin/python src/tune.py --grid switch_cost=10,20,30 dead_cost=10,20
.venv/bin/python evaluation/run_evaluation.py                # official scorer, all results
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
```

The first run parses the raw ATD (~200 MB) and caches it in `cache/` as `.npz`; later runs take
a few seconds.

## How it works

```
ATD (players + ball) ──► features ──┐
                                    ├──► per-frame evidence (H, A, D) ──► Viterbi ──► labels
Smart Tagging ─────────► priors ────┘
```

| module | does |
|---|---|
| `possession/atd.py` | raw ATD to arrays `players (N, 46, 3)`, `ball (N, 3)`, slot teams; npz cache |
| `possession/features.py` | metres; ball interpolated over short gaps; nearest home/away distance; ball speed; mean player speed; ball outside pitch |
| `possession/tagging.py` | Smart Tagging to per-frame attacking-team prior, set-piece windows and restart frames |
| `possession/model.py` | evidence scores per frame, then a switch-penalised Viterbi per half |
| `possession/config.py` | every tunable number (`Params`); `src/params.json` is the submitted config |
| `possession/pipeline.py` | glue: load, predict, write and validate, score against the public GT |

**Evidence.** On frames with a ball position, the team whose nearest player is closer gets
evidence, scaled by the distance gap and faded by `exp(-nearest / control_radius)`, so a loose
ball far from everyone says almost nothing. Dead-ball evidence comes from a ball tracked outside
the pitch, long missing-ball runs, slow players, and the start of tagged set pieces; a fast-moving
observed ball counts against dead.

**Decoder.** States H, A, D; switching H<->A costs `switch_cost`, alive<->D costs `dead_cost`.
A team only takes possession once the accumulated evidence beats the cost, which is the "controlled
possession" rule from CHALLENGE.md, and with weak evidence the last team keeps the ball. Each half
starts dead (kickoff) and is decoded separately; everything outside the halves is `X`.

## Scores (public 5 minutes, 17 possessions)

| | spellF1 | macroF1 | possessions |
|---|---|---|---|
| baseline_smooth125 (upstream) | 0.4242 | 0.5812 | 16 |
| `params.json` | 0.4800 | 0.7112 | 18 |

The public window is tiny, and upstream shows that tuning on it can pick the wrong config for the
full match. Prefer changes that make sense on video over changes that only move this number.

## v2: TXT + XML metadata + homography

```bash
venv/bin/python src/run_v2.py                    # params_v2.json -> evaluation/results/barca_v2.csv
venv/bin/python src/run_v2.py --bootstrap 1000   # with the bootstrap interval
venv/bin/python src/run_v2.py --grid w_memory=0,0.6 d_bias=-0.1,0   # sweep, no file written
venv/bin/python evaluation/run_evaluation.py     # official board
```

```
metadata (XML) ─┐
ATD (TXT) ──────┼──► features_v2 ──┐
homography ─────┘                  ├──► model_v2.evidence ──► Viterbi per half ──► labels
Smart Tagging ──► tagging.priors ──┘
```

| module | does |
|---|---|
| `possession/metadata.py` | XML: home/away ids and names, slot -> team (checked against the TXT), field size, fps, channel units, attack direction per half (swapped at halftime); real halves from `games.py` |
| `possession/homography.py` | matrices normalised to h22 = 1, quality = valid x exp(-reprojection_error), image size inferred from the projected TXT ball, camera aim / zoom / pan speed, pitch-point `in_view` |
| `possession/features_v2.py` | v1 features plus last-seen ball, metres to nearest line, last ball still in shot, players in view, per-half robust z-scores (median/MAD) of player, camera and ball speed |
| `possession/model_v2.py` | v1 evidence, team evidence gated by normalised slowness and homography quality; D from slowness, still camera, set-piece window before the tagged start, and a dead memory (ball leaves play -> dead until a fast restart kick) |
| `possession/pipeline_v2.py`, `run_v2.py` | glue, validated write, official metrics on the public window, GT-free whole-match summary |

| public 5 min | spellF1 | macroF1 | D F1 | transF1 | possessions |
|---|---|---|---|---|---|
| v1 `params.json` | 0.4800 | 0.7112 | 0.515 | 0.381 | 18 |
| v2 `params_v2.json` | 0.4970 | 0.7735 | 0.674 | 0.410 | 16 |

The spellF1 gap is inside the public noise band (bootstrap ~0.35 on 17 possessions). The macroF1 and
dead-ball gains came from the new terms: with them switched off (w_memory=0, team_gate=0) macroF1
fell back to ~0.72. Whole match: 222 possessions, mean 14.6 s,
D share ~49% (v1 ~43%) -- watch that number, it is the most likely over-fit.

On the tactical feed the camera is wide (~60 m across) and the last ball position is almost always
still in shot, so camera aim is a poor ball proxy (~23 m median error); its value here is quality
weighting and stillness. On broadcast, `valid = 0` and `in_view` would carry much more signal.

## Open ideas

- Smart Tagging prior (`w_tag`) hurts on the public window: check its timing and team mapping
  against the video before trusting it.
- Dead-ball detection is the weakest class (D F1 ~0.5): restarts, ball out of play, goal kicks.
- Ball in the air: the nearest player is not in control while the ball is high and fast.
- Optional CV: detect the ball on frames where the ATD lost it (homography projects tracking to
  pixels, which gives training labels for free).

## Rules to keep in mind

- Labels must come out of the algorithm, never hand-annotated.
- `src/run.py` must regenerate the submitted file exactly.
- No match data (or extracts) in the public repo, and none sent to hosted LLMs.
