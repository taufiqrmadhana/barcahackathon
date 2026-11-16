# External possession benchmark and supervised ownership prototype

The later [frozen DINOv2 experiments](DINO_EXPERIMENTS.md) test class/patch
features with the macro-v7 ensemble. All 66 visual variants fail to improve
their corresponding retained checkpoint; the local extraction and screening
commands are documented there.

Metrica Sports sample games 1 and 2 are used with acknowledgement to
<https://github.com/metrica-sports/sample-data>. Their event-derived possession labels
are **weak supervision**, not corrected tracking ownership. The labelling convention
is documented in `metrica_open.py`; `ownership_bench.py` records source hashes, label
hashes, frame alignment, conflicting ownership timestamps, spell statistics and
conditional missingness. Labelling rules are frozen for this first experiment.

## Reproduce the first experiment

Run from the repository root, using the existing `.venv`:

```bash
.venv/bin/python src/bench/metrica_open.py --download
.venv/bin/python src/bench/ownership_bench.py --audit-only
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/bench/ownership_bench.py
.venv/bin/python -m unittest discover -s src/bench -p 'test_*.py' -v
```

The full run reads the competition ball **locally**, only to transplant its missingness
mask. It does not read competition ground truth. Nothing is uploaded. Results, dataset
and code hashes, exact parameters and fold checkpoints go to `cache/ownership_v1/logo/`.
Use a new `--out` directory when code or settings change; an existing manifest must
match exactly. The default run uses one evaluation corruption seed. More seeds can be
specified with `--seeds 0 1 2`, with a fresh output directory.

Each outer fold trains on the other game, clean plus a corrupted copy. Corruptions are
applied before cleaning and features: empirical ATD ball gaps, smooth player position
noise, player dropouts, same-team slot exchanges, and smooth ball position noise.
Noise/dropout rates are explicit assumptions, not fitted ATD statistics. Training and
evaluation use different random seeds. All variants of a game stay in the same fold.

The evaluation conditions are clean, ATD ball masking, and masking plus player/ball
corruption. Shape is rebuilt for each condition with that condition's input, including
all pseudo-label rounds. Within a condition its predictions and the D evidence are
then shared across all four comparisons:

| Variant | Final decode change |
|---|---|
| A | Existing pipeline, submitted parameters |
| A_near | `w_near=0.75`, `w_near_switch=10`, `near_rounds=0` |
| B_owner | Add supervised ownership with weight 1 |
| B_both | Also add directed boundary evidence with weight 12 |

These settings are fixed before the run. The script does not rank, select or refit
parameters from held-out scores. This is an initial two-game comparison, not nested
hyperparameter search and not evidence of generalisation across many competitions.
The baseline still self-trains on unlabelled held-out tracking, as it does in deployment;
the supervised heads never train on held-out labels.

## Model and integration

`possession/ownership.py` uses two binary histogram gradient boosting models: own-team
possession conditional on alive play, and a direct opponent-to-own-team transition
within +/-0.4 s. Each classifier is shared between home/away views. Features use the
three nearest players per team, distances, relative/radial speed, motion alignment,
ball speed/acceleration, line distances, observed/known flags and time to/from a ball
observation. Five temporal offsets span +/-1 s. No slot identity, event-derived feature,
absolute team name or validation-derived normalisation is used.

Ownership evidence is applied equally to a team's control and loose states. It is
added only to the **final** decode, leaving pseudo-labels and shape training unchanged.
The first prototype retains old control evidence for an additive ablation; it is not
yet the full replacement proposed in option B. Boundary evidence takes the maximum
with existing team-switch bonuses, avoiding additive double reward for one touch.
Deep in a missing-ball gap (no observation within 1 s), the heads abstain and the
baseline shape model remains responsible.

D emissions and D-entry/exit bonuses are unchanged. Final D segments can nevertheless
move because the Viterbi path is coupled to the new H/A scores. The report includes
the number of D frames changed to make that interaction visible.

The boundary head estimates a *neighbourhood* probability, not a per-frame hazard or
an HSMM posterior. No MBR claim is made. A future MBR decoder needs separately checked
probability calibration and a structured loss implementation.

## Fit a deployable checkpoint and run a public sanity check

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/bench/ownership_bench.py \
  --fit-all --out pretrain_data/ownership_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/run.py \
  --name ownership_v1 --set ownership_model=pretrain_data/ownership_v1/ownership_all.joblib
.venv/bin/python evaluation/validate_submission.py evaluation/results/ownership_v1.csv
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/bench/public_ownership.py
```

The checkpoint includes the training game list and full experiment manifest. Load
only locally produced/trusted joblib files. The pipeline rejects mismatched feature
schemas and cleaning settings. `ownership_model` defaults to empty, so `src/params.json`
and the existing default submission remain unchanged. The public window is a sanity
check, not a selector for model weights.
`public_ownership.py` also verifies exact parity between the production integration and
the frozen-shape experiment, then rebuilds shape for two transplanted public-window
masks. It writes aggregates to `cache/ownership_v1/public_sanity.json`.

## Stage 2: game 3, boundary audit and supervised shape (`ownership_v2`)

Game 3 (EPTS tracking + JSON events) is converted once with kloppy 3.19.1:

```bash
.venv-pretrain/bin/python src/bench/metrica_epts.py
.venv/bin/python src/bench/boundary_audit.py          # -> cache/ownership_v2/boundary_audit.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/bench/shape_bench.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/bench/shape_bench.py \
  --fit-all --out pretrain_data/ownership_v2
```

Game 3 adds 375 spells (1,039 total, 19.7 per 5 min, 39.7% under 4 s). The boundary
audit found that treating `CARRY` as live play or ignoring advantage fouls changes no
label frame in any game. Game 3 has one timestamp where both teams claim ownership;
training masks +/-1 s around it (51 frames), evaluation keeps the frozen labels. In all
three games the ball is visible at more than 99% of direct team transitions, so ATD
gaps over a transition exist only through the transplanted masks.

`shape_bench.py` is a three-game leave-one-game-out run. Shape stays self-trained
for pseudo-labels; `supervised_shape.py` replaces only the final H/A ratio of the
shape model, keeping its P(D), so D emissions are bit-identical (asserted). Variants:
A baseline, B_ball supervised ownership/transition (as stage 1), S_shape supervised
shape only, BS_both both. Weights are the stage-1 values, fixed before the run.

Mean spellF1 over the three held-out games (one corruption seed):

| Condition | A | B_ball | S_shape | BS_both |
|---|---:|---:|---:|---:|
| Clean | 0.480 | 0.635 | 0.484 | **0.643** |
| ATD ball mask | 0.413 | 0.523 | 0.450 | **0.562** |
| Mask + tracking corruption | 0.363 | 0.472 | 0.400 | **0.519** |

BS_both is best in all nine game x condition cells. Supervised shape alone barely
moves clean scores but adds +0.04 on masked inputs, where the shape model carries the
missing-ball frames; on top of ownership it adds +0.04-0.05. Short-spell recall (mean
of IoU 0.3-0.7) under the ATD mask: 6.9% (A) -> 23.6% (B_ball) -> 26.2% (BS_both).
Team-transition F1 within 1 s under the mask: 0.28 -> 0.44 -> 0.50. Predicted spells
remain well below the labels (276 vs 346 mean under the mask), so under-segmentation
is still the main error.

Public sanity check of the all-games checkpoints (same transplant seeds as stage 1;
17 spells, so one spell moves spellF1 by roughly 0.03-0.06):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python src/bench/public_ownership.py \
  --model pretrain_data/ownership_v2/ownership_all.joblib \
  --shape-model pretrain_data/ownership_v2/shape_all.joblib \
  --name ownership_v2 --out cache/ownership_v2/public_sanity.json
```

| Public condition | A | B_ball | S_shape | BS_both |
|---|---:|---:|---:|---:|
| Clean | 0.514 | 0.550 | 0.369 | 0.550 |
| Transplant seed 0 | 0.490 | 0.400 | 0.635 | 0.600 |
| Transplant seed 1 | 0.459 | 0.357 | 0.662 | 0.563 |

Ownership alone still regresses on both masks, as in stage 1. Adding supervised shape
removes that regression: BS_both is above A in all three public conditions. Supervised
shape alone drops the clean public score, so it is only a candidate together with
ownership. Production parity with `ownership_model` + `supervised_shape_model` set is
exact; `evaluation/results/ownership_v2.csv` passes the validator. Nothing was selected
on these numbers and `src/params.json` is unchanged.

## Limits and next decisions

- Both clean Metrica games have almost all event-labelled D frames without a ball.
  OR-ing ATD gaps onto these games does not recreate ATD's conditional missingness;
  the masked run is a stress test. It can produce roughly 65-67% missing ball.
- Aggregate spell counts cannot establish boundary correctness. Advantage, recovery,
  and restart conventions still need semantic audit; no manual labels are generated.
- Report short-spell recall across the official IoU thresholds, transition F1 within
  1 s, signed offsets among matched transitions, possession count, and frame accuracy
  split by observed/missing ball. A single overall spellF1 can hide regressions.
- Add game 3 before broad parameter search. Test supervised shape as a separate
  ablation after assessing this ownership model's transfer to ATD.
