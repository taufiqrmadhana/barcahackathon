# Broadcast phase evidence: macroF1 0.962146, spellF1 0.903226

The November 17 default adds geometrically aligned broadcast tracking to
[macro v7](MACRO_V7.md). A separate phase branch decides D/alive; its output
retains v7's established H/A labels wherever both branches predict alive play.

| Public metric | v7 | v8 |
|---|---:|---:|
| MacroF1 | 0.957604 | **0.962146** |
| SpellF1 | 0.875000 | **0.903226** |
| Home F1 | 0.958455 | 0.961832 |
| Away F1 | 0.968889 | 0.968141 |
| Dead F1 | 0.945467 | 0.956464 |
| Transition F1 within one second | 0.736842 | 0.810811 |

All selection reuses the same 7,500 public frames and 17 GT possessions.
Held-out fitting does not remove this method-selection bias. Private and
unseen-match performance are unknown. V7 remains in `configs/macro_v7.json`.

## Synchronise the second recording without possession labels

Broadcast ATD and homography use a different clock from the tactical reference.
`possession/broadcast.py` matches 5-minute ball trajectories over each graded
half. One-second cross-correlation screens time offsets and pitch reflections;
frame-level refinement clips large geometric errors to limit replay/ghost impact.
Accept anchors with median ball discrepancy below 2 m, and require at least
three consistent non-reflected anchors per half with a span of at most five frames.
The half's mapping uses the median accepted offset.

Both halves have ten accepted anchors, spanning three and two frames respectively.
No possession labels or broadcast RGB frames enter this alignment.
Unreliable camera frames, including many replays and close-ups, abstain. Quality
requires valid homography with reprojection error below 1, at least five broadcast
and eight tactical players, and median same-team nearest-player discrepancy below
1.5 m. Off-pitch and implausibly fast player tracks are excluded.

The gate supports 94,058 graded frames, including 5,590 public frames (74.53%).
Median supported player discrepancy is 0.288 m. These are cross-view geometric
agreement measurements, not accuracy against an independent location ground truth.

## Separate phase and team decisions

The broadcast head is a binary logistic classifier with C=0.3 and 17 geometric,
quality, ball-motion, nearest-player and contact features. It uses 223 supported
native rows, including the public endpoint. Five matching held-out heads purge
125 frames from the **actual time-fold edges**, including when coverage has holes.
Each imputer/scaler is fitted only to that head's training rows.

The previous v7 mixture, C=10 dead head and temporal rules remain the primary
branch. For every full/held-out head independently:

1. Compute v7's probability vector and fully refined team sequence using its
   existing 12-frame D-edge limit.
2. At supported expert frames, average that P(D) with broadcast P(D), 50:50,
   preserving conditional H/A odds. Decode a second phase branch with a 50-frame
   edge limit, fixed D cores and the existing auxiliary/temporal operations.
3. Take the second branch's D mask. If both branches predict alive, retain the
   primary's H/A label. Where primary D becomes alive, take the phase branch's team.
   This operation is pointwise; no further temporal processing follows it.
4. Copy only the matching head's public fold. Full-public refits apply outside
   the public window.

The expert supplies probabilities on 15,419 primary-video frames. Missing expert
probabilities abstain; zero weight or empty expert coverage gives exact v7 output.
The wider phase decode remains bounded by its fixed D cores and primary video mask.
Unsupported primary frames retain the physical baseline. Schema 7 records the
phase expert, weight, edge limit and team-retention rule; earlier schemas retain
their original inference paths.

## Ablation and paired sensitivity

| Five-fold candidate | MacroF1 | SpellF1 |
|---|---:|---:|
| V7 with wider D edges alone | 0.957604 | 0.875000 |
| Broadcast C=0.3, 50-frame edges, joint team output | 0.958850 | 0.848485 |
| Broadcast C=1, 50-frame edges, joint team output | 0.961962 | 0.903226 |
| **C=0.3 with primary team retention, selected** | **0.962146** | **0.903226** |

| Time folds | v7 macroF1 | v8 macroF1 | v7 spellF1 | v8 spellF1 |
|---|---:|---:|---:|---:|
| 3 | 0.940817 | 0.943440 | 0.729412 | 0.729412 |
| 4 | 0.942151 | 0.946871 | 0.752941 | 0.775758 |
| 5, deployed | 0.957604 | **0.962146** | 0.875000 | **0.903226** |
| 6 | 0.951251 | 0.949453 | 0.825000 | 0.825000 |

Six-fold macroF1 regresses by 0.001797. Mean macroF1 across these paired checks
rises from 0.947956 to 0.950478; mean spellF1 rises from 0.795588 to 0.808349.
They contain the same possessions and are not independent test sets.

The public output has 14 predictions: all 14 match at every official IoU threshold,
with zero spurious predictions and three GT possessions still missed. Removing the
previous spurious spell gives 28/(14+17)=0.903226. Of 32 changed public labels,
26 correct previous errors, six remain wrong, and no previously correct label regresses.
Away F1 nevertheless falls slightly because some remaining errors change class.

## Predicted match statistics

| Window | Possessions | Mean duration | Home share of alive time | Dead share |
|---|---:|---:|---:|---:|
| First half | 202 | 8.82 s | 61.62% | 43.97% |
| Second half | 198 | 8.37 s | 58.71% | 47.82% |
| Both halves | **400** | **8.59 s** | **60.22%** | **45.90%** |

Home has 197 spells averaging 10.51 s; away has 203 averaging 6.74 s and 39.78%
of alive time. Counts are H 51,756 / A 34,191 / D 72,910 / X 40,462, with 130 D
intervals counted by half. Across the match, 299 labels change from v7, all D/alive
changes. These are predictions over the graded halves, without full-match GT scoring.

## Verification and reproduction

All 55 tests pass, including synthetic clock recovery, sparse-coverage fold-edge
purging, held-out-label perturbation, phase/team symmetry, X/coverage preservation,
zero-weight fallback and schema checks. The 17 feature arrays, coverage mask and
six fitted specialist models match the prototype exactly. Prototype, builder,
production and frozen-shape tuner agree on all 199,319 frames; v7 reproduces its
previous CSV exactly. The official validator and public scorer confirm the result.

With v7's member caches and the provided broadcast ATD/homography available:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v7.json --interp film_dense \
  --sets broadcast_dead --models dead_logreg --logreg-c .3 --strict-fold-edges \
  --weights .5 --lock-dead 1 --objective macro \
  --out pretrain_data/broadcast_dead --name broadcast_dead
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
           pretrain_data/macro_dead_video_c10/film_dead_motion_dead_logreg.joblib \
           pretrain_data/broadcast_dead/broadcast_dead_dead_logreg.joblib \
  --weights .75 .25 0 0 0 0 --team-expert 2 --team-expert 3 \
  --consensus-weights .5 0 .5 0 0 0 --consensus-dead-weight .5 \
  --oscillation-max-frames 100 --dead-expert 4 --dead-expert-weight .5 \
  --phase-expert 5 --phase-weight .5 --phase-dead-margin 50 --phase-keep-teams \
  --params src/configs/macro_v7.json \
  --out pretrain_data/macro_v8/ensemble.joblib --name macro_v8
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v7.json --params src/configs/macro_v8.json \
  --out cache/improve_nov17/v8_production_fold_audit.json
```

Broadcast feature caches bind all source files, tactical geometry, physical
parameters and output feature hashes. Alignment maps and raw geometry remain local.
Aggregate trial records and proofs are in `bench/results/macro_v8_2026-11-17.json`.
Recommended local CSV: `evaluation/results/Borrow_Khan.csv`, identical to `barca.csv`.
SHA-256: `96e37d1abc9985fbe73fcbb159fa048db81aa7de5966346cb831609bb4af648a`.
