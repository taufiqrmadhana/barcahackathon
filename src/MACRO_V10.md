# Broadcast availability: public macroF1 0.963167, spellF1 0.903226

V10 retains [v9's](MACRO_V9.md) classifiers and temporal rules, replacing only
its broadcast phase classifier. It adds an explicit availability feature and
trains on all 301 native public rows, including frames with missing or rejected
broadcast geometry. The 17 geometric columns remain NaN at those frames;
imputation and standardisation are fitted inside each training fold.

| Public metric | v9 | v10 |
|---|---:|---:|
| MacroF1 | 0.962321 | **0.963167** |
| SpellF1 | 0.903226 | 0.903226 |
| Home F1 | 0.961832 | 0.962539 |
| Away F1 | 0.967393 | 0.967305 |
| Dead F1 | 0.957737 | 0.959656 |
| Frame accuracy | 0.962933 | **0.963600** |

Nineteen public labels change: 12 errors are corrected and seven previously
correct labels regress. All changes are alive-to-D. The net gain is five correct
frames, reducing errors from 278 to 273. This is a small improvement with a
class tradeoff, not a statistically established generalisation gain.

## What changed

`broadcast_dead_available` has 18 features: the earlier 17 plus the observable
quality gate as `broadcast_available`. Its binary logistic classifier keeps
`C=0.3`, five time folds, a 125-frame purge measured from actual fold edges,
and the existing 1 Hz training grid. The availability-aware head supports all
21,175 primary-video frames, compared with v9's 15,419. Geometry itself still
passes the same clock, homography and player-agreement checks. The underlying
geometry gate supports 5,590 public frames; this count is unchanged.

Ordinary `broadcast_dead` heads keep their earlier quality-only coverage.
Feature names explicitly identify the new policy, so v9 still regenerates its
exact CSV. The ensemble schema and other five members are unchanged. The phase
branch still combines 75% broadcast P(D) with 25% previous P(D), uses 50-frame
edge limits and pointwise retention of primary teams. Fixed D cores, half
boundaries and primary-video coverage remain protected.

## Paired sensitivity and control

| Time folds | v9 macroF1 | v10 macroF1 | v9 / v10 spellF1 |
|---|---:|---:|---:|
| 3 | 0.950779 | **0.955526** | 0.700000 / 0.700000 |
| 4 | 0.947047 | **0.948389** | 0.775758 / 0.775758 |
| 5, deployed | 0.962321 | **0.963167** | 0.903226 / 0.903226 |
| 6 | 0.949979 | **0.953040** | 0.825000 / 0.825000 |

Mean macroF1 increases from 0.952531 to 0.955030. SpellF1 is unchanged at every
partition. The production audit reproduces the prototype's eight rows exactly.
All partitions reuse the same 7,500 frames and 17 public possessions; they are
sensitivity checks, not independent tests. Selection bias remains, and private
or unseen-match performance is unknown.

Training the original 17 columns on all rows, without the explicit availability
feature, reaches 0.962457 / 0.903226 but slightly lowers mean macroF1 across
partitions. It was not selected. The additional feature improves this control
at all four partitions. Broadcast audio, naive image motion, FILM residuals and
temporal geometry screens are documented in [bench/BROADCAST_FOLLOWUPS.md](bench/BROADCAST_FOLLOWUPS.md).

The final public prediction has 14 spells, all matched at every IoU threshold
0.3--0.7, with three GT spells missed and zero spurious predictions. Transition
F1 within one second remains 0.810811.

## Predicted match statistics

| Window | Possessions | Mean duration | Home share of alive time | Dead share |
|---|---:|---:|---:|---:|
| First half | 202 | 8.81 s | 61.63% | 43.99% |
| Second half | 199 | 8.32 s | 58.79% | 47.87% |
| Both halves | **401** | **8.57 s** | **60.26%** | **45.93%** |

Home has 197 spells averaging 10.51 s; away has 204 averaging 6.69 s and 39.74%
of alive time. Counts are H 51,765 / A 34,134 / D 72,958 / X 40,462. There are
130 D intervals by half. All 156 full-match label changes versus v9 are D/alive
changes. These statistics describe predictions without full-match GT scoring.

## Reproduction

With v9's other five members and existing video/geometry caches:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v9.json --interp film_dense \
  --sets broadcast_dead_available --models dead_logreg --logreg-c .3 \
  --strict-fold-edges --weights .5 --lock-dead 1 --objective macro \
  --out pretrain_data/broadcast_dead_available --name broadcast_dead_available
```

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
           pretrain_data/macro_dead_video_c10/film_dead_motion_dead_logreg.joblib \
           pretrain_data/broadcast_dead_available/broadcast_dead_available_dead_logreg.joblib \
  --weights .75 .25 0 0 0 0 --team-expert 2 --team-expert 3 \
  --consensus-weights .5 0 .5 0 0 0 --consensus-dead-weight .5 \
  --oscillation-max-frames 100 --dead-expert 4 --dead-expert-weight .5 \
  --phase-expert 5 --phase-weight .75 --phase-dead-margin 50 --phase-keep-teams \
  --params src/configs/macro_v9.json --out pretrain_data/macro_v10/ensemble.joblib --name macro_v10
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v9.json --params src/configs/macro_v10.json \
  --out cache/broadcast_followup_repro/v10_production_fold_audit.json
```

All 56 tests pass. Prototype, builder, production and frozen-shape tuner match
on all 199,319 frames. The fitted phase probabilities reproduce the prototype
exactly; v9 remains byte-for-byte reproducible. Official validation and the
public scorer confirm v10. Aggregates and fingerprints are in
`bench/results/macro_v10_2026-11-17.json`.

The recommended `evaluation/results/Borrow_Khan.csv` and `barca.csv` are
identical. SHA-256:
`e6f0cae02ef06eed8971aaa2e96f39b05b6c9a2e2a250b622faaba89462ca76f`.
