# Availability context blend: public macroF1 0.963966, spellF1 0.903226

V11 averages two broadcast phase classifiers: [v10's](MACRO_V10.md) original
18-feature head and a 22-feature head with availability history. Both use
binary logistic regression, C=0.3, and the same purged training rows. The other
five ensemble members, weights and temporal rules stay unchanged.

| Public metric | v10 | v11 |
|---|---:|---:|
| MacroF1 | 0.963167 | **0.963966** |
| SpellF1 | 0.903226 | 0.903226 |
| Home F1 | 0.962539 | 0.963257 |
| Away F1 | 0.967305 | 0.967318 |
| Dead F1 | 0.959656 | 0.961322 |
| Frame accuracy | 0.963600 | **0.964267** |

Seven public labels change: six errors are corrected and one correct frame
regresses, reducing errors from 273 to 268. The 151 H/A confusions remain;
false D falls from 104 to 102 and missed D from 18 to 15. These are small gains
on a repeatedly reused public slice, not established generalisation gains.

## Four output labels and the binary expert

The CSV still contains H/A/D/X. Primary video classifiers estimate H/A/D;
binary experts estimate only P(D), leaving conditional team ownership to the
primary branch. X applies outside the graded half windows. The official
macroF1 averages H/A/D only; predicting X on a graded H/A/D frame counts as a
miss. A binary phase loss therefore describes one specialist, not the entire
four-label output system.

Tracking and event cues first produce a physical HSMM baseline. ML probabilities
then refine the evidence used by Viterbi HSMM decoding, followed by temporal
corrections and pointwise fusion of the team and phase branches. HSMM duration
constraints include a 100-frame minimum D interval; the ML output is not simply
an independent argmax at every frame.

## What changed

The new features are observable, label-free context computed separately within
each half: seconds since/until valid broadcast tracking (capped at 5 s), and
availability fractions in centred 1 s and 5 s windows. This is an offline system:
future observations are permitted and explicitly used. Half gaps never feed
these features. Missing geometry stays NaN until each training fold's own
imputation and scaling.

The original head sees 17 geometry features plus the availability flag; the
context head sees those 18 plus four history features. Their P(D) values are
averaged equally. The existing phase route then combines 75% of that P(D) with
25% of the previous P(D). Both classifiers use the same 301-row native 1 Hz
pool, five time folds and a 125-frame purge from actual fold edges. At inference
they cover the same 21,175 primary-video frames. The geometry gate itself still
supports 5,590 public frames.

The phase branch retains its 50-frame edge bounds, fixed D cores, temporal
corrections and primary team labels wherever both branches predict alive.
The primary branch keeps 12-frame edge bounds. Existing 17/18-feature artifacts
retain their original feature provenance and regenerate their earlier outputs.
The new context values have an additional provenance hash.

## Paired sensitivity and rejected alternatives

| Time folds | v10 macroF1 | v11 macroF1 | Both spellF1 |
|---|---:|---:|---:|
| 3 | 0.955526 | **0.955869** | 0.700000 |
| 4 | 0.948389 | **0.948545** | 0.775758 |
| 5, deployed | 0.963167 | **0.963966** | 0.903226 |
| 6 | 0.953040 | **0.954750** | 0.825000 |

Mean macroF1 rises from 0.955030 to 0.955782. Every partition improves macroF1
and preserves spellF1. These repartitions use the same 7,500 frames and 17 GT
possessions; they are sensitivity checks, not independent tests. Private and
unseen-match scores remain unknown.

Context alone reaches 0.963735 at five folds but hurts the paired mean, especially
at three folds. Bagging the original head reaches 0.963684 at five folds but also
hurts paired mean macroF1. Neither replacement was selected. Balanced weighting,
team-symmetric features, alternate C values, and confidence/logit blends were
also screened. Aggregate trials are in `bench/results/macro_v11_2026-11-17.json`.

A follow-up screen varies context weights (25/75%), stronger regularisation,
and timing-only versus fraction-only context. None improves deployed macroF1
without hurting another audited partition. Timing-only reaches 0.964138 at five
folds but falls by 0.000354 at three folds versus v11. The default stays v11.
`bench/phase_blend_bench.py --folds 5` reproduces these checks.

A further H/A-only expert screen uses 241 alive training rows from raw video,
motion and fused crops (C=0.3/1/3, plus balanced motion). None of ten candidates
improves the deployed score; stronger variants introduce wrong switches. Those
results are retained separately in `bench/results/team_alive_2026-11-17.json`,
reproduced with `bench/team_alive_bench.py`. No extra expert is deployed.

The final public output has 14 predictions against 17 GT spells: all 14 match at
every IoU threshold 0.3--0.7, with three missed GT spells and zero spurious
predictions. One-second transition F1 remains 0.810811.

## Predicted match statistics

| Window | Possessions | Mean duration | Home alive share | Dead share |
|---|---:|---:|---:|---:|
| First half | 202 | 8.82 s | 61.64% | 43.97% |
| Second half | 198 | 8.36 s | 58.82% | 47.89% |
| Both halves | **400** | **8.59 s** | **60.28%** | **45.93%** |

Home has 197 spells averaging 10.51 s; away has 203 averaging 6.72 s and 39.72%
of alive time. Counts are H 51,777 / A 34,115 / D 72,965 / X 40,462. There are
130 D intervals by half. All 45 full-match changes versus v10 are D/alive changes.
These are prediction summaries without full-match GT scoring. The 400 full-match
predictions are separate from the public 17 GT / 14 predicted spells.

## Reproduction

With v10's other five members and existing video/geometry caches:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v10.json --interp film_dense \
  --sets broadcast_dead_context --models dead_context_blend --logreg-c .3 \
  --strict-fold-edges --weights .5 --lock-dead 1 --objective macro \
  --out pretrain_data/broadcast_dead_context --name broadcast_dead_context
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
           pretrain_data/macro_dead_video_c10/film_dead_motion_dead_logreg.joblib \
           pretrain_data/broadcast_dead_context/broadcast_dead_context_dead_context_blend.joblib \
  --weights .75 .25 0 0 0 0 --team-expert 2 --team-expert 3 \
  --consensus-weights .5 0 .5 0 0 0 --consensus-dead-weight .5 \
  --oscillation-max-frames 100 --dead-expert 4 --dead-expert-weight .5 \
  --phase-expert 5 --phase-weight .75 --phase-dead-margin 50 --phase-keep-teams \
  --params src/configs/macro_v10.json --out pretrain_data/macro_v11/ensemble.joblib --name macro_v11
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v10.json --params src/configs/macro_v11.json \
  --out cache/v11_repro/production_fold_audit.json
```

`bench/phase_context_bench.py --cases context_blend_C0.3` reproduces the chosen
prototype. Pass `--folds 3`, `4` or `6` for its paired checks. The separate
learning curve and evaluation tables are described in
[bench/EVALUATION_BUNDLE.md](bench/EVALUATION_BUNDLE.md).

All 59 tests pass. Prototype, builder, production and frozen-shape tuner match
on all 199,319 frames; v10 and both fitted component probabilities reproduce
exactly. Official validation and public scoring confirm v11. The recommended
`Borrow_Khan.csv`, `barca.csv` and `macro_v11.csv` are identical, SHA-256:
`6bfa4efe8c89ea9dc0030fee8ab13f8d8f77024b1b92024d8ee64302b1334064`.
