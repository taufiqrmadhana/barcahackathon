# Temporal agreement: public macroF1 0.956130, spellF1 0.875000

The November 16 default combines macro v4's video prediction with the retained
motion mixture at the temporal level. It improves both requested metrics while
keeping the previous checkpoints available.

| Frozen configuration | Public macroF1 | Public spellF1 |
|---|---:|---:|
| Macro v4, `configs/macro_v4.json` | 0.950003 | 0.752941 |
| **Macro v5, current default** | **0.956130** | **0.875000** |
| Earlier motion mixture, `configs/balanced_v1.json` | 0.942279 | 0.800000 |

These are held-out video predictions over the same 7,500 public frames and 17
ground-truth possessions. All method selection reuses that slice. Private and
unseen-match scores remain unknown.

## Approach

The tracking/audio/shape/contact baseline and all four classifiers are unchanged
from [macro v4](MACRO_V4.md). Primary probabilities use 0.75 dense/raw C=1 plus
0.25 raw/5 Hz C=3, followed by the ordered motion and fused-crop H/A experts.
The auxiliary prediction uses 0.5 dense/raw C=1 plus 0.5 motion C=1, as in
`balanced_v1`. All ordinary heads still use 301 native 1 Hz training samples,
five contiguous folds and a 125-frame purge.

For every held-out head, the algorithm runs these steps over whole halves:

1. Average primary and auxiliary P(D) with equal weights. Rescale H/A by the new
   alive probability, retaining the primary's conditional H/A odds. Invalid or
   missing auxiliary evidence abstains.
2. Decode both predictions independently through the existing physical/contact
   decoder. Dead-ball edits retain the existing 12-frame limit and fixed D cores.
3. In the primary result, consider H/A/H or A/H/A triplets where **each** spell
   is at most 100 frames and the entire triplet has video coverage. Merge the
   middle only when both neighbours support their shared team above chance and
   their duration-weighted mean confidence exceeds the middle's own confidence.
   Select non-overlapping proposals once, by confidence gain; merges do not cascade.
4. Match short primary and auxiliary spells of the same team at IoU >= 0.5. Use
   their union only if it remains at most 100 frames, has complete coverage of
   both models, and contains only H/A primary labels. Resolve conflicts by larger
   IoU, applying each region once.
5. Copy only this head's held-out public slice. The all-public refits supply the
   output outside the public window.

The temporal edits never cross D/X labels, missing video or half boundaries. The
earlier probabilistic P(D) average can change D through the bounded decoder; it is
not a temporal overwrite. Every auxiliary model uses the **same** held-out fold
as the primary. Using an already stitched auxiliary sequence would violate this
property, so the implementation decodes both matching heads before stitching.
No GT file is read by the temporal operations, and no frame-specific corrections
are used. Schema 5 stores the auxiliary weights and temporal settings; schemas
1–4 keep their previous inference paths.

## Ablation and sensitivity

| Five-fold change from v4 | macroF1 | spellF1 |
|---|---:|---:|
| Oscillation removal only | 0.955667 | 0.837500 |
| Agreed short-spell union only | 0.950342 | 0.788235 |
| Both temporal operations | 0.956008 | 0.875000 |
| Both plus separate P(D) average, deployed | **0.956130** | **0.875000** |

Both mixtures were refitted for each paired time-partition check:

| Time folds | v4 macroF1 | v5 macroF1 | v4 spellF1 | v5 spellF1 |
|---|---:|---:|---:|---:|
| 3 | 0.938995 | **0.941304** | 0.729412 | 0.729412 |
| 4 | 0.935521 | **0.938985** | 0.688889 | **0.752941** |
| 5, deployed | 0.950003 | **0.956130** | 0.752941 | **0.875000** |
| 6 | 0.943613 | **0.951242** | 0.729412 | **0.825000** |

These partitions contain the same possessions and are not independent test sets.
The five-fold gain cannot establish statistical significance or generalisation.
The union operation alone regresses at four folds; the complete method improves
macroF1 at every tested partition and does not reduce spellF1.

## Predicted match statistics

The public result has 15 predicted possessions versus 17 GT. At each spell IoU
threshold from 0.3 through 0.7, 14 are matched, one is spurious and three GT spells
are missed. Class F1 values are H 0.955923, A 0.970389 and D 0.942077. Transition
F1 within one second is 0.736842.

Full-match figures describe predictions over the two graded half windows:

| Window | Possessions | Mean duration | Home share of alive play | Dead share |
|---|---:|---:|---:|---:|
| First half | 204 | 8.71 s | 61.67% | 44.12% |
| Second half | 198 | 8.33 s | 58.84% | 48.05% |
| Both halves | **402** | **8.52 s** | **60.30%** | **46.08%** |

Away has 39.70% of alive play. Home has 198 spells averaging 10.43 s; away has
204 averaging 6.67 s. The labels contain H 51,650 / A 33,999 / D 73,208 / X 40,462
frames, with 130 D intervals counted separately by half. Compared with v4,
96 labels change across the match, including 69 public labels and 18 D/alive
changes. These are predicted statistics, with no full-match GT comparison.

## Validation and reproduction

All 48 tests pass, including held-out-label perturbation of both mixtures,
missing auxiliary coverage, conditional H/A preservation, team exchange, temporal
conflicts, one-pass merging and D/X/half boundaries. The builder reproduces the
prototype over all 199,319 frames. The production CLI, official validator, scorer
and paired fold audit confirm the checkpoint. Production and the frozen-shape
tuner match the builder exactly; macro v4 still reproduces its previous CSV.

With the existing v4 member artifacts and video features:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
  --weights .75 .25 0 0 --team-expert 2 --team-expert 3 \
  --consensus-weights .5 0 .5 0 --consensus-dead-weight .5 \
  --oscillation-max-frames 100 --consensus-max-frames 100 --consensus-min-iou .5 \
  --params src/configs/macro_v4.json --out pretrain_data/macro_v5/ensemble.joblib --name macro_v5
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v4.json --params src/configs/macro_v5.json
```

[Kevinza's complete-cache reproduction](KEVINZA_2026_11_16.md), rejected tree,
contact and pose trials, ablations, paired checks and artifact fingerprints are
recorded in `bench/results/macro_v5_2026-11-16.json`. Features, model checkpoints,
video crops and submission CSVs remain gitignored.
