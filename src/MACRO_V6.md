# Binary dead-ball head: public macroF1 0.957088, spellF1 0.875000

The November 17 default adds a binary alive/dead classifier to
[macro v5](MACRO_V5.md). Its public macroF1 increases from 0.956130 to
**0.957088**, while spellF1 stays **0.875000**. The earlier model and
`configs/macro_v5.json` remain available.

All selection reuses the same 7,500 public frames and 17 GT possessions.
These numbers are held-out classifier predictions, with public-selected
features and methods; private and unseen-match performance are unknown.

## Approach

The physical baseline, four earlier classifiers, ordered H/A routing,
oscillation removal and short-spell agreement are unchanged. The new head is
an imputed, standardised binary logistic classifier with C=1. It uses 28
FILM, flow, tracking-quality and player-motion features, omitting the
existing decoder's dead-ball scores and labels. It learns only whether a
frame is D, leaving team selection to the previous models.

Let q be v5's probability vector after its primary/auxiliary P(D) average,
and d the new head's P(D). Apply:

```
new_D = 0.5 * q_D + 0.5 * d
new_H = (1 - new_D) * q_H / (q_H + q_A)
new_A = (1 - new_D) * q_A / (q_H + q_A)
```

This preserves conditional H/A odds. Missing specialist coverage abstains;
unsupported primary frames retain the physical baseline. The specialist
does not supply auxiliary team probabilities or temporal boundaries.
Existing 12-frame D flexibility and fixed D cores remain unchanged.

The new classifier uses the same 301 native 1 Hz rows, five contiguous folds
and 125-frame purge as the other members. Each whole-half decode uses the
matching held-out specialist before temporal cleanup and public stitching.
All-public refits apply only outside the public window. Schema 6 records the
specialist and its weight; older schema paths keep their prior predictions.

## Paired sensitivity and limits

| Time folds | v5 macroF1 | v6 macroF1 | SpellF1, both |
|---|---:|---:|---:|
| 3 | 0.941304 | 0.941007 | 0.729412 |
| 4 | 0.938985 | 0.942031 | 0.752941 |
| 5, deployed | 0.956130 | **0.957088** | 0.875000 |
| 6 | 0.951242 | 0.951251 | 0.825000 |

The gain is small and not uniform: three folds regress by 0.000297.
Repartitions contain the same possessions and are not independent test sets.
At five folds H/D class F1 improves, while away F1 decreases:

| Class | v5 F1 | v6 F1 |
|---|---:|---:|
| H | 0.955923 | 0.958455 |
| A | 0.970389 | 0.968291 |
| D | 0.942077 | 0.944519 |

The public output still has 15 possessions: 14 matches, one spurious and
three missed GT spells at every IoU threshold from 0.3 through 0.7.
One-second transition F1 remains 0.736842. Forty public labels change from v5.

Rejected screens include fused context replacement, extra time context,
five sampling phases, wider D edges, contact recalibration and alternate
binary classifiers/gates. YOLO11s pose plus jersey-colour features gave
at most a 0.000033 gain over v5, below the selected candidate, and were not
integrated. Detailed aggregate trial records and their protocol limits are
in `bench/results/macro_v6_2026-11-17.json`.

## Predicted match statistics

| Window | Possessions | Mean duration | Home share of alive play | Dead share |
|---|---:|---:|---:|---:|
| First half | 203 | 8.75 s | 61.76% | 44.10% |
| Second half | 198 | 8.34 s | 58.85% | 48.00% |
| Both halves | **401** | **8.55 s** | **60.36%** | **46.05%** |

Away has 39.64% of alive time. Home has 198 spells averaging 10.45 s;
away has 203 averaging 6.69 s. Counts are H 51,731 / A 33,971 / D 73,155 /
X 40,462, with 130 D intervals counted separately by half. Across the match,
138 labels change from v5, including 93 D/alive changes. These statistics
describe predictions over the graded halves, without full-match GT scoring.

## Validation and reproduction

All 51 contract tests pass. They include binary target mapping, constant
target fallback, held-out-label perturbation, off-coverage preservation,
zero-weight compatibility and schema validation. All six fitted specialist
models match the prototype coefficients exactly. Prototype, builder,
production and frozen-shape tuner match over all 199,319 frames. V5 still
reproduces its previous CSV exactly. The official validator and public scorer
confirm the candidate.

With v5's member artifacts and feature caches already prepared:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v5.json --interp film_dense \
  --sets film_dead_motion --models dead_logreg --logreg-c 1 \
  --weights .5 --lock-dead 1 --objective macro \
  --out pretrain_data/macro_dead_video --name macro_dead_video
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
           pretrain_data/macro_roi_fused/film_soft_logreg.joblib \
           pretrain_data/macro_dead_video/film_dead_motion_dead_logreg.joblib \
  --weights .75 .25 0 0 0 --team-expert 2 --team-expert 3 \
  --consensus-weights .5 0 .5 0 0 --consensus-dead-weight .5 \
  --oscillation-max-frames 100 --dead-expert 4 --dead-expert-weight .5 \
  --params src/configs/macro_v5.json \
  --out pretrain_data/macro_v6/ensemble.joblib --name macro_v6
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v5.json --params src/configs/macro_v6.json \
  --out cache/improve_nov17/production_fold_audit.json
```

Recommended local submission: `evaluation/results/Borrow_Khan.csv`, identical
to `barca.csv`. CSV SHA-256:
`3d0706bb818ad9ff8ea152de75d57a954e1b0c52040e3b21b9fe764699215e21`.
Model, data and output files remain gitignored.
