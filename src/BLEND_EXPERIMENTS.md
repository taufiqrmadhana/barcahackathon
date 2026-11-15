# Complementary video classifiers, November 15

The video head receives both continuous evidence and the earlier decoders' discrete labels.
Those labels help on stable play but can repeat an earlier incorrect team assignment. A second
classifier omits those labels and their derived durations. Averaging the two classifiers'
probabilities improves both deployed public metrics:

| Public five minutes, five purged time folds | spellF1 | macroF1 | predicted possessions |
|---|---:|---:|---:|
| Previous default, `configs/pre_blend.json` | 0.7444 | 0.9292 | 19 |
| **Blended default** | **0.7529** | **0.9353** | 17 |

Transition F1 within one second rises from 0.6190 to 0.6500. H F1 rises from 0.9222 to
0.9297, A F1 from 0.9239 to 0.9347. The unchanged D mask has F1 0.9416. The ground truth
contains 17 possessions; matching its count alone does not establish correct boundaries.

## Method

`logreg_blend` in `possession/hybrid.py` is a standard sklearn soft-voting classifier:

- Original member: the same regularised logistic model and all 42 FILM/context features.
- Complementary member: the same logistic specification with 34 features. It omits
  `current_H/A/D`, `v2_is_H/A/D`, `v2_since_change` and `v2_until_change`; continuous team/dead
  scores, external shape probabilities, geometry, camera and FILM evidence remain available.
- Probability weights: one part original, three parts complementary. Feature ablations and
  complementary weights 0.25, 0.50 and 0.75 were compared on the public window. The deployed
  weight is a public-selected choice.

Both members fit their imputer, scaler and classifier separately inside each training fold.
The 1 Hz training grid, five 60-second time folds and five-second purge are unchanged. Each
fold's two-member model predicts and decodes the full halves independently before its held-out
public output is spliced. Only the all-public refit is used outside the public window.
The blend supplies probabilities to the existing fusion and confidence-pruning rules.
Inference consults no ground truth. No additional video extraction is required.

The earlier D mask and every frame without supported video remain unchanged. The prior
classifier and its configuration are retained locally for exact reproduction.
The blend changes 362 match frames, including 106 public frames. Full-match possession count
changes from 412 to 404; the dead share stays 46.13%.

## Sensitivity to time partitions

The original classifier is compared with the same fixed 1:3 blend and pruning settings:

| Time folds | Previous spell / macro | Blended spell / macro |
|---|---:|---:|
| 3 | 0.6737 / 0.9191 | 0.6947 / 0.9248 |
| 4 | 0.6421 / 0.9138 | 0.5895 / 0.9159 |
| 5, deployed | 0.7444 / 0.9292 | **0.7529 / 0.9353** |
| 6 | 0.7000 / 0.9232 | 0.6842 / 0.9292 |

MacroF1 improves in all four partitions. SpellF1 falls in the four- and six-fold checks;
the blend is a public-score candidate with an unstable spell gain across partitions.
All partitions use the same 17 public possessions. Feature/weight selection used this window,
so held-out classifier training does not remove selection bias. There is no unseen-match
validation of this fixture-trained video head and private performance is unknown.

## Other ideas checked

The cached screen also tested contact votes reset at inferred dead phases; logistic/boosted
classifier probability ensembles; short temporal smoothing; denser native/interpolated
training grids; alive-only team training; bounded dead-boundary movement; and feature ablations.
None of those alternatives improved both deployed metrics. Removing all discrete context
without blending reached 0.9431 macroF1 but 0.7294 spellF1. It was retained as the complementary
member, rather than deployed alone. No wider dead-boundary adjustment was promoted.

## Reproduce

Use the existing tracking, ownership/shape checkpoints and FILM cache described in
[VALDI_INTEGRATION.md](VALDI_INTEGRATION.md):

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/pre_blend.json --sets film --models logreg_blend \
  --weights 1 --lock-dead 1 --out pretrain_data/valdi_hybrid_blend --name hybrid_blend
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/blend_bench.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python -m unittest discover \
  -s src/bench -p 'test_*.py' -v
# Reproduce the previous default:
.venv/bin/python src/run.py --params src/configs/pre_blend.json --name before_blend
```

Checkpoints, cached match features and submissions stay gitignored. Aggregates and fingerprints
are recorded in `bench/results/blend_2026-11-15.json`. Contract tests cover held-out label
perturbations through both ensemble members, exclusion of hard context from the complementary
member, probability class alignment, and the existing D/video/fold boundaries.
All 28 tests pass. The official validator accepts 199,319 rows; the official scorer confirms
the reported metrics. Training-CLI and inference-CLI CSVs are byte-identical, and production
labels match the frozen-shape tuning path over the full match. The previous configuration
still reproduces its original CSV hash exactly.
