# Macro-focused classifier, November 15

This records b441107 and `configs/macro_v1.json`. The subsequent two-view model is described
in [MACRO_V2.md](MACRO_V2.md).

The user changed the selection priority to macroF1. The default now uses a single logistic
classifier with continuous context and video features, fitted with C=1 and fusion weight 0.5.
The earlier spell-focused checkpoint is preserved separately.

| Same public five minutes, five purged time folds | macroF1 | spellF1 |
|---|---:|---:|
| Spell-focused alternative, `configs/spell_best.json` | 0.9355 | **0.7882** |
| **Macro v1, `configs/macro_v1.json`** | **0.9473** | 0.7647 |

The macro gain is 0.0118. H F1 rises from 0.9303 to 0.9458 and A F1 from 0.9342 to
0.9553; D F1 falls from 0.9421 to 0.9408. Transition F1 within one second rises from
0.6500 to 0.7000. Both predictions contain 17 public possessions.

## Change

`film_soft` is the existing 42-feature FILM set with the eight discrete decoder-label and
duration columns removed. It retains continuous scores, geometry, camera/context, external
shape probabilities and FILM evidence. The classifier uses the same 301 native 1 Hz rows,
five 60-second time folds and five-second purge. Each held-out head independently predicts
and decodes the full halves; the full public refit supplies only non-public CSV rows.

The logistic regularisation is exposed as `--logreg-c` and stored in the checkpoint.
`--objective macro` ranks fitting candidates by macroF1, with spellF1 as a tie breaker.
The default CLI objective remains `spell` for compatibility with earlier recipes. No official
metric implementation or ground truth was changed.

The deployed head uses C=1, weight 0.5, confidence pruning and the existing 12-frame bounded
D-edge adjustment. Loose-state transitions are disabled for this head: its confidence is
different from the former two-member classifier. Inference reads no corrected private labels.
No new video extraction was needed for this checkpoint; it uses the existing `film_dense` cache.

## Checks and limits

| Time folds | Spell-profile macroF1 | Macro-profile macroF1 | Macro-profile spellF1 |
|---|---:|---:|---:|
| 3 | 0.9275 | 0.9272 | 0.6842 |
| 4 | 0.9276 | 0.9277 | 0.6889 |
| 5, deployed | 0.9355 | 0.9473 | 0.7647 |
| 6 | 0.9396 | 0.9410 | 0.7444 |

The macro gain varies with the partition and is slightly negative with three folds. All
partitions and choices reuse the same 17 public possessions. These are sensitivity checks,
not independent validation sets; selection bias remains. Private performance is unknown.

The screen compared C=0.3/1/3, original/complementary feature sets, sparse/dense video, fusion
strengths, decoder switches and class balancing. Forest, Extra Trees and boosted prototypes
also failed to beat the selected macro result. C=3 with weight 0.5 reached essentially the
same macroF1, with lower spellF1. All inspected choices are recorded as public-selected trials.

All 33 contract tests pass, including held-out label perturbations with non-default C.
Production, the fitting CLI and the frozen-shape tuning path produce identical labels across
all 199,319 frames. The official validator and scorer confirm the file and metrics. The prior
spell checkpoint still reproduces its original CSV hash. There are 275 changed match frames,
including 157 public frames; unsupported video remains unchanged. Full-match possession count
changes from 407 to 404, with 46.11% dead frames.

## Reproduce

Use the existing inputs, external models and `film_dense` cache:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/spell_best.json --sets film_soft --models logreg --interp film_dense \
  --logreg-c 1 --objective macro --weights 0.5 --lock-dead 1 \
  --set hybrid_loose_threshold=0 hybrid_dead_margin=12 \
  --out pretrain_data/macro_v1 --name macro_v1
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py
# Retained spell-focused profile:
.venv/bin/python src/run.py --params src/configs/spell_best.json --name spell_best
```

Checkpoints, features and submissions stay gitignored. Aggregates and fingerprints are in
`bench/results/macro_v1_2026-11-15.json`. Work continues with macroF1 as the primary objective.
