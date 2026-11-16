# Dead-head regularisation: public macroF1 0.957604, spellF1 0.875000

Macro v7 keeps [v6's architecture](MACRO_V6.md) and changes only the binary
dead-ball logistic head's C from 1 to 10. The 50:50 P(D) blend, 28 features,
301 native training rows, purged folds and temporal operations are unchanged.
V6 remains available in `configs/macro_v6.json`.

| Public metric | v6 | v7 |
|---|---:|---:|
| MacroF1 | 0.957088 | **0.957604** |
| SpellF1 | 0.875000 | 0.875000 |
| Home F1 | 0.958455 | 0.958455 |
| Away F1 | 0.968291 | 0.968889 |
| Dead F1 | 0.944519 | 0.945467 |

Only three public frames change, all correcting errors; eight frames change
across the match. This is a very small gain on reused public labels. It does
not establish better private or unseen-match performance.

## Paired time partitions

| Folds | v6 macroF1 | v7 macroF1 | SpellF1, both |
|---|---:|---:|---:|
| 3 | 0.941007 | 0.940817 | 0.729412 |
| 4 | 0.942031 | 0.942151 | 0.752941 |
| 5, deployed | 0.957088 | **0.957604** | 0.875000 |
| 6 | 0.951251 | 0.951251 | 0.825000 |

Three folds regress by 0.000190; four and five improve, and six stay equal.
All partitions contain the same 17 public possessions and are sensitivity
checks, not independent validation. Method and hyperparameter selection both
reuse those labels. The public output still contains 15 predicted possessions:
14 matched, one spurious and three GT possessions missed at every official
IoU threshold. One-second transition F1 remains 0.736842.

## Predicted match statistics

| Window | Possessions | Mean duration | Home share of alive time | Dead share |
|---|---:|---:|---:|---:|
| First half | 203 | 8.75 s | 61.76% | 44.10% |
| Second half | 198 | 8.34 s | 58.86% | 48.00% |
| Both halves | **401** | **8.55 s** | **60.36%** | **46.05%** |

Home has 198 spells averaging 10.45 s; away has 203 averaging 6.69 s and
39.64% of alive time. Label counts are H 51,735 / A 33,975 / D 73,147 /
X 40,462. There are 130 D intervals counted separately by half. These are
predicted summaries over 158,857 graded frames, without full-match GT scoring.

## Reproduction

The ordinary specialist can be refitted with the existing caches:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v6.json --interp film_dense \
  --sets film_dead_motion --models dead_logreg --logreg-c 10 \
  --weights .5 --lock-dead 1 --objective macro \
  --out pretrain_data/macro_dead_video_c10 --name macro_dead_video_c10
```

Use v6's ensemble construction command with this specialist as its fifth
member and `--out pretrain_data/macro_v7/ensemble.joblib --name macro_v7`.
For inference and the paired audit:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py \
  --params src/configs/macro_v7.json --name Borrow_Khan
.venv/bin/python evaluation/validate_submission.py evaluation/results/Borrow_Khan.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v6.json --params src/configs/macro_v7.json \
  --out cache/improve_nov17/v7_production_fold_audit.json
```

CSV SHA-256:
`e09c0524b08d88b8c198c00321bd63a90dd727331383350aec31d156dc86055c`.
Checkpoints, feature caches and submission labels remain gitignored.

All six specialist models reproduce the prototype coefficients exactly.
Prototype, builder, production and frozen-shape tuner match every frame, and
v6 still reproduces its old CSV. Uncovered frames and the existing D-edge
bounds are preserved. The official validator and public scorer confirm the
checkpoint. The production code is unchanged from the v6 implementation with
51 passing contract tests. Aggregate trials, paired audits, validation proofs
and hashes are in `bench/results/macro_v7_2026-11-17.json`.
