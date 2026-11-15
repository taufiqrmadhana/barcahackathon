# Two video cadences and separate dead-ball calibration

The macro-focused default combines two held-out logistic heads and calibrates ownership
and dead-ball evidence separately. On the same public five minutes:

| Configuration | macroF1 | spellF1 |
|---|---:|---:|
| Retained spell profile | 0.9355 | **0.7882** |
| Macro v1 | 0.9473 | 0.7647 |
| **Macro v2, current checkpoint** | **0.9491** | 0.7529 |

The macro gain over v1 is 0.0018, with a 0.0118 spellF1 decrease. H/A/D class F1 values are
0.9464 / 0.9589 / 0.9420. Transition F1 within one second remains 0.7000, with 17 predicted
public possessions. Older profiles are retained in `configs/`.

## Model and extraction

The first member uses `film_dense` features and C=1; the second uses `film_5hz` and C=3.
Both use the 34-feature `film_soft` set and the same 301 native 1 Hz training rows. Their
probabilities are averaged with weights 0.75 / 0.25 before fusion and decoding. Both the
all-public refits and each corresponding held-out head are combined in that order.

The new frame plan `cv/frames/valdi_film_5hz.txt` contains 2,221 frame indices. It preserves
all 1,739 earlier observations and adds 482 samples to complete the public 5 Hz grid.
The added crops produced finite mean residuals and all earlier sample values are exactly
preserved. After seeding the separate cache from `film_dense`, the incremental CPU pass took
646 seconds. A fresh cache needs the complete plan; inference requires only the cached values.
The original cache remains unchanged. More sampling alone did not win: the 5 Hz member alone
reached only 0.9465 macroF1 and 0.6375 spellF1.

The probability factorisation is P(team)=P(alive)*P(team|alive). Team evidence retains weight
0.5; alive/dead evidence uses `hybrid_dead_weight=2`. The shared-weight default (-1) follows
the old arithmetic exactly. The existing 12-frame edge bound and fixed phase centres remain.
Wider confidence-gated edge adjustments and class-logit bias screens failed to offer a
meaningful improvement and were not promoted.

Checkpoint schema 2 stores ordinary schema-1 members. Construction rejects mismatched public
windows, fold edges, purge/stride settings, physical baselines and ball-tracking hashes.
Each public segment is independently decoded using only the corresponding held-out head from
every member. All those heads excluded that segment's labels. Coverage is the intersection of
active views; unsupported frames retain the physical baseline. Zero-weight views abstain.

## Sensitivity and limits

| Time folds | v1 macroF1 | v2 macroF1 | v2 spellF1 |
|---|---:|---:|---:|
| 3 | 0.9272 | 0.9250 | 0.6842 |
| 4 | 0.9277 | 0.9336 | 0.7053 |
| 5, deployed | 0.9473 | 0.9491 | 0.7529 |
| 6 | 0.9410 | 0.9391 | 0.7444 |

The gain is not consistent across partitions. All choices reuse the same 17 public possessions,
so held-out classifier training does not remove model-selection bias. No independent private
or unseen-match validation of this fixture-trained video layer is available.

All 35 tests pass, covering equal-weight compatibility, independent team/dead odds, ensemble
fold alignment, zero-weight views, common coverage and held-out-label perturbations. Fitting,
inference and frozen-shape tuning labels agree over all 199,319 frames. Macro v1 still reproduces
its exact previous CSV. The change affects 88 match frames, including 31 public frames;
unsupported video remains unchanged. Full-match count stays 404 possessions, with 46.08% D.

## Reproduce

Use the existing physical inputs and macro-v1 member. Extract the new cache, fit the second
member, then combine them:

```bash
TFHUB_CACHE_DIR=cache/film_model TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 \
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-v3/bin/python src/run_v3.py extract \
  --frames src/cv/frames/valdi_film_5hz.txt --cache-kind film_5hz --chunk 8
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v1.json --sets film_soft --models logreg --interp film_5hz \
  --logreg-c 3 --objective macro --weights 0.5 --lock-dead 1 \
  --out pretrain_data/macro_view5hz --name macro_view5hz
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
  --weights 0.75 0.25 --params src/configs/macro_v1.json --set hybrid_dead_weight=2 \
  --out pretrain_data/macro_v2/ensemble.joblib --name macro_v2
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v1.json --params src/configs/macro_v2.json
```

The official validator and scorer confirm the submission. Checkpoints, cached images/features
and CSVs remain gitignored. Aggregate trials and fingerprints are in
`bench/results/macro_v2_2026-11-15.json`. Macro-focused experimentation continues.
