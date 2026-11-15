# Conditional ownership expert and a stronger spell alternative

The macro-focused default adds a player-motion expert to macro v2. The public
five-fold score changes from **0.949098** to **0.949550 macroF1**, with spellF1
unchanged at **0.752941**. This is a small gain on reused public labels, not evidence
of a private-score or unseen-match improvement.

| Frozen profile | macroF1 | spellF1 |
|---|---:|---:|
| Earlier spell profile, `spell_best.json` | 0.935513 | 0.788235 |
| Macro v2, `macro_v2.json` | 0.949098 | 0.752941 |
| New default, `macro_v3.json` | **0.949550** | 0.752941 |
| New spell alternative, `balanced_v1.json` | 0.942279 | **0.800000** |

## Features and routing

The new ordinary logistic head uses the same 34 continuous/video features as macro
v1 plus six existing player-only shape features: each team's change in pressure
distance and speed over 75 frames, and each team's width smoothed over 25 frames.
It uses `film_dense`, C=1, and the same 301 public 1 Hz training rows, five time folds
and 125-frame purge. It needs no new video extraction. These columns are requested
only by checkpoints that use them; earlier models retain their exact feature paths.

The primary remains 0.75 times the dense-video head plus 0.25 times the 5 Hz head.
The motion expert has zero ordinary blend weight. It supplies conditional H/A
probabilities only when the primary's larger conditional probability is below 0.8
and the expert's is above 0.9. The primary's P(D) is preserved. Missing expert
coverage or invalid probabilities abstain, without reducing primary coverage.
Existing contact decoding, 12-frame dead-edge bounds and confidence pruning follow.

Checkpoint schema 3 distinguishes routing from ordinary schema-2 averaging. Every
member must have identical fold edges, purge, training stride, physical parameters
and tracking-ball provenance. The expert uses its corresponding held-out head inside
each independent whole-half decode. The all-public refits apply outside the public
window only. Tests perturb a held-out segment's labels and verify that its routed
predictions cannot change.

The spell alternative is an ordinary equal-weight blend of the dense C=1 and motion
C=1 heads, using macro v2's decoder settings. It reaches 0.8 spellF1 with 15 public
possessions, versus 17 in the ground truth. Earlier profiles remain available.

## Sensitivity

All rows use the same 17 public possessions, repartitioned in time.

| Time folds | v2 macroF1 | v3 macroF1 | v3 spellF1 | Spell alternative macroF1 / spellF1 |
|---|---:|---:|---:|---:|
| 3 | 0.924945 | 0.931539 | 0.722222 | 0.934930 / 0.600000 |
| 4 | 0.933586 | 0.930173 | 0.673684 | 0.925332 / 0.712500 |
| 5, deployed | 0.949098 | 0.949550 | 0.752941 | 0.942279 / 0.800000 |
| 6 | 0.939074 | 0.936910 | 0.722222 | 0.945559 / 0.752941 |

The gains are inconsistent across partitions. The five-fold profiles are selected
on public performance; classifier cross-fitting does not remove that selection bias.
No independent private or unseen-match labels are available.

Other trials failed to improve macro v2: camera-compensated foot/body motion, naive
interpolation residuals, binary dead-ball heads, weak labels outside the public
window, contact recalibration, and C=0.3/3 versions of the motion expert. Their
aggregate scores are retained in the report, along with the unsuccessful
[Kevinza-Valdi adaptations](KEVINZA_VALDI_AUDIT.md).

## Reproduce

Start with macro v1/v2's existing checkpoints and video feature stores:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_hybrid.py \
  --params src/configs/macro_v2.json --sets film_motion --models logreg \
  --interp film_dense --logreg-c 1 --objective macro --weights .5 --lock-dead 1 \
  --out pretrain_data/macro_motion --name macro_motion
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_view5hz/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
  --weights .75 .25 0 --team-expert 2 --params src/configs/macro_v2.json \
  --out pretrain_data/macro_v3/ensemble.joblib --name macro_v3
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run_ensemble.py \
  --models pretrain_data/macro_v1/film_soft_logreg.joblib \
           pretrain_data/macro_motion/film_motion_logreg.joblib \
  --weights .5 .5 --params src/configs/macro_v2.json \
  --out pretrain_data/balanced_v1/ensemble.joblib --name balanced_v1
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v2.json --params src/configs/macro_v3.json
```

The production, builder and frozen-shape tuner are checked for exact full-match
agreement. The earlier macro-v2 CSV is reproduced byte for byte. All 37 tests pass,
including missing expert coverage, class exchange, dead probability preservation
and held-out-label isolation. The new default changes 12 match frames, including
four public frames; its D labels and full-match count of 404 possessions are unchanged.
Aggregate scores and fingerprints are in
`bench/results/macro_v3_2026-11-15.json`. Checkpoints, features, images and CSVs stay
gitignored. Macro-focused experimentation continues.
