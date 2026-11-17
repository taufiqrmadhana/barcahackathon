# Phase weighting: public macroF1 0.962321, spellF1 0.903226

Macro v9 changes only [v8's](MACRO_V8.md) phase-branch broadcast weight from
50% to 75%. Classifiers, features, five time folds, the 50-frame phase limit and
pointwise retention of primary teams are unchanged. V8 remains available in
`configs/macro_v8.json`.

| Public metric | v8 | v9 |
|---|---:|---:|
| MacroF1 | 0.962146 | **0.962321** |
| SpellF1 | 0.903226 | 0.903226 |
| Home F1 | 0.961832 | 0.961832 |
| Away F1 | 0.968141 | 0.967393 |
| Dead F1 | 0.956464 | 0.957737 |

The headline gain is tiny. Four public frames change from false D to false A:
their alive/dead phase becomes correct, but their team remains wrong. Exact frame
accuracy stays 0.962933; no correct public frame regresses. This is a macroF1
tradeoff between classes, not a reduction in the number of frame errors.

## Paired sensitivity

| Time folds | v8 macroF1 | v9 macroF1 | v8 spellF1 | v9 spellF1 |
|---|---:|---:|---:|---:|
| 3 | 0.943440 | 0.950779 | 0.729412 | 0.700000 |
| 4 | 0.946871 | 0.947047 | 0.775758 | 0.775758 |
| 5, deployed | 0.962146 | **0.962321** | 0.903226 | 0.903226 |
| 6 | 0.949453 | 0.949979 | 0.825000 | 0.825000 |

MacroF1 rises in all four partitions. Its paired mean increases from 0.950478
to 0.952531. Three-fold spellF1 falls by 0.029412, with 19 predictions versus
v8's 17; spellF1 stays equal in the other partitions. The default prioritises
macroF1 as requested, while retaining the previous model as an alternative.

All choices reuse the same 7,500 public frames and 17 GT possessions. These
partitions are sensitivity checks, not independent test sets. Method-selection
bias remains, and private/unseen-match performance is unknown. The deployed
public output still matches all 14 predictions, with three GT spells missed
and no spurious prediction at every official IoU threshold. Transition F1
within one second remains 0.810811.

## Rejected follow-ups

A frozen ownership model trained on Metrica games 1--3 was evaluated on aligned,
cleaned broadcast tracking without reading possession GT. It supports 5,152
public frames, with confidence above 0.9 on 2,824. High-confidence corrections,
region votes and another D-locked team decode did not improve v8. Unrestricted
mixing and contact overrides regressed. The diagnostic can be regenerated with:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/broadcast_owner_probe.py
```

Changing phase regularisation, extending phase edges, and replacing team-specific
features with symmetric min/max features also failed to beat the selected profile.
An all-public refit has training-window macroF1 0.965302 but spellF1 0.786207;
that is not held-out validation and was not selected for submission.

## Predicted match statistics

| Window | Possessions | Mean duration | Home share of alive time | Dead share |
|---|---:|---:|---:|---:|
| First half | 202 | 8.82 s | 61.61% | 43.93% |
| Second half | 199 | 8.33 s | 58.70% | 47.78% |
| Both halves | **401** | **8.58 s** | **60.20%** | **45.85%** |

Home has 197 spells averaging 10.51 s; away has 204 averaging 6.71 s and
39.80% of alive time. Counts are H 51,786 / A 34,233 / D 72,838 / X 40,462.
There are 130 D intervals by half. Compared with v8, 84 full-match labels change,
all D/alive changes. These statistics describe predictions, without full-match GT scoring.

## Reproduction and validation

Use v8's ensemble construction command with `--phase-weight .75`,
`--out pretrain_data/macro_v9/ensemble.joblib --name macro_v9`. Member models
and caches are unchanged. Then run:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/macro_bench.py \
  --baseline src/configs/macro_v8.json --params src/configs/macro_v9.json \
  --out cache/improve_nov17_owner/v9_production_fold_audit.json
```

Prototype, builder, production and frozen-shape tuner agree on all 199,319 frames.
The earlier v8 CSV still reproduces exactly. Primary coverage and fixed D-core
bounds remain protected. The production implementation is unchanged from the
55-test v8 checkpoint; the new ownership diagnostic reproduces all probabilities
and support frames exactly. The official validator and public scorer confirm v9.
Aggregate results and fingerprints are in `bench/results/macro_v9_2026-11-17.json`.

Recommended local output `evaluation/results/Borrow_Khan.csv` is identical to
`barca.csv`. SHA-256:
`7821dfa01fed05b4506ff3de101e49a8568209cc5b457fcccc1d79adbbdcd019`.
