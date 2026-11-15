# Video-gated decoder adjustments, November 15

This records a808c06, retained in `configs/spell_best.json`. The subsequent macro-focused
configuration is documented in [MACRO_EXPERIMENTS.md](MACRO_EXPERIMENTS.md).

The same dense-video classifier now supplies limited flexibility to the final decoder.
Public macroF1 rises from 0.9346 to 0.9355 while spellF1 stays 0.7882. No classifier
retraining or further video extraction is needed. The previous configuration is frozen
in `configs/pre_loose.json`.

| Five purged time folds, same 17 public possessions | spellF1 | macroF1 |
|---|---:|---:|
| Previous default | 0.7882 | 0.9346 |
| Confident loose-state entries | 0.7882 | 0.9353 |
| Bounded D-edge adjustment alone | 0.7882 | 0.9348 |
| **Combined default** | **0.7882** | **0.9355** |

This is a small public-selected improvement affecting 21 public frames. It does not
resolve the missed brief possessions or the delayed early handover. Private performance
is unknown.

## Rules

`hybrid_loose_threshold=0.8` opens transitions into the opposing team's loose-ball state
when its held-out video classifier assigns that team more than 80% probability conditional
on alive play. This allows a last-touch change before sustained control. The existing base
team-switch cost, directional event bonus, emissions and duration rules still choose the
path. Missing video and unusable probabilities abstain. These arcs are absent from the
shape-training decodes and the contact-only feature baseline.

`hybrid_dead_margin=12` permits D/alive adjustments within 12 frames (0.48 seconds) of an
inferred phase boundary, restricted to supported video and one half. A fixed centre retains
at least min(original D length, `dead_min_frames`) dead frames. An alive phase retains a
centre frame, preventing two stoppages from merging across a brief restart. These fixed
centres also constrain independently decoded folds when their held-out outputs are spliced.
Frames outside eligible edges keep their original D/alive assignment.

Both rules run inside each held-out head's independent whole-half decode, before pruning
and public splicing. Neither rule reads ground truth during inference. Setting both options
to zero exactly reproduces the previous CSV; old configurations default to zero.

Across the full match 122 frames change, including 96 D/alive changes confined to eligible
edges. All frames without supported video are unchanged. Full-match possession count changes
from 404 to 407; dead share moves from 46.13% to 46.12%.

## Sensitivity checks

| Time folds | Before macroF1 | Combined macroF1 | spellF1 before and after |
|---|---:|---:|---:|
| 3 | 0.9247 | 0.9275 | 0.6737 |
| 4 | 0.9273 | 0.9276 | 0.7250 |
| 5, deployed | 0.9346 | 0.9355 | 0.7882 |
| 6 | 0.9390 | 0.9396 | 0.7333 |

These partitions reuse the same public labels. Feature and parameter selection used that
window, so these are correlated sensitivity checks, not independent validation matches.
Looser confidence thresholds 0.5/0.6 raised some frame scores but reduced deployed spellF1
to 0.7263. A 0.9 threshold reproduced the old deployed metrics. Wider D margins hurt macroF1.

The latest `origin/valdi` update, 1e82466, was also reviewed. Its 20 additional physics,
shape, crowding and event-bonus columns were tested on this stronger pipeline with logistic,
blended logistic, boosted and Extra Trees heads, plus probability ensembles. None beat the
chosen tradeoff; the 62-feature logistic blend reached 0.9379 macroF1 but 0.7059 spellF1.
The broader branch update was not merged over the current pipeline.

## Reproduce

With the existing checkpoints and `film_dense` feature cache:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/loose_bench.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python -m unittest discover \
  -s src/bench -p 'test_*.py' -v
.venv/bin/python src/run.py --params src/configs/pre_loose.json --name before_loose
```

All 33 tests pass, including sparse/reference parity with gated arcs, disabled-mode parity,
direct loose transitions, half/video and phase-centre constraints, and held-out-label
perturbations through the combined decoder. Production and frozen-shape tuning labels agree
over all 199,319 frames. The old configuration reproduces its exact original CSV hash.
The official format validator accepts the new file and the official scorer confirms its
public metrics. Aggregates and fingerprints are in `bench/results/decoder_2026-11-15.json`.
