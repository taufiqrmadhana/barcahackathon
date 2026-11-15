# Confidence-based possession pruning, November 15

This records the improvement at d0b6ec2, now frozen in `configs/pre_blend.json`.
The subsequent classifier blend is documented in [BLEND_EXPERIMENTS.md](BLEND_EXPERIMENTS.md).

The current tracking, contact and FILM pipeline sometimes emits a short possession even
when its held-out hybrid classifier favours the surrounding team. Merging these unsupported
islands improves both public metrics without retraining a classifier or changing video inputs.

| Public five minutes, same classifier and five held-out folds | spellF1 | macroF1 | possessions |
|---|---:|---:|---:|
| Previous default, `configs/pre_pruning.json` | 0.7053 | 0.9231 | 21 |
| **Default with confidence pruning** | **0.7444** | **0.9292** | 19 |

The ground truth has 17 possessions. Transition F1 within one second rises from 0.5909
to 0.6190. The change affects 148 frames over the full match (54 public, 94 outside the
public window); every D frame and every frame without supported video stays unchanged.
These are public-window improvements; corrected private labels are unavailable.

## Rule

`hybrid_prune_max_frames=100` and `hybrid_prune_min_confidence=0.5` enable the rule.
For an H/A/H or A/H/A island of at most four seconds:

1. Require the whole island to have video coverage and both neighbouring spells to have
   the same alive team, inside the same half.
2. Compute the classifier's mean confidence for the island's team, **conditional on alive
   play**: `P(team) / (P(H) + P(A))`.
3. Merge into the neighbours only if that confidence is below 0.5: the classifier prefers
   the other team. A short but confident possession is retained.

D, X, half edges, missing video, non-finite probabilities and a lack of H/A probability
are left alone. Tied votes retain the prediction. The function runs inside each fold's
independent whole-half decode, before held-out predictions are spliced. It never consults
ground truth at inference or borrows confidence from another fold's classifier.

The rule uses the same four-second scale as the existing external benchmark's short-spell category.
This differs from simply raising the minimum duration: the correctly found sub-second
public possession remains present. The old behaviour is recovered exactly by setting
`hybrid_prune_max_frames=0` or using the frozen prior configuration.

## Paired fold audit

The same logistic classifier specification, features, 1 Hz training grid and five-second
purge were retained. Three-, four- and six-fold heads were refitted only to vary the temporal
partition; the original five-fold checkpoint was reused unchanged.

| Time folds | Before spell / macro | After spell / macro |
|---|---:|---:|
| 3 | 0.6300 / 0.9142 | 0.6737 / 0.9191 |
| 4 | 0.6421 / 0.9138 | 0.6421 / 0.9138 |
| 5, deployed | 0.7053 / 0.9231 | **0.7444 / 0.9292** |
| 6 | 0.6632 / 0.9208 | 0.7000 / 0.9232 |

The paired change is positive or neutral in these checks. These are correlated views of
the **same 17 possessions**, not four independent validation sets. Model/feature/parameter
selection still used the public window; the audit does not establish private generalisation.
The default remains five folds, as before this experiment.
The 5–95% spellF1 bootstrap intervals overlap: 0.568–0.821 before and 0.600–0.867 after;
these intervals do not account for parameter selection.

A one-second cap also improves the deployed score (0.7444 / 0.9255). Four seconds retains
that spellF1 and improves macroF1, with a positive or neutral paired change across fold counts.
More aggressive confidence thresholds (0.55/0.60) under the four-second cap raise macroF1 to
0.9364 but reduce spellF1 to 0.7412; they were not promoted.

## Other ideas tested

Cached explorations compared 48 combinations of the original FILM features, original-ATD
and fused-track ownership/control features, +/-1 second context, home/away augmentation,
fusion weights and locked/unlocked D. Home/away augmentation was fitted after the fold purge,
with half-weight duplicated rows and averaged original/swapped predictions. None improved both
deployed metrics. Flattening control/loose emissions and adding switch bonuses from the video
probability derivative also failed the paired baseline comparison. These ideas were rejected.

This is evidence against those implementations on this window, not proof that the ideas
cannot help with more training matches or denser video. Missed short possessions and a delayed
early handover remain unresolved.

## Reproduce

With the existing classifier, ATD and video cache from `VALDI_INTEGRATION.md`:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/run.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/barca.csv
.venv/bin/python evaluation/run_evaluation.py --no-detail
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/prune_bench.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python -m unittest discover \
  -s src/bench -p 'test_*.py' -v
# Exact old behaviour:
.venv/bin/python src/run.py --params src/configs/pre_pruning.json --name before_pruning
```

The model checkpoint and feature cache are unchanged. The new tests cover weak/confident
short spells, home/away symmetry, conditional-alive confidence, D/half/video boundaries,
unusable probabilities and held-out-label perturbations with pruning enabled. Production and
the frozen-shape tuning path are compared over all 199,319 frames. Aggregate results and
fingerprints are in `bench/results/pruning_2026-11-15.json`. **Private performance is unknown.**
