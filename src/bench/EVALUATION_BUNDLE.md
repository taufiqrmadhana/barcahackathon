# Evaluation tables and learning curve

The public evaluation covers 7,500 frames: 17 GT possessions and 14 predictions,
all matched, with three missed GT spells and no spurious predictions. The 400
full-match possessions are predictions, not additional evaluated GT.

## Export

```bash
.venv-cv/bin/python src/bench/export_evaluation.py
```

Matplotlib is required. This reads the three aggregate reports in `src/bench/results/`
and needs no match data or fitted models.

Outputs in `src/submission/build/evaluation/`:

- `evaluation_tables.pdf`: public metrics, errors and time-fold results.
- `evaluation_summary.png`, `kfold_results.png`: table images.
- `learning_curve.png`, `learning_curve.pdf`: measured learning curve.
- Seven CSVs containing the plotted and tabulated values.
- `learning_curve_details.json`: sample counts, losses and protocol.

The full package is `src/submission/build/Borrow_Khan_Evaluation.zip`.

## Learning curve

To recompute it with the match assets and fitted ensemble:

```bash
.venv/bin/python src/bench/phase_learning_curve.py --params src/configs/macro_v11.json
```

Only the two broadcast phase heads are refitted. Other ensemble members and the
decoder remain fixed. Training subsets of 25/50/75/100% are nested and stratified
within each purged time fold. The 301-row native pool leaves 231–236 eligible
training rows per fold at 100% (mean 232.8).

The left panel shows binary alive/dead train and held-out log loss; the right
shows pooled H/A/D macro-F1 and spell-F1 for the complete pipeline. Shading is
fold standard deviation, not a confidence interval. This is a training-data-size
curve, not an epoch history. Output labels remain H/A/D/X.

## Time folds

`kfold_sensitivity.csv` compares K=3/4/5/6 on the same public interval.
`five_fold_breakdown.csv` describes the deployed five individual folds, each
with 1,500 frames. Its macro-F1 includes all three H/A/D classes, assigning zero
to an absent class with zero F1. Read it alongside class support: folds 1 and 3
contain no GT D, and fold 4 has only 33 GT H frames.

The official macro-F1 of 0.963966 pools all held-out predictions. It is not the
arithmetic mean of the per-fold F1 scores. All repartitions reuse the same public
possessions; private and unseen-match performance remain unknown.
