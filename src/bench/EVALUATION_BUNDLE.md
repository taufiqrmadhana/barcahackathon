# Separate evaluation tables and learning curve

This package is separate from the four-page IEEE report. It uses the frozen
macro-v10 checkpoint, without changing the submission. Public evaluation has
**17 GT possessions and 14 predictions**, all matched, with three GT spells
missed and zero spurious predictions. The 401 full-match possessions are
predicted counts and do not enter the public score.

From the repository root:

```bash
# Requires the match assets and the existing v10 model/features.
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/phase_learning_curve.py

# Export the frozen aggregates; no match assets are needed for this step.
.venv-cv/bin/python src/bench/export_evaluation.py
```

The learning-curve report is frozen in
`results/macro_v10_learning_curve_2026-11-17.json`. The exporter reads that
committed report, together with the v9/v10 aggregate reports. A fresh computed
curve goes to `cache/phase_learning_curve/results.json` for comparison before
replacing the frozen report.

Outputs are under `src/submission/build/evaluation/`:

- `evaluation_tables.pdf`: two pages of evaluation and time-fold tables.
- `evaluation_summary.png`, `kfold_results.png`: shareable table images.
- `learning_curve.png`, `.pdf`: raster and vector plots.
- Seven CSVs: overall metrics, class metrics, evaluation scope, K sensitivity,
  individual five-fold metrics, learning-curve values and frame-error counts.
- `learning_curve_details.json`: full aggregate protocol, per-fold losses,
  source/model fingerprints and sample counts.
- `README.md`: explanations, including the difference between 17 public GT
  spells, 14 public predictions and 401 full-match predictions.

Everything is also in `src/submission/build/Borrow_Khan_Evaluation.zip`.
Generated files stay gitignored; the source and aggregates regenerate them.

## What the learning curve measures

The variable is the amount of training data for the **18-feature broadcast
phase logistic classifier**, at fixed `C=0.3`. It is not an epoch history.
The other five members and the decoder remain fixed. Each of five contiguous
time folds excludes its own public interval and a 125-frame purge. Subsets at
25/50/75/100% are nested and class-stratified within eligible training rows,
using a fixed seed. Imputation and scaling are fitted on each subset only.

The full pool contains 301 native 1 Hz rows. At 100%, each held-out model has
231--236 eligible training rows (mean 232.8), not all 301. The left panel shows
binary alive/dead training and held-out log loss on native rows. Shading is
population SD over five time folds, not a confidence interval. The right panel
shows full-pipeline H/A/D macroF1 and spellF1 on pooled public predictions.
The two panels measure different tasks. The 100% output reproduces every
frozen checkpoint label exactly. Validation loss is not monotonic; 75% and
100% produce equal public pipeline scores in this single subset experiment.
This does not establish convergence or independent generalisation.

## Read the two K-fold tables correctly

`kfold_sensitivity.csv` compares complete held-out outputs with K=3/4/5/6,
each over the same 7,500 public frames and 17 GT spells. Those partitions are
sensitivity checks, not independent matches.

`five_fold_breakdown.csv` describes the five individual 1,500-frame intervals
of the deployed model. Its macroF1 always averages all three H/A/D class F1
values, assigning zero to an absent class with zero F1. Folds 1 and 3 contain
no GT D; fold 4 has just 33 GT H frames. Read scores alongside class support.
The official pooled macroF1 is **0.963167**, computed from all held-out
predictions together; it is not the arithmetic mean of the fold F1 scores.
