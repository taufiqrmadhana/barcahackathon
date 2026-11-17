"""Measure a data-size learning curve for the frozen broadcast phase classifier.

Only the phase classifier is refitted. The other five ensemble members and all
decoding rules stay fixed. Training subsets are nested and stratified within
each purged time fold. These descriptive curves reuse the public match slice;
they are not an independent test or an epoch/convergence history.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broadcast_followup_bench import Probe
from possession import broadcast, hybrid, pipeline, validator
from possession.paths import ROOT


def summarise(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=Path(ROOT, 'src/configs/macro_v10.json'))
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/phase_learning_curve/results.json'))
    args = parser.parse_args()
    probe = Probe(args.params, 5)
    phase_index = probe.a['phase_router']['expert']
    original = joblib.load(probe.p.hybrid_model)['members'][phase_index]
    if (original['model_name'] not in ('dead_logreg','dead_context_blend') or not original.get('strict_fold_edges')
            or 'broadcast_available' not in original['names']):
        raise ValueError('expected the availability-aware binary phase classifier')
    columns, coverage, frames, _ = probe.views[phase_index]
    native = np.intersect1d(frames, np.r_[np.arange(original['public_start'],
        original['public_stop'], original['train_stride']), original['public_stop']-1])
    native = native[coverage[native]]
    if len(native) != original['training_rows']:
        raise ValueError('native training pool differs from the fitted checkpoint')
    matrix = validator.matrix(columns, original['names'], native)
    target = probe.gt[native]
    binary = (target == 2).astype(np.int8)
    assignment = np.clip(np.searchsorted(original['edges'], native, side='right')-1, 0, 4)
    reference = pipeline.score(probe.reference, probe.gt)
    per_fold = []
    for fold in range(5):
        held = probe.public[probe.held == fold]
        truth = probe.gt[held]
        predicted = np.array([{'H': 0, 'A': 1, 'D': 2, 'X': 3}[x] for x in probe.reference[held]])
        per_fold.append(dict(fold=fold+1, frames=len(held),
            accuracy=float(accuracy_score(truth, predicted)),
            macroF1_3classes=float(f1_score(truth, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
            class_f1=f1_score(truth, predicted, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            gt_class_counts=np.bincount(truth, minlength=3)[:3].tolist()))
    seed, fractions = 20261117, (.25, .5, .75, 1.)
    report = dict(date='2026-11-17', model=('broadcast phase logistic blend'
        if original['model_name']=='dead_context_blend' else 'broadcast phase logistic regression'),
        model_name=original['model_name'], base_classifier_count=2 if original['model_name']=='dead_context_blend' else 1,
        feature_count=len(original['names']), C=original['logreg_c'],
        target='binary alive versus dead; D is the positive class',
        x_axis='fraction of each fold\'s eligible training rows, not epochs',
        loss='binary log loss on native 1 Hz rows; lower is better',
        variability='population standard deviation across the five fixed time folds, not a confidence interval',
        fractions=list(fractions), seed=seed, folds=5, purge_frames=original['purge_frames'],
        strict_fold_edges=True, native_pool=len(native),
        subset_policy='one nested, class-stratified permutation per eligible training fold',
        pipeline_policy='refit only phase heads; other members and decoder frozen; full public refit outside public unchanged',
        public_frames=len(probe.public), public_gt_spells=reference['n_gt_spells'],
        baseline=dict(macroF1=reference['macro_f1'], spellF1=reference['spell_f1_mean'],
                      accuracy=reference['accuracy'], predicted_spells=reference['n_pred_spells']),
        params_sha256=broadcast.digest(args.params), model_sha256=broadcast.digest(probe.p.hybrid_model),
        source_sha256=broadcast.digest(__file__),
        protocol_sha256=broadcast.digest(Path(__file__).with_name('broadcast_followup_bench.py')),
        per_fold_final=per_fold, results=[],
        same_public_possessions=True, private_score_known=False, used_to_change_submission=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for fraction in fractions:
        models, folds = [original['full']], []
        for fold in range(5):
            train = ((native < original['edges'][fold]-original['purge_frames'])
                     | (native >= original['edges'][fold+1]+original['purge_frames']))
            held = assignment == fold
            rng = np.random.default_rng(seed+fold)
            selected = []
            for label in (0, 1):
                rows = np.flatnonzero(train & (binary == label))
                rows = rng.permutation(rows)
                selected.extend(rows[:max(1, int(np.ceil(fraction*len(rows))))])
            selected = np.sort(np.asarray(selected, dtype=np.int64))
            assert train[selected].all() and not held[selected].any()
            if fraction == 1:
                np.testing.assert_array_equal(selected, np.flatnonzero(train))
                model = original['heads'][fold]
            else:
                model = hybrid.factory(original['model_name'], names=original['names'],
                                        logreg_c=original['logreg_c']).fit(matrix[selected], target[selected])
            models.append(model)
            train_pd = validator._proba3(model, matrix[selected])[:, 2]
            valid_pd = validator._proba3(model, matrix[held])[:, 2]
            folds.append(dict(fold=fold+1, eligible_rows=int(train.sum()),
                train_rows=len(selected), validation_rows=int(held.sum()),
                train_dead_rows=int(binary[selected].sum()), validation_dead_rows=int(binary[held].sum()),
                train_log_loss=float(log_loss(binary[selected], train_pd, labels=[0, 1])),
                validation_log_loss=float(log_loss(binary[held], valid_pd, labels=[0, 1]))))
        probabilities = probe.probabilities(models, columns, original['names'], coverage)
        labels = probe.render(probabilities, coverage, 'phase')
        if fraction == 1:
            np.testing.assert_array_equal(labels, probe.reference)
        score = pipeline.score(labels, probe.gt)
        row = dict(fraction=fraction, train_rows=summarise([f['train_rows'] for f in folds]),
            train_log_loss=summarise([f['train_log_loss'] for f in folds]),
            validation_log_loss=summarise([f['validation_log_loss'] for f in folds]),
            pipeline_macroF1=score['macro_f1'], pipeline_spellF1=score['spell_f1_mean'],
            pipeline_accuracy=score['accuracy'], pipeline_predicted_spells=score['n_pred_spells'],
            per_fold=folds)
        report['results'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'per_fold'}), flush=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
