"""Screen availability context and regularisation against frozen macro v10.

These experiments reuse public labels for selection. All phase classifiers fit
training-only preprocessing under the original purged time folds; complete
half decodes precede stitching. Submitted profiles are never changed here.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broadcast_followup_bench import Probe
from possession import broadcast, clean, hybrid, pipeline, validator
from possession.dead_head import ContextDeadBallClassifier, DeadBallClassifier
from possession.paths import ROOT


def context_features(columns, available, blocks):
    result = dict(columns)
    n = len(available)
    for key, function in [('since', clean.since_last), ('until', clean.until_next)]:
        value = np.full(n, np.nan)
        for _, first, last in blocks:
            value[first:last+1] = np.minimum(function(available[first:last+1]), 125.)/25
        result['broadcast_available_'+key] = value
    for width in (25, 125):
        value = np.full(n, np.nan)
        for _, first, last in blocks:
            value[first:last+1] = uniform_filter1d(available[first:last+1].astype(float), width, mode='nearest')
        result[f'broadcast_available_fraction{width}'] = value
    return result


def symmetric_features(columns):
    result = {f'broadcast_sym_{key}': columns['broadcast_'+key] for key in
        ('quality', 'player_match', 'players', 'ball_seen', 'since_ball', 'ball_speed', 'ball_disagreement')}
    result['broadcast_sym_contact_any'] = np.abs(columns['broadcast_contact_team'])
    result['broadcast_sym_distance_abs'] = np.abs(columns['broadcast_team_distance'])
    for key in ('dist', 'relative', 'player_speed', 'count'):
        a, b = columns[f'broadcast_{key}_home'], columns[f'broadcast_{key}_away']
        result[f'broadcast_sym_{key}_min'] = np.fmin(a, b)
        result[f'broadcast_sym_{key}_max'] = np.fmax(a, b)
    result['broadcast_available'] = columns['broadcast_available']
    return result


def factory(family, c):
    if family == 'context_blend':
        return ContextDeadBallClassifier(C=c)
    if family == 'balanced':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(C=c, class_weight='balanced', max_iter=3000))
    if family == 'bagged':
        return BaggingClassifier(estimator=DeadBallClassifier(C=c), n_estimators=5,
                                 max_samples=.8, bootstrap=True, random_state=0, n_jobs=1)
    return DeadBallClassifier(C=c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', type=Path, default=Path(ROOT, 'src/configs/macro_v10.json'))
    parser.add_argument('--out', type=Path, default=Path(ROOT, 'cache/improve_nov17_round11'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--cases', nargs='+', help='run only these case names')
    args = parser.parse_args()
    probe = Probe(args.params, args.folds)
    i = probe.a['phase_router']['expert']
    view = probe.views[i]
    names = probe.a['members'][i]['names']
    columns = {key: view[0][key] for key in names}
    available = columns['broadcast_available'].astype(bool)
    groups = dict(baseline=columns, context=context_features(columns, available, probe.inputs.blocks),
                  symmetric=symmetric_features(columns))
    native = np.intersect1d(view[2], np.r_[np.arange(probe.public[0], probe.public[-1]+1, 25), probe.public[-1]])
    native = native[view[1][native]]
    target = probe.gt[native]
    cases = [('baseline_C0.3', 'baseline', 'logreg', .3),
             ('baseline_C0.1', 'baseline', 'logreg', .1),
             ('baseline_C1', 'baseline', 'logreg', 1.),
             ('context_C0.3', 'context', 'logreg', .3),
             ('context_C1', 'context', 'logreg', 1.),
             ('symmetric_C0.3', 'symmetric', 'logreg', .3),
             ('symmetric_C1', 'symmetric', 'logreg', 1.),
             ('balanced_C0.3', 'baseline', 'balanced', .3),
             ('bagged_C0.3', 'baseline', 'bagged', .3),
             ('context_bagged_C0.3', 'context', 'bagged', .3),
             ('context_blend_C0.3', 'context', 'context_blend', .3)]
    if args.cases:
        if set(args.cases)-{case[0] for case in cases}:
            raise ValueError('unknown case')
        cases = [case for case in cases if case[0] in args.cases]
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(folds=args.folds, public_selection=True, same_public_possessions=True,
        private_score_known=False, training_rows=len(native), purge_frames=125,
        strict_fold_edges=True, baseline=pipeline.score(probe.reference, probe.gt),
        params_sha256=broadcast.digest(args.params), model_sha256=broadcast.digest(probe.p.hybrid_model),
        source_sha256=broadcast.digest(__file__), results=[])
    np.save(args.out / 'baseline.npy', probe.reference)
    for key, group, family, c in cases:
        cols = groups[group]
        feature_names = tuple(cols)
        x = validator.matrix(cols, feature_names, native)
        models = []
        for head in range(-1, args.folds):
            train = (np.ones(len(native), bool) if head < 0 else
                ((native < probe.a['edges'][head]-125) | (native >= probe.a['edges'][head+1]+125)))
            model = factory(family, c)
            y = np.where(target[train] == 2, 2, 0) if family == 'balanced' else target[train]
            if family == 'bagged' and not np.array_equal(np.unique(y), [0, 1, 2]):
                raise ValueError('this prototype bagger requires all three classes before label encoding')
            model.fit(x[train], y)
            models.append(model)
        pack = probe.probabilities(models, cols, feature_names, view[1])
        labels = probe.render(pack, view[1], 'phase')
        if key == 'baseline_C0.3':
            np.testing.assert_array_equal(labels, probe.reference)
            np.testing.assert_array_equal(pack, probe.phase)
        result = pipeline.score(labels, probe.gt)
        public = probe.gt < 3
        truth = np.array(['H', 'A', 'D', 'X'])[probe.gt]
        before, after = probe.reference == truth, labels == truth
        row = dict(case=key, group=group, family=family, C=c, features=len(feature_names),
            macroF1=result['macro_f1'], spellF1=result['spell_f1_mean'],
            accuracy=result['accuracy'], possessions=result['n_pred_spells'], per_class=result['per_class'],
            changed_frames=int((labels != probe.reference).sum()),
            public_changes=int((public & (labels != probe.reference)).sum()),
            public_errors_fixed=int((public & ~before & after).sum()),
            public_correct_regressed=int((public & before & ~after).sum()))
        report['results'].append(row)
        joblib.dump(dict(models=models, names=feature_names, family=family, C=c,
            folds=args.folds, edges=probe.a['edges'], purge_frames=125,
            strict_fold_edges=True, training_rows=len(native), baseline_sha256=report['model_sha256']),
            args.out / (key+'.joblib'))
        np.save(args.out / (key+'.npy'), labels)
        print(json.dumps(row), flush=True)
        (args.out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
