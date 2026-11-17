"""Test Kevinza-Valdi's fusion and tree ideas with Taufiq's held-out decoder.

The physical baseline, public folds, 5 s purge and 1 Hz training rows stay fixed.
This is an adaptation audit, not a reproduction of the branch's standalone score.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import hybrid, pipeline, validator
from possession.config import Params
from possession.paths import ROOT


def loop_columns(inputs, p, debug):
    ev = inputs.evidence(p)
    columns = {}
    for field in ('ctrl', 'rel', 'second', 'crowd'):
        limit = 15. if field == 'rel' else 40.
        fill = limit if field in ('rel', 'second') else 0.
        values = np.nan_to_num(ev[field], nan=fill, posinf=limit)
        if field in ('rel', 'second'):
            values = np.clip(values, 0, limit)
        for side, name in enumerate(('home', 'away')):
            columns[field + '_' + name] = values[:, side].astype(float)
    for k, label in enumerate(('H', 'A', 'D')):
        columns['shape_' + label] = np.nan_to_num(debug['shape'][:, k], nan=1/3)
    for field in ('ball_speed_smooth', 'player_speed_smooth', 'still_run', 'missing_run'):
        columns[field] = np.nan_to_num(ev[field], nan=0.).astype(float)
        if field in ('still_run', 'missing_run'):
            columns[field] = np.minimum(columns[field], 500)
    for k, name in enumerate(('to_H', 'to_A', 'dead_in', 'H_out', 'A_out')):
        columns['bonus_' + name] = debug['bonus'][:, k]
    return columns


def tree_artifact(template, columns, frames, gt, names):
    public = np.flatnonzero(gt < 3)
    grid = np.r_[np.arange(public.min(), public.max()+1, template['train_stride']), public.max()]
    rows = np.intersect1d(frames, grid)
    x, y = validator.matrix(columns, names, rows), gt[rows]
    ids = np.clip(np.searchsorted(template['edges'], rows, side='right')-1,
                  0, template['folds']-1)
    models = []
    for fold in [-1] + list(range(template['folds'])):
        train = (np.ones(len(rows), dtype=bool) if fold < 0 else
                 validator.purged_train(rows, ids, fold, template['purge_frames']))
        model = make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features='sqrt',
            random_state=0, n_jobs=3)).fit(x[train], y[train])
        models.append(model)
    artifact = dict(template)
    artifact.update(names=names, model_name='kevin_extra', logreg_c=None,
                    full=models[0], heads=models[1:])
    return artifact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params', default=str(Path(ROOT, 'src/configs/macro_v2.json')))
    ap.add_argument('--out', default=str(Path(ROOT, 'cache/kevinza_audit')))
    args = ap.parse_args()
    p = Params.load(args.params)
    inputs = pipeline.Inputs()
    base, debug = pipeline.predict(inputs, replace(p, hybrid_model=''), debug=True)
    gt = pipeline.public_gt()
    artifact = joblib.load(p.hybrid_model)
    members = artifact['members'] if artifact['version'] == hybrid.ENSEMBLE_VERSION else [artifact]
    weights = artifact['weights'] if len(members) > 1 else [1.]
    views = [hybrid.prepare_features(inputs, p, base, debug['emit'], debug['shape'], m)
             for m in members]
    mask = np.logical_and.reduce([view[1] for view in views])
    rows = np.flatnonzero(mask)
    matrices = [validator.matrix(v[0], m['names'], rows) for v, m in zip(views, members)]
    probabilities = []
    for i in range(artifact['folds']+1):
        q = np.full((inputs.n, 3), np.nan)
        q[rows] = sum(w * validator._proba3(m['full'] if i == 0 else m['heads'][i-1], x)
                      for w, m, x in zip(weights, members, matrices))
        probabilities.append(q)
    public = np.arange(artifact['public_start'], artifact['public_stop'])
    ids = np.clip(np.searchsorted(artifact['edges'], public, side='right')-1,
                  0, artifact['folds']-1)
    columns, view_mask, frames, legacy = views[0]
    v2 = np.column_stack([columns['v2_team']/2, -columns['v2_team']/2,
                          columns['v2_dead']+np.abs(columns['v2_team'])/2])
    v2 -= v2.mean(axis=1, keepdims=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []

    def record(settings, labels):
        r = pipeline.score(labels, gt)
        row = dict(**settings, macroF1=r['macro_f1'], spellF1=r['spell_f1_mean'],
                   possessions=r['n_pred_spells'])
        results.append(row)
        print(json.dumps(row), flush=True)
        (out/'results.json').write_text(json.dumps(dict(
            source_revision='81e220d6ec81aa30fe72f4e36c4d8a3465dcb482',
            params=args.params, selected_on_public=True, public_csv_is_out_of_fold=True,
            results=results), indent=2)+'\n')

    for power, blend in [(0., 0.), (2., .15), (2., 0.), (0., .15),
                         (1., 0.), (1., .15), (2., .05)]:
        labels = None
        for i, q in enumerate(probabilities):
            delta = hybrid.probability_evidence(q, mask, p.hybrid_weight, p.hybrid_dead_weight)
            scale = (np.nan_to_num(np.clip(q, .02, .98).max(axis=1), nan=.33)/.65)**power
            # decode adds delta itself. Only add the proposed change here.
            adjustment = (scale[:, None]-1)*delta
            adjustment[mask] += blend*v2[mask]
            emit = debug['emit'] + adjustment[:, [0, 0, 1, 1, 2]]
            prediction = hybrid.decode(inputs, p, base, emit, debug['bonus'], q, mask)
            if i == 0:
                labels = prediction
            else:
                held = public[ids == i-1]
                labels[held] = prediction[held]
        record(dict(kind='fusion', confidence_power=power, v2_blend=blend), labels)

    extra = loop_columns(inputs, p, debug)
    columns = dict(columns, **extra)
    prepared = (columns, view_mask, frames, legacy)
    for name, names in [
        ('soft', hybrid.FEATURE_SETS['film_soft']),
        ('soft_loop', hybrid.FEATURE_SETS['film_soft'] + tuple(extra)),
        ('kevin', validator.V2_COLS + tuple(extra) + validator.FLOW_COLS + validator.FILM_COLS),
    ]:
        trees = tree_artifact(members[0], columns, frames, gt, names)
        labels = hybrid.refine(inputs, p, base, debug['emit'], debug['bonus'],
                               debug['shape'], trees, prepared)
        record(dict(kind='trees', features=name, n_features=len(names)), labels)


if __name__ == '__main__':
    main()
