"""Fit and evaluate Valdi's video validator on the current Taufiq + Neo pipeline.

Scores AND the public portion of the CSV use independently decoded held-out folds.
The full public classifier is applied only outside the training/validation window.
"""
import argparse
from dataclasses import asdict, replace
import json
import hashlib
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from possession import hybrid, pipeline
from possession.config import Params
from possession.paths import ROOT
from run import parse_overrides, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params', default=str(Path(ROOT, 'src/params.json')))
    ap.add_argument('--set', nargs='+')
    ap.add_argument('--interp', default='film')
    ap.add_argument('--sets', nargs='+', choices=list(hybrid.FEATURE_SETS), default=['context','flow','film'])
    ap.add_argument('--models', nargs='+', choices=['logreg','hist_gbm','logreg_blend'], default=['logreg','hist_gbm'])
    ap.add_argument('--weights', nargs='+', type=float, default=[.25,.5,1.])
    ap.add_argument('--lock-dead', nargs='+', type=int, default=[1,0])
    ap.add_argument('--out', default=str(Path(ROOT,'cache/valdi_hybrid')))
    ap.add_argument('--name', default='valdi_hybrid')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--purge', type=int, default=125)
    ap.add_argument('--train-stride', type=int, default=25)
    ap.add_argument('--logreg-c', type=float, default=.3)
    ap.add_argument('--objective', choices=['spell','macro'], default='spell')
    args = ap.parse_args()
    p = replace(Params.load(args.params, **parse_overrides(args.set)), hybrid_model='')
    inputs = pipeline.Inputs()
    baseline, dbg = pipeline.predict(inputs, p, debug=True)
    gt = pipeline.public_gt()
    base_score = pipeline.score(baseline, gt)
    print('current baseline', flush=True); report(base_score)
    prepared = hybrid.features(inputs, p, baseline, dbg['emit'], dbg['shape'], args.interp)
    columns, mask, frames, legacy_params = prepared
    print('video samples',len(frames),'covered frames',int(mask.sum()),flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []; best = None
    for feature_set in args.sets:
        for model in args.models:
            artifact = hybrid.fit(columns, frames, gt, hybrid.FEATURE_SETS[feature_set],
                model, args.interp, legacy_params, args.folds, args.purge,args.train_stride,
                logreg_c=args.logreg_c)
            artifact['base_params'] = asdict(p)
            artifact['tracking_ball_sha256'] = hashlib.sha256(inputs.atd['ball'].tobytes()).hexdigest()
            path = out/f'{feature_set}_{model}.joblib'
            joblib.dump(artifact,path)
            for weight in args.weights:
                for lock in args.lock_dead:
                    pp = replace(p,hybrid_weight=weight,hybrid_lock_dead=lock)
                    labels = hybrid.refine(inputs, pp, baseline, dbg['emit'], dbg['bonus'],
                                           dbg['shape'],artifact,prepared)
                    result = pipeline.score(labels,gt)
                    row = dict(features=feature_set, model=model, weight=weight, lock_dead=lock,
                        spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'],
                        possessions=result['n_pred_spells'], artifact=str(path),
                        changed_frames=int((labels!=baseline).sum()))
                    row.update(kind=args.interp, objective=args.objective,
                               logreg_c=artifact['logreg_c'])
                    row['improves_both'] = (row['spellF1'] >= base_score['spell_f1_mean']
                        and row['macroF1'] >= base_score['macro_f1'])
                    rows.append(row); print(json.dumps(row),flush=True)
                    rank = ((row['macroF1'],row['spellF1']) if args.objective=='macro'
                            else (row['improves_both'],row['spellF1'],row['macroF1']))
                    if best is None or rank > best[0]:
                        best=(rank,row)
                        pipeline.write(str(Path(ROOT,'evaluation/results',args.name+'.csv')),labels)
                        (out/'best.json').write_text(json.dumps(row,indent=2)+'\n')
                    (out/'results.json').write_text(json.dumps(dict(selected_on_public=True,
                        objective=args.objective,
                        public_csv_is_out_of_fold=True,purge_frames=args.purge,
                        folds=args.folds,results=rows),indent=2)+'\n')
    print('best',json.dumps(best[1]),flush=True)


if __name__ == '__main__':
    main()
