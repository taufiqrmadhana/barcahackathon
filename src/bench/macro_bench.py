"""Compare a macro-focused checkpoint with the retained spell-focused checkpoint.

Partitions share the same public possessions. This is a paired sensitivity check,
not independent private or unseen-match validation.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from possession import hybrid,pipeline
from possession.config import Params
from possession.paths import ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',default=str(Path(ROOT,'src/configs/spell_best.json')))
    ap.add_argument('--params',default=str(Path(ROOT,'src/params.json')))
    ap.add_argument('--folds',nargs='+',type=int,default=[3,4,5,6])
    ap.add_argument('--out',default=str(Path(ROOT,'cache/improve_nov15_macro/fold_audit.json')))
    args=ap.parse_args()
    specs=[('baseline',Params.load(args.baseline)),('candidate',Params.load(args.params))]
    physical=lambda p:{k:v for k,v in asdict(p).items() if not k.startswith('hybrid_')}
    if physical(specs[0][1])!=physical(specs[1][1]):
        raise ValueError('this paired audit requires the same physical baseline')
    inputs=pipeline.Inputs();p=specs[0][1]
    base,dbg=pipeline.predict(inputs,replace(p,hybrid_model=''),debug=True)
    gt=pipeline.public_gt();ready={};rows=[]
    for name,p in specs:
        a=joblib.load(p.hybrid_model)
        prepared=hybrid.prepare_features(inputs,p,base,dbg['emit'],dbg['shape'],a)
        ready[name]=(p,a,prepared)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    for folds in args.folds:
        for name in ('baseline','candidate'):
            p,original,prepared=ready[name]
            def refit(source,view):
                if source['version'] in hybrid.ENSEMBLE_VERSIONS:
                    return hybrid.combine_artifacts([refit(m,v) for m,v in zip(source['members'],view)],
                                                    source['weights'],source.get('team_router'),
                                                    source.get('team_routers'),source.get('temporal'),
                                                    source.get('dead_router'),source.get('phase_router'))
                if folds==source['folds']:
                    return source
                columns,mask,frames,legacy=view
                return hybrid.fit(columns,frames,gt,source['names'],source['model_name'],
                    source['kind'],legacy,n_folds=folds,gap=source['purge_frames'],
                    train_stride=source['train_stride'],logreg_c=source.get('logreg_c',.3),
                    strict_fold_edges=source.get('strict_fold_edges',False))
            a=refit(original,prepared)
            mask=(np.logical_and.reduce([v[1] for v,w in zip(prepared,a['weights']) if w>0])
                  if isinstance(prepared,list)
                  else prepared[1])
            labels=hybrid.refine(inputs,p,base,dbg['emit'],dbg['bonus'],dbg['shape'],a,prepared)
            np.testing.assert_array_equal(labels[~mask],base[~mask])
            r=pipeline.score(labels,gt)
            row=dict(folds=folds,objective=name,macroF1=r['macro_f1'],spellF1=r['spell_f1_mean'],
                possessions=r['n_pred_spells'],per_class=r['per_class'])
            rows.append(row);print(json.dumps(row),flush=True)
            path.write_text(json.dumps(dict(selection_used_public=True,same_public_possessions=True,
                params={k:asdict(v[0]) for k,v in ready.items()},
                classifier_sha256={k:hashlib.sha256(Path(v[0].hybrid_model).read_bytes()).hexdigest()
                                   for k,v in ready.items()},results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
