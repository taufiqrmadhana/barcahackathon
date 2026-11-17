"""Paired audit of confidence-based short-spell pruning under fixed time folds.

Each fold is independently decoded by a head that excluded its labels. Changing
fold counts is a sensitivity check on the same 17 possessions, not new GT data.
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
from possession import hybrid, pipeline
from possession.config import Params
from possession.paths import ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--params',default=str(Path(ROOT,'src/configs/pre_pruning.json')))
    ap.add_argument('--folds',nargs='+',type=int,default=[3,4,5,6])
    ap.add_argument('--maximum',nargs='+',type=int,default=[0,25,100])
    ap.add_argument('--confidence',type=float,default=.5)
    ap.add_argument('--out',default=str(Path(ROOT,'cache/improve_nov15/fold_audit.json')))
    args=ap.parse_args()
    p=Params.load(args.params)
    inputs=pipeline.Inputs()
    base,dbg=pipeline.predict(inputs,replace(p,hybrid_model=''),debug=True)
    original=joblib.load(p.hybrid_model)
    prepared=hybrid.features(inputs,p,base,dbg['emit'],dbg['shape'],
                             original['kind'],original['legacy_params'])
    columns,mask,frames,legacy=prepared
    gt=pipeline.public_gt()
    rows=[]
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    for folds in args.folds:
        artifact=original if folds==original['folds'] else hybrid.fit(columns,frames,gt,
            original['names'],original['model_name'],original['kind'],legacy,
            n_folds=folds,gap=original['purge_frames'],train_stride=original['train_stride'],
            logreg_c=original.get('logreg_c',.3))
        before=None
        for maximum in sorted(set([0]+args.maximum)):
            pp=replace(p,hybrid_prune_max_frames=maximum,
                       hybrid_prune_min_confidence=args.confidence)
            labels=hybrid.refine(inputs,pp,base,dbg['emit'],dbg['bonus'],dbg['shape'],
                                  artifact,prepared)
            if maximum==0:
                before=labels
            np.testing.assert_array_equal(labels=='D',before=='D')
            np.testing.assert_array_equal(labels[~mask],before[~mask])
            score=pipeline.score(labels,gt)
            row=dict(folds=folds,maximum_frames=maximum,confidence=args.confidence,
                     spellF1=score['spell_f1_mean'],macroF1=score['macro_f1'],
                     public_possessions=score['n_pred_spells'],
                     changed_frames=int((labels!=before).sum()),
                     public_changed_frames=int(((labels!=before)&(gt<3)).sum()),
                     D_changed_frames=int(((labels=='D')!=(before=='D')).sum()))
            rows.append(row);print(json.dumps(row),flush=True)
            path.write_text(json.dumps(dict(same_public_possessions=True,
                selection_used_public=True,params=asdict(p),purge_frames=original['purge_frames'],
                classifier_sha256=hashlib.sha256(Path(p.hybrid_model).read_bytes()).hexdigest(),
                results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
